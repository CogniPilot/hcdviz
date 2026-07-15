//! egui shell: schema status, load status, the Display toggle tree, and the selection inspector.
//! Runs in `EguiPrimaryContextPass` (bevy_egui 0.40).
// Re-exported so embedders register the shared tree panel via the same `ui::` path as `joints_panel`
// (it lives in its own module to keep the pure, unit-tested `hierarchy_rows` layout helper alongside it).
pub use crate::hierarchy::hierarchy_panel;

use crate::connectivity::{CanonicalConnectivityState, SelectedConnectivityObject};
use crate::datasheet;
use crate::display::DisplayRegistry;
use crate::doc::{HcdfDoc, SchemaStatus};
use crate::joints::{ArticulatedJoints, JointKind, JointPositions};
use crate::loop_solver::{DrivenJoint, LoopClosureStatus, LoopSolveEnabled};
use crate::open::OpenChannel;
use crate::pick::{IsolateSelection, Selected, SelectionOverrides};
use crate::scene::{CompEntity, VisualToggleGroups, ID_VISUAL, PER_LINK_KINDS};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

/// Max height (px) of an egui ComboBox popup before it scrolls. egui's default (~200 px) fits only
/// ~4 rows, and with the default auto-hiding scrollbar a long enum silently masquerades as a short
/// one: the ten-entry joint-type combo was mistaken for having four types. Sized to fit our longest
/// common enums (10 joint types; the 17 port types still scroll, now with a visible bar).
const COMBO_POPUP_MAX_HEIGHT: f32 = 420.0;

/// Tune the shared egui style for BOTH apps (dendrite_build re-hosts these panels and registers
/// [`apply_style_tuning`] itself, exactly as it re-hosts the panels): taller combo popups
/// ([`COMBO_POPUP_MAX_HEIGHT`]) and SOLID always-visible scrollbars, so a list that overflows says
/// so instead of hiding its tail behind an invisible scroll.
pub fn tune_egui_style(ctx: &egui::Context) {
    ctx.all_styles_mut(|s| {
        s.spacing.combo_height = COMBO_POPUP_MAX_HEIGHT;
        s.spacing.scroll = egui::style::ScrollStyle::solid();
    });
}

/// Apply [`tune_egui_style`] once per app run. Guarded by a `Local` (not a Startup system) because
/// the egui context is only usable inside the egui pass; idempotent thereafter so a future
/// user-driven style change is never fought frame-by-frame.
pub fn apply_style_tuning(mut contexts: EguiContexts, mut done: Local<bool>) {
    if *done {
        return;
    }
    if let Ok(ctx) = contexts.ctx_mut() {
        tune_egui_style(ctx);
        *done = true;
    }
}

/// Default half-range for continuous / limitless movable joints (radians); sliders span ±this.
const LIMITLESS_HALF_RANGE: f32 = std::f32::consts::PI;

/// Largest |bound| the Joints slider still honours as a real range (rad or m). SDF-derived robots
/// author "revolute" joints whose limits are SENTINELS standing in for "unbounded" (±1e16, or ±f64::MAX)
/// rather than declaring `continuous`; a slider literally spanning ±1e16 is unusable (any pixel of travel
/// jumps astronomically). Bounds beyond this threshold make the SLIDER fall back to the same ±π range a
/// continuous joint gets, DISPLAY-ONLY: the stored limits are untouched and [`crate::joints::resolve_q`]
/// still clamps against the real values (a no-op inside the slider's range).
const SLIDER_LIMIT_MAX_ABS: f32 = 1e3;

/// The slider range for one movable joint: the declared `lower..=upper` when the joint is a bounded kind
/// with both bounds present, ordered, and SANE (each |bound| ≤ [`SLIDER_LIMIT_MAX_ABS`]); otherwise the
/// ±[`LIMITLESS_HALF_RANGE`] fallback (continuous joints always, and revolute/prismatic whose limits are
/// missing, inverted, or unbounded-sentinel huge). Pure, so the sentinel-limit fallback is unit-testable.
pub(crate) fn joint_slider_range(
    kind: JointKind,
    lower: Option<f32>,
    upper: Option<f32>,
) -> std::ops::RangeInclusive<f32> {
    match (kind, lower, upper) {
        (JointKind::Continuous, _, _) => -LIMITLESS_HALF_RANGE..=LIMITLESS_HALF_RANGE,
        (_, Some(lo), Some(hi))
            if lo <= hi && lo.abs() <= SLIDER_LIMIT_MAX_ABS && hi.abs() <= SLIDER_LIMIT_MAX_ABS =>
        {
            lo..=hi
        }
        _ => -LIMITLESS_HALF_RANGE..=LIMITLESS_HALF_RANGE,
    }
}

/// The mutable display/selection state the panels write, grouped into one `SystemParam` so `panels`
/// stays within the argument budget (idiomatic Bevy; mirrors `scene::SceneAssets`, no lint suppression).
#[derive(bevy::ecs::system::SystemParam)]
pub struct DisplayControls<'w> {
    registry: ResMut<'w, DisplayRegistry>,
    isolate: ResMut<'w, IsolateSelection>,
    overrides: ResMut<'w, SelectionOverrides>,
    shininess: ResMut<'w, crate::scene::RenderShininess>,
    toggle_groups: ResMut<'w, VisualToggleGroups>,
    /// Comms overlay: the resolved graph (read) + per-network visibility / selection / isolate (write).
    network_overlay: Res<'w, crate::network::NetworkOverlayScene>,
    network_overrides: ResMut<'w, crate::network::NetworkVizOverrides>,
    selected_network: ResMut<'w, crate::network::SelectedNetwork>,
    network_isolate: ResMut<'w, crate::network::IsolateNetwork>,
    connectivity_state: Res<'w, CanonicalConnectivityState>,
    selected_connectivity: ResMut<'w, SelectedConnectivityObject>,
}

/// The mutable joint / loop-closure state [`joints_panel`] writes, grouped into one `SystemParam` so
/// the panel stays within the argument budget (mirrors [`DisplayControls`]; no lint suppression).
/// `positions` is the commanded-pose source of truth the sliders + the state picker write; `driven`
/// records the solver-held joint; `solve_enabled` + `loop_status` back the loop-closure controls.
#[derive(bevy::ecs::system::SystemParam)]
pub struct JointDrive<'w> {
    positions: ResMut<'w, JointPositions>,
    driven: ResMut<'w, DrivenJoint>,
    solve_enabled: ResMut<'w, LoopSolveEnabled>,
    loop_status: Res<'w, LoopClosureStatus>,
}

pub fn panels(
    mut contexts: EguiContexts,
    mut status: ResMut<SchemaStatus>,
    doc: Res<HcdfDoc>,
    selected: Res<Selected>,
    comps: Query<&CompEntity>,
    open_channel: Res<OpenChannel>,
    mut controls: DisplayControls,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("hcdviz").default_width(280.0).show(ctx, |ui| {
        // Runtime file-open: lets the browser viewer (which starts empty, no CLI arg on wasm) load a
        // document, and is a harmless addition on native. A `.hcdfz` bundle is the self-contained,
        // mesh-bearing choice; a plain `.hcdf` loads structure (meshes need their bytes already in store).
        if ui.button("Open .hcdf / .hcdfz…").clicked() {
            crate::open::request_open(&open_channel);
        }
        ui.separator();

        ui.heading("Schema");
        ui.label(format!("pin verified: {}", status.pin_ok));
        let sha = status.xsd_sha.get(..16).unwrap_or(&status.xsd_sha);
        ui.label(format!("hcdf.xsd 1.0 sha: {sha}…"));

        ui.separator();
        ui.heading("Document");
        ui.label(&status.message);
        // Warn-on-open (set once per open by `doc::load_hcdf_system`): the opened doc's RAW bytes are
        // not schema-valid, though the viewer shows whatever parsed. Amber (the same informational-
        // warning color dendrite_build uses for include-sha drift) and dismissible, because the
        // viewer is read-only and there is nothing to fix here.
        if let Some(warning) = status.open_warning.clone() {
            ui.horizontal_wrapped(|ui| {
                if ui.small_button("✕").on_hover_text("Dismiss").clicked() {
                    status.open_warning = None;
                }
                ui.colored_label(egui::Color32::from_rgb(220, 160, 40), warning);
            });
        }

        ui.separator();
        ui.heading("Displays");
        let items: Vec<(&'static str, String, bool)> = controls
            .registry
            .entries()
            .iter()
            .map(|e| (e.id, e.label.clone(), controls.registry.enabled(e.id)))
            .collect();
        for (id, label, mut on) in items {
            if ui.checkbox(&mut on, label).changed() {
                controls.registry.set_enabled(id, on);
            }
            // Visual toggle groups (`<visual toggle="…">`, legacy per-group show/hide, e.g. `case`
            // over a bare PCB): one checkbox per group in the doc, nested under Visual. Checked =
            // shown; all groups reset to shown on every doc load. Absent when the doc has none.
            // Edited via targeted insert/remove so the resource's change tick fires only on a real
            // flip; it is what re-triggers `scene::sync_visual_visibility`.
            if id == ID_VISUAL && !controls.toggle_groups.groups.is_empty() {
                ui.indent("visual-toggle-groups", |ui| {
                    ui.label("Toggle groups");
                    let names = controls.toggle_groups.groups.clone();
                    for name in names {
                        let mut shown = !controls.toggle_groups.hidden.contains(&name);
                        if ui.checkbox(&mut shown, &name).changed() {
                            if shown {
                                controls.toggle_groups.hidden.remove(&name);
                            } else {
                                controls.toggle_groups.hidden.insert(name.clone());
                            }
                        }
                    }
                });
            }
        }
        // Opt-in metallic sheen (RViz-style look); OFF = the faithful matte the bake produced. Edited on
        // a local copy so the resource is marked changed only on a real toggle; its change tick is what
        // gates `scene::apply_shininess`.
        let mut shiny = controls.shininess.0;
        if ui
            .checkbox(&mut shiny, "Render shininess")
            .on_hover_text("Add a metallic/specular sheen to mesh materials (viewer look; baked materials are matte)")
            .changed()
        {
            controls.shininess.0 = shiny;
        }

        // Networks: the comms overlay list, per-network show/hide + click-to-select, plus an
        // isolate-to-network toggle. Present only when the doc actually models a `<network>`; the edges
        // themselves render only while the "Networks (comms overlay)" display above is enabled.
        let networks = controls.network_overlay.networks();
        if !networks.is_empty() {
            ui.separator();
            egui::CollapsingHeader::new("Networks")
                .default_open(false)
                .show(ui, |ui| {
                    for network in &networks {
                        ui.horizontal(|ui| {
                            let mut shown = controls.network_overrides.visible(&network.id);
                            if ui.checkbox(&mut shown, "").changed() {
                                controls
                                    .network_overrides
                                    .0
                                    .insert(network.id.clone(), shown);
                            }
                            let sel = controls.selected_network.0.as_ref() == Some(&network.id);
                            if ui.selectable_label(sel, &network.label).clicked() {
                                // Toggle selection off if it was already the selected one.
                                controls.selected_network.0 = if sel {
                                    None
                                } else {
                                    Some(network.id.clone())
                                };
                            }
                        });
                    }
                    let mut iso = controls.network_isolate.0;
                    if ui
                        .checkbox(&mut iso, "Isolate to selected network")
                        .changed()
                    {
                        controls.network_isolate.0 = iso;
                    }
                });
        }

        if let Some(id) = controls.selected_connectivity.0.as_ref() {
            ui.separator();
            ui.heading("Connectivity selection");
            if let Some(node) = controls
                .connectivity_state
                .graph()
                .and_then(|graph| graph.node(id))
            {
                let name = node
                    .identity()
                    .local()
                    .last()
                    .map(|part| part.value.as_str())
                    .unwrap_or("unnamed");
                let instance = connectivity_instance_qualifier(node.identity())
                    .map_or_else(String::new, |instance| format!("{instance} "));
                ui.strong(format!("{}: {instance}{name}", node.kind().as_str()));
                for (label, value) in connectivity_detail_lines(node) {
                    ui.label(format!("{label}: {value}"));
                }
            } else {
                ui.label("The selected object is not present in the accepted connectivity graph.");
            }
            if ui.small_button("Clear connectivity selection").clicked() {
                controls.selected_connectivity.0 = None;
            }
        }
    });

    // Inspector: structural HCDF fields of the selected component (read-only). Connectivity details
    // remain on the canonical graph boundary and are not reconstructed from the structural model.
    if let Some(entity) = selected.0 {
        if let (Ok(ce), Some(h)) = (comps.get(entity), doc.0.as_ref()) {
            if let Some(comp) = h.comp.get(ce.comp_index) {
                egui::Window::new("Inspector")
                    .default_width(300.0)
                    .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
                    .show(ctx, |ui| {
                        ui.heading(&comp.name);
                        // Isolate selection: render only this comp's items (still honoring the global
                        // kind-toggles). Bound directly to the resource; reverts on deselect because
                        // isolate is a no-op when nothing is selected.
                        let mut on = controls.isolate.0;
                        if ui.checkbox(&mut on, "Isolate (show only this)").changed() {
                            controls.isolate.0 = on;
                        }
                        // Per-link display overrides for THIS comp only. Each checkbox's shown state is
                        // the override if set, else the live global toggle; changing it records an
                        // override. These clear automatically on deselect (SelectionOverrides resets when
                        // the selection changes), so nothing persists to other comps or past Esc.
                        ui.separator();
                        ui.label("This link");
                        for (id, label) in PER_LINK_KINDS {
                            let mut k = controls
                                .overrides
                                .kinds
                                .get(id)
                                .copied()
                                .unwrap_or(controls.registry.enabled(id));
                            if ui.checkbox(&mut k, label).changed() {
                                controls.overrides.kinds.insert(id, k);
                            }
                        }
                        if let Some(role) = &comp.role {
                            ui.label(format!("role: {role}"));
                        }
                        if let Some(board) = &comp.board {
                            ui.label(format!("board: {board}"));
                        }
                        if let Some(hwid) = &comp.hwid {
                            ui.label(format!("hwid: {hwid}"));
                        }
                        if let Some(d) = &comp.description {
                            ui.label(d);
                        }
                        ui.separator();
                        if let Some(inertial) = &comp.inertial {
                            if let Some(m) = &inertial.mass {
                                ui.label(format!("mass: {m}"));
                            }
                        }
                        ui.label(format!("visuals: {}", comp.visual.len()));
                        ui.label(format!("collisions: {}", comp.collision.len()));
                        ui.label(format!("frames: {}", comp.frame.len()));
                        ui.label(format!("sensors: {}", comp.sensor.len()));
                        // Datasheet detail: read-only spec sections for the comp's motor(s), power
                        // source(s), HMI element(s), dynamic surface(s), and firmware identity. Each
                        // renders only the fields the doc authors ([`crate::datasheet`] omits absent
                        // leaves), so an unpopulated component adds no clutter: a motor with no specs
                        // still shows its header (its presence is itself information), a comp with no
                        // motor shows no motor section at all.
                        for m in &comp.motor {
                            datasheet_section(
                                ui,
                                m.name.as_deref(),
                                "Motor",
                                datasheet::motor_lines(m),
                            );
                        }
                        for ps in &comp.power_source {
                            datasheet_section(
                                ui,
                                ps.name.as_deref(),
                                "Power source",
                                datasheet::power_source_lines(ps),
                            );
                        }
                        for hmi in &comp.hmi {
                            datasheet_section(
                                ui,
                                hmi.name.as_deref(),
                                "HMI",
                                datasheet::hmi_lines(hmi),
                            );
                        }
                        for ds in &comp.dynamic_surface {
                            datasheet_section(
                                ui,
                                ds.name.as_deref(),
                                "Dynamic surface",
                                datasheet::dynamic_surface_lines(ds),
                            );
                        }
                        if let Some(sw) = &comp.software {
                            datasheet_section(
                                ui,
                                sw.name.as_deref(),
                                "Software",
                                datasheet::software_lines(sw),
                            );
                        }
                        if !comp.extension.is_empty() {
                            ui.separator();
                            ui.label(format!("extensions: {}", comp.extension.len()));
                        }
                    });
            }
        }
    }
    Ok(())
}

/// Compact read-only metadata for the currently selected canonical connectivity object.
pub fn connectivity_detail_lines(
    node: &crate::schema::connectivity::ConnectivityNode,
) -> Vec<(&'static str, String)> {
    use crate::schema::connectivity::ConnectivityNodeData;

    let identity = node
        .identity()
        .local()
        .iter()
        .map(|part| format!("{}={}", part.field, part.value))
        .collect::<Vec<_>>()
        .join(" / ");
    let mut lines = Vec::new();
    if let Some(instance) = connectivity_instance_qualifier(node.identity()) {
        lines.push(("instance", instance));
    }
    lines.push(("identity", identity));
    match node.data() {
        ConnectivityNodeData::Port { capabilities } => {
            append_capability_lines(&mut lines, capabilities);
        }
        ConnectivityNodeData::Channel {
            capabilities,
            role,
            local_group,
        } => {
            if let Some(role) = role {
                lines.push(("role", role.to_string()));
            }
            if let Some(group) = local_group {
                lines.push(("local group", group.clone()));
            }
            append_capability_lines(&mut lines, capabilities);
        }
        ConnectivityNodeData::Connector {
            family: Some(family),
        } => lines.push(("family", family.to_string())),
        ConnectivityNodeData::Connector { family: None } => {}
        ConnectivityNodeData::Position {
            kind,
            role,
            local_group,
        } => {
            lines.push(("position kind", enum_label(kind)));
            if let Some(role) = role {
                lines.push(("role", role.to_string()));
            }
            if let Some(group) = local_group {
                lines.push(("local group", group.clone()));
            }
        }
        _ => {}
    }
    lines
}

pub(crate) fn connectivity_instance_qualifier(
    identity: &crate::schema::connectivity::ObjectIdentity,
) -> Option<String> {
    (!identity.instance().is_root()).then(|| format!("[{}]", identity.instance().display_path()))
}

fn append_capability_lines(
    lines: &mut Vec<(&'static str, String)>,
    capabilities: &crate::schema::model::connectivity::Capabilities,
) {
    if !capabilities.purposes.is_empty() {
        lines.push((
            "purposes",
            capabilities
                .purposes
                .iter()
                .map(enum_label)
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    if !capabilities.carriers.is_empty() {
        lines.push((
            "carriers",
            capabilities
                .carriers
                .iter()
                .map(enum_label)
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    if !capabilities.profiles.is_empty() {
        lines.push((
            "profiles",
            capabilities
                .profiles
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    let limits = &capabilities.limits;
    let supported_axes = [
        ("rate", limits.rate.is_some()),
        ("voltage", limits.voltage.is_some()),
        ("current", limits.current.is_some()),
        ("power", limits.power.is_some()),
        ("impedance", limits.impedance.is_some()),
        ("frequency", limits.frequency.is_some()),
        ("bandwidth", limits.bandwidth.is_some()),
        ("pressure", limits.pressure.is_some()),
        ("flow", limits.flow.is_some()),
        ("temperature", limits.temperature.is_some()),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect::<Vec<_>>();
    if !supported_axes.is_empty() {
        lines.push(("capability limits", supported_axes.join(", ")));
    }
}

fn enum_label(value: &impl std::fmt::Debug) -> String {
    camel_case_to_kebab(&format!("{value:?}"))
}

/// Convert a Rust enum variant name to the schema's kebab-case spelling.
pub(crate) fn camel_case_to_kebab(value: &str) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(value.len() + 4);
    for (index, ch) in chars.iter().copied().enumerate() {
        if ch.is_ascii_uppercase()
            && index > 0
            && (chars[index - 1].is_ascii_lowercase()
                || chars[index - 1].is_ascii_digit()
                || chars
                    .get(index + 1)
                    .is_some_and(|next| next.is_ascii_lowercase()))
        {
            out.push('-');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// Render one read-only datasheet section in the Inspector: a separator, a bold `Kind: name`
/// header (or a bare `Kind` when the element is unnamed), then one `label: value` line per authored
/// field. The section shows even when `lines` is empty: the element's PRESENCE is itself information
/// (a motor with no authored specs is still a motor), so the caller renders one section per element
/// unconditionally and lets [`crate::datasheet`] decide which field rows exist.
fn datasheet_section(
    ui: &mut egui::Ui,
    name: Option<&str>,
    kind: &str,
    lines: Vec<(&'static str, String)>,
) {
    ui.separator();
    let header = match name.map(str::trim).filter(|n| !n.is_empty()) {
        Some(n) => format!("{kind}: {n}"),
        None => kind.to_string(),
    };
    ui.strong(header);
    for (label, value) in lines {
        ui.label(format!("{label}: {value}"));
    }
}

/// Draw the transmission-endpoint badge for `joint_name` when a `<transmission>` drives it: a
/// small teal line (name + reduction + spring/clutch tag) beneath the joint, so a motor-driven joint
/// reads as such at a glance. A pure `doc.transmission` lookup ([`crate::datasheet::transmission_badge`]);
/// a no-op when the joint is not a transmission endpoint, so it is safe to call after every joint row.
fn transmission_badge_line(
    ui: &mut egui::Ui,
    transmissions: &[crate::schema::model::Transmission],
    joint_name: &str,
) {
    if let Some(badge) = datasheet::transmission_badge(transmissions, joint_name) {
        ui.colored_label(egui::Color32::from_rgb(120, 190, 210), badge.summary());
    }
}

/// Render ONE DOF of a joint as a slider, writing the edited coordinate back into [`JointPositions`]
/// only when it actually moves (so the articulation system fires on real edits, not every draw).
///
/// `dof` selects which coordinate (0 for single-DOF joints). `label` overrides the slider text: `None`
/// uses the joint's own name (the 1-DOF standalone case); `Some(l)` uses a short DOF label (a multi-DOF
/// sub-slider). `sel` is `Some` only for the 1-DOF embedder path, turning the name into a clickable
/// selector beside a bare slider (matching the pre-multi-DOF layout exactly).
///
/// A REAL slider change also records this joint as the loop solver's [`DrivenJoint`]: the joint the
/// solver holds fixed while the passive loop members close around it. Only a user edit writes it: the
/// solver's write-backs land straight in [`JointPositions`], never pass through here, and therefore can
/// never usurp the driver.
fn joint_dof_slider(
    ui: &mut egui::Ui,
    positions: &mut ResMut<JointPositions>,
    driven: &mut ResMut<DrivenJoint>,
    sel: Option<&mut crate::pick::SelectedJoint>,
    joint: &crate::joints::ArticulatedJoint,
    dof: usize,
    label: Option<&str>,
) {
    // Range: the declared [lower, upper] for sanely-limited DOFs, else a symmetric ±π default for
    // continuous / limitless / sentinel-limit ones (see joint_slider_range).
    let range = joint_slider_range(joint.kind, joint.lower[dof], joint.upper[dof]);
    // Edit a local copy seeded from the current command; only write back on a real change.
    let mut q = positions.dof(&joint.name, dof);
    match sel {
        // WIRED (embedder), single-DOF: the name is a clickable selector; the slider drops its inline
        // text label (the selectable name IS the label) and sits beside it.
        Some(sel) => {
            ui.horizontal(|ui| {
                let on = sel.0.as_deref() == Some(joint.name.as_str());
                if ui.selectable_label(on, &joint.name).clicked() {
                    sel.0 = Some(joint.name.clone());
                }
                if ui.add(egui::Slider::new(&mut q, range)).changed() {
                    positions.set_dof(&joint.name, dof, q);
                    set_driven(driven, &joint.name);
                }
            });
        }
        // Standalone single-DOF (name label) or a multi-DOF sub-slider (short DOF label).
        None => {
            let text = label.unwrap_or(joint.name.as_str());
            if ui
                .add(egui::Slider::new(&mut q, range).text(text))
                .changed()
            {
                positions.set_dof(&joint.name, dof, q);
                set_driven(driven, &joint.name);
            }
        }
    }
}

/// Record `name` as the solver-held driver, touching the resource only on an actual handoff (dragging
/// the same slider frame after frame must not re-dirty it every frame).
fn set_driven(driven: &mut ResMut<DrivenJoint>, name: &str) {
    if driven.0.as_deref() != Some(name) {
        driven.0 = Some(name.to_string());
    }
}

/// The "Joints" panel: one slider per MOVABLE 1-DOF joint (revolute/continuous/prismatic/screw), and a
/// labelled slider GROUP per multi-DOF joint (cylindrical/universal/planar/ball, one slider per DOF),
/// excluding mimic-driven and fixed/other, bounded by each DOF's limits, plus a Reset-to-zero button.
///
/// Editing a slider writes the joint's commanded `q` into [`JointPositions`]: the single
/// source of truth the articulation system reacts to (a future topic listener writes the same map).
/// A separate `egui::Window` anchored bottom-left, collapsible, so it stays out of the hcdviz panel and
/// the Inspector. Slider range: `joint_slider_range` gives `lower..=upper` for sanely-limited
/// revolute/prismatic joints, else ±π for continuous/limitless joints AND for bounds beyond
/// `SLIDER_LIMIT_MAX_ABS` (SDF "revolute standing in for continuous" sentinels like ±1e16). q is still
/// stored unclamped; [`crate::joints::resolve_q`] applies the real limit on apply, so continuous joints
/// wrap freely beyond the visible range.
///
/// SELECTION HOOK (embedder-only). The optional [`crate::pick::SelectedJoint`] out-param is `None` in the
/// standalone viewer (the resource is never registered), keeping this panel byte-identical there. When an
/// embedder (dendrite_build) `init_resource`s it, the panel becomes a joint PICKER: each slider's name
/// turns into a `selectable_label` that writes the clicked joint's NAME into `SelectedJoint`, and a second
/// section lists every NON-movable catalogued joint (fixed/ball/universal/…) as a clickable name too, so
/// the embedder's Inspector can XML-edit ANY tree-edge joint, not just the movable ones the sliders cover.
///
/// LOOP CLOSURES. When the doc has loop closures ([`LoopClosureStatus`] non-empty, one entry per
/// closure, kept even while solving is off), the panel grows a "Solve loop closures" checkbox bound to
/// [`LoopSolveEnabled`] plus a warning line per closure the solver could NOT close ("open by X mm / Y°").
/// dendrite_build re-hosts this exact function, so the loop UX appears in both apps for free. Slider
/// edits also record the touched joint into [`DrivenJoint`]: the joint the solver holds while the
/// passive loop members close around it.
pub fn joints_panel(
    mut contexts: EguiContexts,
    doc: Res<HcdfDoc>,
    joints: Res<ArticulatedJoints>,
    mut drive: JointDrive,
    // `None` in the standalone viewer (resource unregistered ⇒ Bevy yields `None`), `Some` in an embedder
    // that registers `SelectedJoint`. Gates every selection behaviour below without changing any call site.
    mut sel_joint: Option<ResMut<crate::pick::SelectedJoint>>,
    // The last state picked from the dropdown, kept per-panel so the combo shows the current selection.
    // Each doc load repopulates the list; a stale name simply never matches and the combo reads "(select)".
    mut sel_state: Local<String>,
) -> Result {
    // Only movable, directly driven joints get a slider; mimic joints follow their source.
    let movable: Vec<usize> = joints
        .0
        .iter()
        .enumerate()
        .filter(|(_, j)| j.kind.is_movable() && j.mimic.is_none() && !j.name.is_empty())
        .map(|(i, _)| i)
        .collect();
    // Standalone (no selection wired): keep the old early-out so an unarticulated model shows no panel.
    // When selection IS wired the panel stays useful even with zero movable joints; it can still list the
    // fixed/other joints to select, so only bail when nothing at all would render.
    if movable.is_empty() && sel_joint.is_none() {
        return Ok(());
    }
    let ctx = contexts.ctx_mut()?;

    // Cap the window to a fraction of the viewport so a densely-articulated robot (~130 perseverance joints)
    // can't grow the panel off-screen: the Reset button stays PINNED and the sliders + other-joints list
    // scroll inside a fill-height ScrollArea below it. The window is anchored LEFT_BOTTOM and grows upward,
    // so the default height must not exceed the viewport; 40% keeps it on-screen while leaving the window
    // resizable (egui remembers the dragged size for the session, so the content follows a resize).
    let default_h = (ctx.content_rect().height() * 0.4).max(160.0);
    egui::Window::new("Joints")
        .default_width(280.0)
        .default_height(default_h)
        .default_open(false)
        .anchor(egui::Align2::LEFT_BOTTOM, [8.0, -8.0])
        .show(ctx, |ui| {
            if ui.button("Reset").clicked() {
                // Zero every commanded position. resolve_q clamps each into range on apply, so a zero
                // that falls outside a joint's limits still lands at the nearest valid bound. Also
                // forget the driven joint: a reset returns to the load state, where every loop joint
                // is free and the solver re-assembles from the zero pose held by nothing.
                drive.positions.0.clear();
                if drive.driven.0.is_some() {
                    drive.driven.0 = None;
                }
            }
            // States: snap the whole robot to a named `<state>` pose. Present only when the doc authors
            // `<state>` elements. Selecting one seeds JointPositions from that state (mimic/limits are
            // resolved downstream by `articulate`); the default state is already applied on load by
            // `reset_on_reload`, so this is the manual "jump to a named pose" control.
            if let Some(h) = doc.0.as_deref() {
                let state_names: Vec<&str> =
                    h.state.iter().filter_map(|s| s.name.as_deref()).collect();
                if !state_names.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label("State:");
                        egui::ComboBox::from_id_salt("hcdviz-state-picker")
                            .selected_text(if sel_state.is_empty() {
                                "(select)"
                            } else {
                                sel_state.as_str()
                            })
                            .show_ui(ui, |ui| {
                                for name in &state_names {
                                    if ui
                                        .selectable_label(sel_state.as_str() == *name, *name)
                                        .clicked()
                                    {
                                        *sel_state = name.to_string();
                                        if let Some(st) = h
                                            .state
                                            .iter()
                                            .find(|s| s.name.as_deref() == Some(*name))
                                        {
                                            crate::joints::apply_state(&mut drive.positions, st);
                                            // Applying a named state defines a fresh pose context, so
                                            // forget the driven joint: the loop solver re-assembles the
                                            // passive members around the new pose, held by nothing.
                                            if drive.driven.0.is_some() {
                                                drive.driven.0 = None;
                                            }
                                        }
                                    }
                                }
                            });
                    });
                }
            }
            // Loop-closure controls, present only when the doc actually HAS closures (the status
            // resource keeps one entry per closure even while solving is disabled, precisely so this
            // checkbox stays reachable to turn it back on). Edited on a local copy so the resource is
            // marked changed only on a real toggle; its change tick re-triggers the solver.
            if !drive.loop_status.0.is_empty() {
                let mut on = drive.solve_enabled.0;
                if ui
                    .checkbox(&mut on, "Solve loop closures")
                    .on_hover_text(
                        "Adjust the passive loop-member joints so every loop closure stays \
                         assembled; off = draw the constraint links without enforcing them",
                    )
                    .changed()
                {
                    drive.solve_enabled.0 = on;
                }
                // One warning line per closure the solver could NOT close (limits/geometry): the
                // same warning-red as the constraint gizmo, split into the units a mechanism
                // designer thinks in.
                for c in &drive.loop_status.0 {
                    if let Some(e) = c.error.filter(|e| e.open()) {
                        ui.colored_label(
                            egui::Color32::from_rgb(240, 60, 40),
                            format!(
                                "loop '{}' open by {:.2} mm / {:.2}°",
                                c.name,
                                e.trans * 1000.0,
                                e.rot.to_degrees()
                            ),
                        );
                    }
                }
            }
            ui.separator();
            // Transmission awareness: a joint driven by a `<transmission>` gets a badge under its
            // slider (name + reduction, spring/clutch tag). A pure `doc.transmission` -> `<joint ref=>`
            // lookup, so absent a transmission section this slice is empty and no joint ever badges.
            let transmissions = doc
                .0
                .as_deref()
                .map(|h| h.transmission.as_slice())
                .unwrap_or_default();
            // Scroll the body (sliders + other-joints list) so the window height is the only bound on how
            // far it grows; fill the available height so a window resize grows the scroll region, not just
            // empty space (mirrors the hierarchy/inspector fill-height feel).
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for &i in &movable {
                        let j = &joints.0[i];
                        // 1-DOF joints render exactly as before (name-labelled single slider); multi-DOF
                        // joints render as a labelled group: the joint name as a header, then one slider
                        // per DOF under a short DOF label. Reset (above) zeroes every DOF of every joint.
                        if j.kind.dof_count() <= 1 {
                            joint_dof_slider(
                                ui,
                                &mut drive.positions,
                                &mut drive.driven,
                                sel_joint.as_deref_mut(),
                                j,
                                0,
                                None,
                            );
                        } else {
                            // Header: selectable name in the embedder (a joint PICKER), plain label standalone.
                            match sel_joint.as_deref_mut() {
                                Some(sel) => {
                                    let on = sel.0.as_deref() == Some(j.name.as_str());
                                    if ui.selectable_label(on, &j.name).clicked() {
                                        sel.0 = Some(j.name.clone());
                                    }
                                }
                                None => {
                                    ui.label(&j.name);
                                }
                            }
                            let labels = j.kind.dof_labels();
                            ui.indent(j.name.as_str(), |ui| {
                                for d in 0..j.kind.dof_count() {
                                    let label = labels.get(d).copied().unwrap_or("");
                                    // The header already carries the name/selection; each sub-slider just
                                    // takes its DOF label and never re-emits the name selector.
                                    joint_dof_slider(
                                        ui,
                                        &mut drive.positions,
                                        &mut drive.driven,
                                        None,
                                        j,
                                        d,
                                        Some(label),
                                    );
                                }
                            });
                        }
                        // A transmission driving this joint badges under its slider(s).
                        transmission_badge_line(ui, transmissions, &j.name);
                    }
                    // WIRED only: also list the NON-movable catalogued joints (fixed/ball/universal/planar/…, and
                    // mimic-driven movable ones) as clickable names with no slider, so EVERY named tree-edge joint
                    // the scene spawned is reachable for XML editing, not just the movable ones the sliders cover.
                    if let Some(sel) = sel_joint.as_deref_mut() {
                        let others: Vec<usize> = joints
                            .0
                            .iter()
                            .enumerate()
                            .filter(|(_, j)| {
                                !(j.name.is_empty() || j.kind.is_movable() && j.mimic.is_none())
                            })
                            .map(|(i, _)| i)
                            .collect();
                        if !others.is_empty() {
                            ui.separator();
                            ui.label("other joints");
                            for &i in &others {
                                let name = &joints.0[i].name;
                                let on = sel.0.as_deref() == Some(name.as_str());
                                if ui.selectable_label(on, name).clicked() {
                                    sel.0 = Some(name.clone());
                                }
                                // A non-movable joint can still be a transmission endpoint, so badge it too.
                                transmission_badge_line(ui, transmissions, name);
                            }
                        }
                    }
                });
        });

    Ok(())
}
