//! Picking + selection, modernized from dendrite-viewer's `on_device_clicked`.
//!
//! In Bevy 0.19 the mesh picking backend makes **every** mesh with `RenderAssetUsages::MAIN_WORLD`
//! pickable by default: `MeshPickingSettings::require_markers` is `false`, so no `Pickable` marker is
//! needed on the meshes or the camera (verified against the installed bevy_picking 0.19 source). glTF
//! meshes load with `RenderAssetUsages::default()` (which includes `MAIN_WORLD`), so the async-spawned
//! visual subtree is pickable without extra wiring. We set `require_markers: false` explicitly anyway
//! so the behaviour is robust against any embedder that flips the default.
//!
//! A left click walks up the `ChildOf` hierarchy from the hit entity to the owning [`CompEntity`] and
//! records it in the [`Selected`] resource, which the inspector panel and highlight gizmo read. The
//! glTF mesh is a descendant of the per-visual `WorldAssetRoot` entity, which is itself a child of the
//! `CompEntity`, so the ancestor walk reaches the comp. Bevy 0.19 picking uses the `On<Pointer<Click>>`
//! observer with `event.entity` / `event.button` (the 0.17 `Pointer.target` field was renamed `entity`).
use bevy::picking::mesh_picking::{MeshPickingPlugin, MeshPickingSettings};
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;

use crate::scene::CompEntity;

/// The currently selected comp entity (if any); drives highlight + inspector.
#[derive(Resource, Default)]
pub struct Selected(pub Option<Entity>);

/// Extra highlight boxes drawn ALONGSIDE the plain yellow [`Selected`] gizmo, each in its own color.
/// Empty (the default) reproduces exactly the single-selection highlight: [`crate::scene::draw_highlight`]
/// always draws [`Selected`] in yellow first, then one bounds box per entry here. The standalone viewer
/// never populates it (so its highlight stays byte-identical); the dendrite_build editor writes it to
/// paint a selected joint's parent/child comps (parent orange, child cyan) with the same bounds math.
#[derive(Resource, Default)]
pub struct HighlightSet(pub Vec<(Entity, Color)>);

/// The joint the embedder's inspector is editing, keyed by joint NAME (names are the stable,
/// rebuild-surviving identity: entities churn on every scene rebuild, and joint XML edits resolve by
/// name). The standalone viewer never registers this resource, so the shared [`crate::ui::joints_panel`]
/// sees `None` there and leaves its behaviour byte-identical; the dendrite_build editor
/// `init_resource`s it and drives its Inspector from it.
#[derive(Resource, Default)]
pub struct SelectedJoint(pub Option<String>);

/// "Isolate selection" mode: when ON *and* a comp is [`Selected`], the displays render only the
/// selected comp's own toggleable items (still honoring the global kind-toggles). When OFF (or when
/// nothing is selected) every display behaves exactly as before. Toggled from the inspector panel.
#[derive(Resource, Default)]
pub struct IsolateSelection(pub bool);

/// Comp-SET isolate: generalizes [`IsolateSelection`] to a set of comps. When `Some(set)`, an item is
/// visible only if its owner comp is IN the set (global kind toggles still apply); a kinematic tree edge
/// stays visible only when BOTH endpoints are in the set. When `None` (the default) every display falls
/// back to the single-[`Selected`] [`IsolateSelection`] path, byte-identical to before. The standalone
/// viewer never populates it; the dendrite_build editor writes {parent, child} here to isolate a selected
/// joint's comps (an empty set therefore hides everything, an intentional "isolate to nothing" state,
/// distinct from `None`).
#[derive(Resource, Default)]
pub struct IsolateSet(pub Option<bevy::platform::collections::HashSet<Entity>>);

/// Per-link display overrides for the currently [`Selected`] comp only (the inspector's "This link"
/// section). `kinds` maps a display id (e.g. `ID_VISUAL`) to a forced on/off that wins over the global
/// `DisplayRegistry` entry, but ONLY for items owned by `comp`. Anything not in `kinds` follows the
/// global toggle live. Overrides are scoped to one selection: [`reset_overrides_on_selection_change`]
/// clears them whenever the selection changes (including deselect), so nothing persists to another comp
/// or survives Esc.
#[derive(Resource, Default)]
pub struct SelectionOverrides {
    pub comp: Option<Entity>,
    pub kinds: std::collections::HashMap<&'static str, bool>,
}

/// Per-sensor visualization override, written by the dendrite_build comp inspector's "Sensors" section.
/// `visible` defaults to `true` (an absent map entry means "follow the global toggle alone"); `full_extent`
/// defaults to `false` (draw the CAPPED lidar radius / FOV depth). The effective full-extent for a sensor is
/// this OR the global [`SensorVizGlobal::full_extent`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SensorVizState {
    /// Force this ONE sensor's viz on/off, ANDed with the global Sensors & FOV toggle.
    pub visible: bool,
    /// Draw THIS sensor at its TRUE extent (uncapped lidar radius / FOV far) instead of the display cap.
    /// ORed with the global [`SensorVizGlobal::full_extent`]; the scene rebuild re-resolves the mesh at the
    /// effective extent (the mesh cache keys on it).
    pub full_extent: bool,
}

impl Default for SensorVizState {
    fn default() -> Self {
        Self {
            visible: true,
            full_extent: false,
        }
    }
}

/// Global "draw sensors at their TRUE extent" toggle: the Displays-panel counterpart to the per-sensor
/// [`SensorVizState::full_extent`]. Default `false`: lidar radius is capped at
/// [`crate::geometry::LIDAR_DISPLAY_CAP_M`] and FOV depth at [`crate::geometry::FOV_DISPLAY_CAP_M`] so a
/// long-range sensor doesn't dwarf the robot. When `true` (or the per-sensor override is set) the scene
/// draws the uncapped extent. The standalone viewer leaves it at the default; dendrite_build's Displays
/// checkbox flips it (and republishes so the meshes re-resolve).
#[derive(Resource, Default)]
pub struct SensorVizGlobal {
    pub full_extent: bool,
}

/// Per-sensor visualization overrides, keyed by `(comp name, sensor name)`: the stable, rebuild-surviving
/// identity the inspector checkboxes carry (entities churn on every scene rebuild). A MISSING entry means
/// "default on": the sensor's viz follows the global Sensors & FOV toggle alone. An entry with
/// `visible: false` force-hides exactly that one sensor's entities (FOV frustum, lidar scan, glyph, triad)
/// regardless of the global toggle; the global toggle ANDs with this, so a sensor shows only when BOTH the
/// global Sensors display AND its own override allow it. Independent of the global kind toggle and of the
/// per-link [`SelectionOverrides`]. The standalone viewer never populates it (map stays empty ⇒ every
/// sensor defaults on), so its sensor visibility is byte-identical; the dendrite_build editor writes it
/// from the comp inspector's per-sensor checkboxes.
#[derive(Resource, Default)]
pub struct SensorVizOverrides(pub std::collections::HashMap<(String, String), SensorVizState>);

impl SensorVizOverrides {
    /// Whether the sensor named `sensor` on comp `comp` should show its viz per its per-sensor override,
    /// BEFORE the global Sensors toggle is applied. Absent ⇒ `true` (default on). Pure so the override
    /// logic unit-tests headless (no World/GPU); the visibility systems AND this with the global toggle.
    pub fn visible(&self, comp: &str, sensor: &str) -> bool {
        self.0
            .iter()
            .find(|((c, s), _)| c == comp && s == sensor)
            .is_none_or(|(_, st)| st.visible)
    }

    /// Whether the sensor named `sensor` on comp `comp` has its PER-SENSOR full-extent override set.
    /// Absent ⇒ `false` (follow the global toggle alone). The scene ORs this with the global to pick the
    /// effective drawn extent. Pure, so the override logic unit-tests headless.
    pub fn full_extent(&self, comp: &str, sensor: &str) -> bool {
        self.0
            .iter()
            .find(|((c, s), _)| c == comp && s == sensor)
            .is_some_and(|(_, st)| st.full_extent)
    }
}

/// Reset the per-link [`SelectionOverrides`] whenever [`Selected`] changes. Each new selection starts
/// with NO overrides (every kind follows the global toggle, live); deselection clears everything. Public
/// so headless tests can drive the same reset the [`PickPlugin`] schedule runs.
pub fn reset_overrides_on_selection_change(
    selected: Res<Selected>,
    mut overrides: ResMut<SelectionOverrides>,
) {
    if overrides.comp != selected.0 {
        overrides.comp = selected.0;
        overrides.kinds.clear();
    }
}

pub struct PickPlugin;

impl Plugin for PickPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<MeshPickingPlugin>() {
            app.add_plugins(MeshPickingPlugin);
        }
        // Make all meshes pickable (incl. async glTF subtrees) regardless of marker components.
        app.insert_resource(MeshPickingSettings {
            require_markers: false,
            ..default()
        });
        app.init_resource::<Selected>()
            .init_resource::<HighlightSet>()
            .init_resource::<IsolateSelection>()
            .init_resource::<IsolateSet>()
            .init_resource::<SelectionOverrides>()
            .init_resource::<SensorVizOverrides>()
            .init_resource::<SensorVizGlobal>()
            .add_systems(Update, clear_selection_on_escape)
            .add_systems(Update, reset_overrides_on_selection_change)
            .add_observer(on_click);
    }
}

/// Resolve the [`CompEntity`] that owns `start` by walking up the parent chain to the FIRST (nearest)
/// comp, inclusive of `start` itself. Pure and closure-based so the resolution can be unit-tested
/// headless (no World/GPU); `on_click` and the integration test both go through this.
///
/// Returns the nearest comp ancestor, or `None` if the chain reaches the top without hitting a comp.
pub fn owning_comp(
    start: Entity,
    parent_of: impl Fn(Entity) -> Option<Entity>,
    is_comp: impl Fn(Entity) -> bool,
) -> Option<Entity> {
    let mut cur = start;
    for _ in 0..10_000 {
        if is_comp(cur) {
            return Some(cur);
        }
        match parent_of(cur) {
            Some(p) => cur = p,
            None => return None,
        }
    }
    None // cycle guard
}

/// Left-click → select the owning comp (the nearest [`CompEntity`] at or above the picked entity).
///
/// Picking `Pointer<Click>` events BUBBLE up the `ChildOf` hierarchy, so this observer would otherwise
/// fire once per ancestor, and because bubbling ends at the root comp, the last write would always
/// win and select the root. We resolve from the ORIGINAL hit (constant across bubbling) and stop
/// propagation so exactly the clicked sub-component is selected.
pub fn on_click(
    mut click: On<Pointer<Click>>,
    comps: Query<&CompEntity>,
    parents: Query<&ChildOf>,
    mut selected: ResMut<Selected>,
) {
    if click.button != PointerButton::Primary {
        return;
    }
    let hit = click.original_event_target();
    click.propagate(false);

    let resolved = owning_comp(
        hit,
        |e| parents.get(e).ok().map(|c| c.parent()),
        |e| comps.contains(e),
    );
    // Opt-in diagnostic (`HCDVIZ_DEBUG_PICK=1`): print the hit entity, its full ancestor chain (marking
    // which links are comps), and what it resolved to. Zero cost when the env var is unset.
    if std::env::var_os("HCDVIZ_DEBUG_PICK").is_some() {
        let mut chain = Vec::new();
        let mut cur = Some(hit);
        while let Some(e) = cur {
            let tag = comps.get(e).ok().map(|c| format!("COMP({})", c.name));
            chain.push(format!(
                "{e:?}{}",
                tag.map(|t| format!("={t}")).unwrap_or_default()
            ));
            cur = parents.get(e).ok().map(|c| c.parent());
            if chain.len() > 128 {
                break;
            }
        }
        let res = resolved
            .and_then(|e| comps.get(e).ok())
            .map(|c| c.name.as_str())
            .unwrap_or("<none>");
        eprintln!(
            "[pick] hit={hit:?} -> resolved={res}\n[pick] chain: {}",
            chain.join(" -> ")
        );
    }
    if let Some(comp) = resolved {
        selected.0 = Some(comp);
    }
}

fn clear_selection_on_escape(keys: Res<ButtonInput<KeyCode>>, mut selected: ResMut<Selected>) {
    if keys.just_pressed(KeyCode::Escape) {
        selected.0 = None;
    }
}
