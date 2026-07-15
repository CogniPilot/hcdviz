//! Read-only "datasheet" formatting for the Inspector's component-detail sections: pure
//! `hcdformat` model → display-line lists, so the egui panel just renders them and every field-
//! selection/omission rule unit-tests headless (no egui). Mirrors the codebase's pure-helper split
//! (`hierarchy::hierarchy_rows`, `scene::collect_toggle_groups`, `connector::resolve_connector_meshes`).
//!
//! Design intent: surface ONLY authored fields. Every datasheet leaf is `Option` in the schema, so an
//! absent datum is OMITTED rather than shown blank or as a fabricated zero: the schema is far richer
//! than any one component populates (a QDD motor authors rotor-inertia + pole-pairs; a hobby servo
//! neither), and a blank line would read as "zero" rather than "unspecified". Numeric leaves keep
//! their authored text + unit verbatim ([`MeasuredValue`]/[`RatedValue`]/[`RangeValue`]), never
//! reformatted, matching the model's text-preserving round-trip contract.
use crate::schema::model::common::{MeasuredValue, RangeValue, RatedValue};
use crate::schema::model::{
    DynamicSurface, HmiElement, Motor, PowerSource, Software, Transmission,
};

/// A `<measured_value>` as `value [unit]` (unit appended only when present), or `None` when it carries
/// no magnitude text, so an empty `<inductance/>` contributes no row rather than a bare unit.
fn measured(m: &MeasuredValue) -> Option<String> {
    let v = m
        .value
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())?;
    Some(match m.unit.as_deref().filter(|u| !u.is_empty()) {
        Some(u) => format!("{v} {u}"),
        None => v.to_string(),
    })
}

/// A motor `<rated_value>` (voltage/current) showing the ratings that are authored, in datasheet order
/// nominal → continuous → peak → max, each tagged so "24 nom, 40 peak V" reads at a glance; a bare
/// text magnitude with no explicit ratings still shows. `None` when nothing at all is authored.
fn rated(r: &RatedValue) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    for (tag, val) in [
        ("nom", &r.nominal),
        ("cont", &r.continuous),
        ("peak", &r.peak),
        ("max", &r.max),
    ] {
        if let Some(v) = val.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            parts.push(format!("{v} {tag}"));
        }
    }
    if parts.is_empty() {
        // No explicit ratings: fall back to the bare `#text` nominal (e.g. `<voltage>5</voltage>`).
        if let Some(v) = r.value.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            parts.push(v.to_string());
        }
    }
    if parts.is_empty() {
        return None;
    }
    Some(match r.unit.as_deref().filter(|u| !u.is_empty()) {
        Some(u) => format!("{} {u}", parts.join(", ")),
        None => parts.join(", "),
    })
}

/// A `<range_value>` as `min..max [unit]` when bounded, else its bare `#text` nominal, used for the
/// control-surface deflection row. `None` when neither a range nor a nominal is authored.
fn range(r: &RangeValue) -> Option<String> {
    let unit = r.unit.as_deref().filter(|u| !u.is_empty());
    let lo = r.min.as_deref().map(str::trim).filter(|v| !v.is_empty());
    let hi = r.max.as_deref().map(str::trim).filter(|v| !v.is_empty());
    if let (Some(lo), Some(hi)) = (lo, hi) {
        return Some(match unit {
            Some(u) => format!("{lo}..{hi} {u}"),
            None => format!("{lo}..{hi}"),
        });
    }
    let v = r
        .value
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())?;
    Some(match unit {
        Some(u) => format!("{v} {u}"),
        None => v.to_string(),
    })
}

/// The datasheet lines for one `<motor>`, `(label, value)`, omitting every absent field. Order
/// follows datasheet convention: `@type`, torque/velocity constants (Kt/Kv), the rated voltage/current
/// envelopes, max-speed, stall-torque, pole-pairs, rotor-inertia, inductance, and the control modes.
/// Rotor-inertia + inductance are the two datasheet leaves a later schema revision added.
pub fn motor_lines(m: &Motor) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    if let Some(t) = &m.type_ {
        out.push(("type", t.to_string()));
    }
    if let Some(v) = m.torque_constant.as_ref().and_then(measured) {
        out.push(("Kt", v));
    }
    if let Some(v) = m.velocity_constant.as_ref().and_then(measured) {
        out.push(("Kv", v));
    }
    if let Some(v) = m.voltage.as_ref().and_then(rated) {
        out.push(("voltage", v));
    }
    if let Some(v) = m.current.as_ref().and_then(rated) {
        out.push(("current", v));
    }
    if let Some(v) = m.max_speed.as_ref().and_then(measured) {
        out.push(("max speed", v));
    }
    if let Some(v) = m.stall_torque.as_ref().and_then(measured) {
        out.push(("stall torque", v));
    }
    if let Some(v) = m
        .pole_pairs
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        out.push(("pole pairs", v.to_string()));
    }
    if let Some(v) = m.rotor_inertia.as_ref().and_then(measured) {
        out.push(("rotor inertia", v));
    }
    if let Some(v) = m.inductance.as_ref().and_then(measured) {
        out.push(("inductance", v));
    }
    if let Some(cm) = &m.control_modes {
        let modes: Vec<&str> = cm
            .mode
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if !modes.is_empty() {
            out.push(("control modes", modes.join(", ")));
        }
    }
    out
}

/// The datasheet lines for one `<power-source>`: the energy-source VARIANT plus a few key fields
/// of whichever sub-tree is present (battery/tank/fuel-cell/solar/supercapacitor). Only the first
/// present variant is described (the schema models these as mutually exclusive choices) and every
/// leaf omits when absent.
pub fn power_source_lines(p: &PowerSource) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    if let Some(b) = &p.battery {
        out.push(("source", "battery".to_string()));
        if let Some(v) = b
            .chemistry
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            out.push(("chemistry", v.to_string()));
        }
        // Cell configuration as SxP (e.g. "6S1P") only when at least the series count is authored.
        if let Some(s) = b
            .cells_series
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            let par = b
                .cells_parallel
                .as_deref()
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .unwrap_or("1");
            out.push(("cells", format!("{s}S{par}P")));
        }
        if let Some(v) = b.nominal_voltage.as_ref().and_then(measured) {
            out.push(("nominal voltage", v));
        }
        if let Some(v) = b.capacity.as_ref().and_then(measured) {
            out.push(("capacity", v));
        }
    } else if let Some(t) = &p.tank {
        out.push(("source", "tank".to_string()));
        if let Some(v) = t.fuel.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            out.push(("fuel", v.to_string()));
        }
        if let Some(v) = t.volume.as_ref().and_then(measured) {
            out.push(("volume", v));
        }
        if let Some(v) = t.pressure.as_ref().and_then(measured) {
            out.push(("pressure", v));
        }
    } else if let Some(f) = &p.fuel_cell {
        out.push(("source", "fuel-cell".to_string()));
        if let Some(v) = f.type_.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            out.push(("type", v.to_string()));
        }
        if let Some(v) = f.fuel.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            out.push(("fuel", v.to_string()));
        }
        if let Some(v) = f.rated_power.as_ref().and_then(measured) {
            out.push(("rated power", v));
        }
    } else if let Some(s) = &p.solar {
        out.push(("source", "solar".to_string()));
        if let Some(v) = s
            .cell_type
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            out.push(("cell type", v.to_string()));
        }
        if let Some(v) = s.peak_power.as_ref().and_then(measured) {
            out.push(("peak power", v));
        }
        if let Some(v) = s.area.as_ref().and_then(measured) {
            out.push(("area", v));
        }
    } else if let Some(c) = &p.supercapacitor {
        out.push(("source", "supercapacitor".to_string()));
        if let Some(v) = c.capacitance.as_ref().and_then(measured) {
            out.push(("capacitance", v));
        }
        if let Some(v) = c.max_voltage.as_ref().and_then(measured) {
            out.push(("max voltage", v));
        }
        if let Some(v) = c.energy.as_ref().and_then(measured) {
            out.push(("energy", v));
        }
    }
    out
}

/// The datasheet lines for one `<hmi>`: `@type` plus, for a scene-illumination HMI, a one-line
/// summary of its `<illumination>` (light-type + intensity) and the element description when authored.
pub fn hmi_lines(h: &HmiElement) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    if let Some(t) = &h.type_ {
        out.push(("type", t.to_string()));
    }
    if let Some(il) = &h.illumination {
        let mut parts: Vec<String> = Vec::new();
        if let Some(v) = il
            .light_type
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            parts.push(v.to_string());
        }
        if let Some(v) = il
            .intensity
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            parts.push(format!("intensity {v}"));
        }
        if !parts.is_empty() {
            out.push(("illumination", parts.join(", ")));
        }
    }
    if let Some(v) = h
        .description
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        out.push(("description", v.to_string()));
    }
    out
}

/// The datasheet lines for one `<dynamic-surface>`: the force-surface VARIANT plus a few key
/// fields of whichever sub-tree is present (prop/aerofoil/hydrofoil/control-surface/wheel/track/
/// gripper). First present variant only (they are mutually exclusive choices); every leaf omits absent.
pub fn dynamic_surface_lines(d: &DynamicSurface) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    if let Some(p) = &d.prop {
        out.push(("surface", "prop".to_string()));
        if let Some(v) = p.diameter.as_ref().and_then(measured) {
            out.push(("diameter", v));
        }
        if let Some(v) = p.pitch.as_ref().and_then(measured) {
            out.push(("pitch", v));
        }
        if let Some(v) = p.blades.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            out.push(("blades", v.to_string()));
        }
    } else if let Some(a) = &d.aerofoil {
        out.push(("surface", "aerofoil".to_string()));
        if let Some(v) = a
            .profile
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            out.push(("profile", v.to_string()));
        }
        if let Some(v) = a.span.as_ref().and_then(measured) {
            out.push(("span", v));
        }
        if let Some(v) = a.chord.as_ref().and_then(measured) {
            out.push(("chord", v));
        }
    } else if let Some(hf) = &d.hydrofoil {
        out.push(("surface", "hydrofoil".to_string()));
        if let Some(v) = hf
            .profile
            .as_deref()
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            out.push(("profile", v.to_string()));
        }
        if let Some(v) = hf.span.as_ref().and_then(measured) {
            out.push(("span", v));
        }
    } else if let Some(cs) = &d.control_surface {
        out.push(("surface", "control-surface".to_string()));
        if let Some(v) = cs.type_.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            out.push(("type", v.to_string()));
        }
        if let Some(v) = cs.deflection.as_ref().and_then(range) {
            out.push(("deflection", v));
        }
    } else if let Some(w) = &d.wheel {
        out.push(("surface", "wheel".to_string()));
        if let Some(v) = w.type_.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            out.push(("type", v.to_string()));
        }
        if let Some(v) = w.radius.as_ref().and_then(measured) {
            out.push(("radius", v));
        }
        if let Some(v) = w.width.as_ref().and_then(measured) {
            out.push(("width", v));
        }
    } else if let Some(t) = &d.track {
        out.push(("surface", "track".to_string()));
        if let Some(v) = t.width.as_ref().and_then(measured) {
            out.push(("width", v));
        }
        if let Some(v) = t.length.as_ref().and_then(measured) {
            out.push(("length", v));
        }
    } else if let Some(g) = &d.gripper {
        out.push(("surface", "gripper".to_string()));
        if let Some(v) = g.type_.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
            out.push(("type", v.to_string()));
        }
        if let Some(v) = g.grip_force.as_ref().and_then(measured) {
            out.push(("grip force", v));
        }
        if let Some(v) = g.payload.as_ref().and_then(measured) {
            out.push(("payload", v));
        }
    }
    out
}

/// The datasheet lines for a comp's `<software>` identity: version + hash, and the firmware
/// manifest URI when authored. The firmware IDENTITY (not the runtime `<discovered>` state).
pub fn software_lines(s: &Software) -> Vec<(&'static str, String)> {
    let mut out: Vec<(&'static str, String)> = Vec::new();
    if let Some(v) = s
        .version
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        out.push(("version", v.to_string()));
    }
    if let Some(v) = s.hash.as_deref().map(str::trim).filter(|v| !v.is_empty()) {
        out.push(("hash", v.to_string()));
    }
    if let Some(v) = s
        .firmware_manifest_uri
        .as_deref()
        .map(str::trim)
        .filter(|v| !v.is_empty())
    {
        out.push(("manifest", v.to_string()));
    }
    out
}

// ── transmission awareness ────────────────────────────────────────────────

/// Is an `xs:boolean`-typed attribute truthy? Matches the schema/validator lexical space (`true`/`1`),
/// case-insensitively, so the viewer reads a `<spring @clutch>` flag exactly as the validator does.
fn xs_true(s: Option<&str>) -> bool {
    matches!(
        s.map(str::trim),
        Some(v) if v.eq_ignore_ascii_case("true") || v == "1"
    )
}

/// A transmission-endpoint badge for a joint: which `<transmission>` drives it, the reduction it
/// carries, and whether the coupling is compliant (a `<spring>` ⇒ series/parallel-elastic) and/or
/// clutched. Purely descriptive: the Joints panel renders [`Self::summary`] beside the joint.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionBadge {
    /// The transmission's `@name`, or "transmission" when it is unnamed.
    pub name: String,
    /// `<reduction>` text, verbatim, when authored.
    pub reduction: Option<String>,
    /// Any `<spring>` present ⇒ a compliant (SEA/PEA-family) coupling, not a rigid gear train.
    pub compliant: bool,
    /// Any `<spring @clutch>` truthy ⇒ a clutched/engageable coupling.
    pub clutched: bool,
}

impl TransmissionBadge {
    /// The one-line badge text for the Joints panel: a gear glyph + the transmission name, the
    /// reduction when known (verbatim: the author owns the ratio notation), and a compliance/clutch
    /// tag when the coupling carries a spring and/or clutch.
    pub fn summary(&self) -> String {
        let mut s = format!("⚙ {}", self.name);
        if let Some(r) = &self.reduction {
            s.push_str(&format!("  {r}"));
        }
        let mut tags: Vec<&str> = Vec::new();
        if self.compliant {
            tags.push("spring");
        }
        if self.clutched {
            tags.push("clutch");
        }
        if !tags.is_empty() {
            s.push_str(&format!("  [{}]", tags.join("+")));
        }
        s
    }
}

/// The [`TransmissionBadge`] for the joint named `joint_name`, if any `<transmission>` lists it as a
/// `<joint ref=>` endpoint. Pure lookup over the doc-level transmissions; the XSD explicitly DEFERS
/// endpoint `@ref` resolution to the companion validator, so this overlay stays lenient: a dangling
/// ref simply never matches, and an empty joint name (an unnamed joint) never badges. The FIRST
/// matching transmission wins (a joint driven by more than one transmission is degenerate; badging one
/// is enough to flag it as motor-driven).
pub fn transmission_badge(
    transmissions: &[Transmission],
    joint_name: &str,
) -> Option<TransmissionBadge> {
    if joint_name.is_empty() {
        return None;
    }
    let t = transmissions.iter().find(|t| {
        t.joint
            .iter()
            .any(|e| e.ref_.as_deref() == Some(joint_name))
    })?;
    Some(TransmissionBadge {
        name: t
            .name
            .as_deref()
            .map(str::trim)
            .filter(|n| !n.is_empty())
            .unwrap_or("transmission")
            .to_string(),
        reduction: t
            .reduction
            .as_deref()
            .map(str::trim)
            .filter(|r| !r.is_empty())
            .map(str::to_string),
        compliant: !t.spring.is_empty(),
        clutched: t.spring.iter().any(|s| xs_true(s.clutch.as_deref())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::model::Comp;
    use crate::schema::Hcdf;

    /// A full actuator-shaped fixture exercising every component datasheet section and transmission.
    const ACTUATOR: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="act" body-frame="FLU" world-frame="ENU">
  <comp name="case">
    <motor name="qdd" type="bldc">
      <voltage unit="V" nominal="24" peak="48"/>
      <current unit="A" continuous="20" peak="60"/>
      <torque-constant unit="Nm/A">0.09</torque-constant>
      <velocity-constant unit="rpm/V">21</velocity-constant>
      <max-speed unit="rad/s">42</max-speed>
      <stall-torque unit="Nm">40</stall-torque>
      <pole-pairs>21</pole-pairs>
      <rotor-inertia unit="kg.m2">0.0006</rotor-inertia>
      <inductance unit="H">0.00012</inductance>
      <control-modes><mode>torque</mode><mode>position</mode></control-modes>
    </motor>
    <power-source name="pack">
      <battery>
        <chemistry>LiPo</chemistry>
        <cells-series>6</cells-series>
        <nominal-voltage unit="V">22.2</nominal-voltage>
        <capacity unit="Ah">5</capacity>
      </battery>
    </power-source>
    <hmi name="light" type="led-illumination">
      <illumination light-type="spot" intensity="1200"/>
    </hmi>
    <dynamic-surface name="wheel0">
      <wheel type="rubber"><radius unit="m">0.1</radius><width unit="m">0.04</width></wheel>
    </dynamic-surface>
    <software name="fw"><version>1.2.3</version><hash>abcd</hash></software>
  </comp>
  <comp name="flange"/>
  <joint name="axis" type="revolute">
    <parent comp="case"/><child comp="flange"/><axis xyz="0 0 1"/>
  </joint>
  <transmission name="gearbox" type="harmonic">
    <motor ref="qdd"/><joint ref="axis"/>
    <reduction>9:1</reduction>
    <spring placement="series" clutch="true"><stiffness unit="Nm/rad">120</stiffness></spring>
  </transmission>
</hcdf>"#;

    fn case(h: &Hcdf) -> &Comp {
        h.comp.iter().find(|c| c.name == "case").expect("case comp")
    }

    #[test]
    fn motor_lines_include_s2_leaves_and_omit_absent() {
        let h = Hcdf::from_xml_str(ACTUATOR).unwrap();
        let m = &case(&h).motor[0];
        let lines = motor_lines(m);
        // Rendered as a lookup so the assertions read by label, not by index.
        let by: std::collections::HashMap<&str, String> = lines.iter().cloned().collect();
        assert_eq!(by.get("type").map(String::as_str), Some("bldc"));
        assert_eq!(by.get("Kt").map(String::as_str), Some("0.09 Nm/A"));
        // Rated envelopes surface the authored ratings, tagged + united.
        assert_eq!(
            by.get("voltage").map(String::as_str),
            Some("24 nom, 48 peak V")
        );
        assert_eq!(
            by.get("current").map(String::as_str),
            Some("20 cont, 60 peak A")
        );
        // The two added leaves round-trip into the panel.
        assert_eq!(
            by.get("rotor inertia").map(String::as_str),
            Some("0.0006 kg.m2")
        );
        assert_eq!(by.get("inductance").map(String::as_str), Some("0.00012 H"));
        assert_eq!(by.get("pole pairs").map(String::as_str), Some("21"));
        assert_eq!(
            by.get("control modes").map(String::as_str),
            Some("torque, position")
        );
        // A field the fixture never authors is omitted (not shown blank/zero).
        assert!(!by.contains_key("thermal resistance"));
    }

    #[test]
    fn motor_lines_omit_everything_for_a_bare_motor() {
        let m = Motor::default();
        assert!(
            motor_lines(&m).is_empty(),
            "a bare <motor/> contributes no rows"
        );
    }

    #[test]
    fn power_hmi_surface_software_key_fields() {
        let h = Hcdf::from_xml_str(ACTUATOR).unwrap();
        let c = case(&h);
        let power: std::collections::HashMap<&str, String> =
            power_source_lines(&c.power_source[0]).into_iter().collect();
        assert_eq!(power.get("source").map(String::as_str), Some("battery"));
        assert_eq!(power.get("chemistry").map(String::as_str), Some("LiPo"));
        assert_eq!(power.get("cells").map(String::as_str), Some("6S1P"));
        assert_eq!(
            power.get("nominal voltage").map(String::as_str),
            Some("22.2 V")
        );

        let hmi: std::collections::HashMap<&str, String> =
            hmi_lines(&c.hmi[0]).into_iter().collect();
        assert_eq!(
            hmi.get("type").map(String::as_str),
            Some("led-illumination")
        );
        assert_eq!(
            hmi.get("illumination").map(String::as_str),
            Some("spot, intensity 1200")
        );

        let surf: std::collections::HashMap<&str, String> =
            dynamic_surface_lines(&c.dynamic_surface[0])
                .into_iter()
                .collect();
        assert_eq!(surf.get("surface").map(String::as_str), Some("wheel"));
        assert_eq!(surf.get("radius").map(String::as_str), Some("0.1 m"));

        let sw = software_lines(c.software.as_ref().unwrap());
        let sw: std::collections::HashMap<&str, String> = sw.into_iter().collect();
        assert_eq!(sw.get("version").map(String::as_str), Some("1.2.3"));
        assert_eq!(sw.get("hash").map(String::as_str), Some("abcd"));
    }

    #[test]
    fn transmission_badge_flags_motor_driven_joint_with_reduction_and_tags() {
        let h = Hcdf::from_xml_str(ACTUATOR).unwrap();
        let b =
            transmission_badge(&h.transmission, "axis").expect("axis is a transmission endpoint");
        assert_eq!(b.name, "gearbox");
        assert_eq!(b.reduction.as_deref(), Some("9:1"));
        assert!(
            b.compliant,
            "the series spring makes it a compliant coupling"
        );
        assert!(
            b.clutched,
            "the spring's clutch=true flags a clutched coupling"
        );
        assert_eq!(b.summary(), "⚙ gearbox  9:1  [spring+clutch]");
    }

    #[test]
    fn transmission_badge_none_for_undriven_or_unnamed() {
        let h = Hcdf::from_xml_str(ACTUATOR).unwrap();
        assert!(
            transmission_badge(&h.transmission, "nonexistent").is_none(),
            "a joint no transmission references gets no badge"
        );
        assert!(
            transmission_badge(&h.transmission, "").is_none(),
            "an unnamed joint never badges"
        );
    }

    #[test]
    fn transmission_badge_rigid_gear_train_has_no_tags() {
        // A reduction with no <spring>: driven, but neither compliant nor clutched.
        let xml = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="rigid">
  <comp name="a"/><comp name="b"/>
  <joint name="j" type="revolute"><parent comp="a"/><child comp="b"/></joint>
  <transmission name="gt"><joint ref="j"/><reduction>50</reduction></transmission>
</hcdf>"#;
        let h = Hcdf::from_xml_str(xml).unwrap();
        let b = transmission_badge(&h.transmission, "j").unwrap();
        assert!(!b.compliant && !b.clutched);
        assert_eq!(b.summary(), "⚙ gt  50");
    }
}
