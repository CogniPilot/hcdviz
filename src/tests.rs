//! Headless unit tests (no GPU) for the three correctness-critical pieces (items 1-3 below):
//!  1. `pose_to_transform`: quat wins over rpy; rpy is XYZ Euler.
//!  2. Two-basis frame handling: body +X/+Z map correctly for all 4 (FLU/FRD × ENU/NED) combos.
//!  3. Kinematic-tree extraction: parent/child edges, roots, and multi-parent (loop) handling.
use crate::frame::{pose_to_transform, BodyConvention, FrameConvention, WorldConvention};
use crate::kinematics::build_kinematic_tree;
use crate::schema::Hcdf;
use bevy::prelude::*;

const EPS: f32 = 1e-4;

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < EPS
}

// ── item 1: pose_to_transform ────────────────────────────────────────────────

#[test]
fn pose_translation_only() {
    let p = crate::schema::Pose {
        xyz: Some([1.0, 2.0, 3.0]),
        rpy: Some([0.0; 3]),
        quat: None,
    };
    let t = pose_to_transform(&p);
    assert!(close(t.translation, Vec3::new(1.0, 2.0, 3.0)));
    assert!(t.rotation.abs_diff_eq(Quat::IDENTITY, EPS));
}

#[test]
fn pose_rpy_is_xyz_euler() {
    // 90° roll about X only.
    let p = crate::schema::Pose {
        xyz: Some([0.0; 3]),
        rpy: Some([std::f32::consts::FRAC_PI_2 as f64, 0.0, 0.0]),
        quat: None,
    };
    let t = pose_to_transform(&p);
    // +Y should rotate to +Z under a +90° rotation about X.
    let rotated = t.rotation * Vec3::Y;
    assert!(close(rotated, Vec3::Z), "got {rotated:?}");
}

#[test]
fn pose_quat_wins_over_rpy() {
    // rpy says identity-ish, quat says 180° about Z; quat must win.
    let q = Quat::from_rotation_z(std::f32::consts::PI);
    let p = crate::schema::Pose {
        xyz: Some([0.0; 3]),
        rpy: Some([1.0, 1.0, 1.0]), // nonzero, must be IGNORED because quat is present.
        quat: Some([q.x as f64, q.y as f64, q.z as f64, q.w as f64]),
    };
    let t = pose_to_transform(&p);
    let rotated = t.rotation * Vec3::X;
    assert!(close(rotated, Vec3::NEG_X), "quat did not win: {rotated:?}");
}

#[test]
fn pose_degenerate_quat_is_identity() {
    let p = crate::schema::Pose {
        xyz: Some([0.0; 3]),
        rpy: Some([0.0; 3]),
        quat: Some([0.0, 0.0, 0.0, 0.0]),
    };
    let t = pose_to_transform(&p);
    assert!(t.rotation.is_finite());
    assert!(t.rotation.abs_diff_eq(Quat::IDENTITY, EPS));
}

// ── item 2: two-basis frame handling (the 4 combos) ──────────────────────────
//
// We assert how a body's authored axes land in Bevy (Y-up, right-handed: +X right, +Y up, −Z fwd).
// The internal basis maps ENU's Up→Bevy +Y and North→Bevy −Z; NED's Down→Bevy −Y, North→Bevy −Z.

fn conv(body: BodyConvention, world: WorldConvention) -> FrameConvention {
    FrameConvention { body, world }
}

#[test]
fn flu_enu_forward_is_minus_z_up_is_plus_y() {
    let c = conv(BodyConvention::Flu, WorldConvention::Enu);
    // FLU body +X = Forward → Bevy −Z (into the screen).
    assert!(
        close(c.body_point_to_bevy(Vec3::X), Vec3::NEG_Z),
        "fwd: {:?}",
        c.body_point_to_bevy(Vec3::X)
    );
    // FLU body +Z = Up → Bevy +Y.
    assert!(
        close(c.body_point_to_bevy(Vec3::Z), Vec3::Y),
        "up: {:?}",
        c.body_point_to_bevy(Vec3::Z)
    );
    // FLU body +Y = Left → Bevy −X.
    assert!(
        close(c.body_point_to_bevy(Vec3::Y), Vec3::NEG_X),
        "left: {:?}",
        c.body_point_to_bevy(Vec3::Y)
    );
}

#[test]
fn frd_ned_forward_is_minus_z_down_is_minus_y() {
    let c = conv(BodyConvention::Frd, WorldConvention::Ned);
    // FRD body +X = Forward → Bevy −Z.
    assert!(
        close(c.body_point_to_bevy(Vec3::X), Vec3::NEG_Z),
        "fwd: {:?}",
        c.body_point_to_bevy(Vec3::X)
    );
    // FRD body +Z = Down → Bevy −Y.
    assert!(
        close(c.body_point_to_bevy(Vec3::Z), Vec3::NEG_Y),
        "down: {:?}",
        c.body_point_to_bevy(Vec3::Z)
    );
    // FRD body +Y = Right → Bevy +X.
    assert!(
        close(c.body_point_to_bevy(Vec3::Y), Vec3::X),
        "right: {:?}",
        c.body_point_to_bevy(Vec3::Y)
    );
}

#[test]
fn flu_ned_mixed_is_well_defined() {
    // Mixed FLU body in an NED world: well-defined but unusual. Forward(X)→world North→Bevy... FLU
    // forward maps to world Y(=East under NED), then NED East→Bevy +X.
    let c = conv(BodyConvention::Flu, WorldConvention::Ned);
    assert!(
        close(c.body_point_to_bevy(Vec3::X), Vec3::X),
        "fwd: {:?}",
        c.body_point_to_bevy(Vec3::X)
    );
    // up(Z)→world Z(=Down under NED)→Bevy −Y.
    assert!(
        close(c.body_point_to_bevy(Vec3::Z), Vec3::NEG_Y),
        "up: {:?}",
        c.body_point_to_bevy(Vec3::Z)
    );
}

#[test]
fn frd_enu_mixed_is_well_defined() {
    let c = conv(BodyConvention::Frd, WorldConvention::Enu);
    // FRD forward(X)→world X(=East under ENU)→Bevy +X.
    assert!(
        close(c.body_point_to_bevy(Vec3::X), Vec3::X),
        "fwd: {:?}",
        c.body_point_to_bevy(Vec3::X)
    );
    // FRD down(Z)→world Z(=Up under ENU)→Bevy +Y.
    assert!(
        close(c.body_point_to_bevy(Vec3::Z), Vec3::Y),
        "down: {:?}",
        c.body_point_to_bevy(Vec3::Z)
    );
}

#[test]
fn all_basis_maps_are_orthonormal() {
    let mats = [
        WorldConvention::Enu.to_bevy_mat3(),
        WorldConvention::Ned.to_bevy_mat3(),
        BodyConvention::Flu.to_world_mat3(),
        BodyConvention::Frd.to_world_mat3(),
    ];
    for m in mats {
        let (x, y, z) = (m.x_axis, m.y_axis, m.z_axis);
        assert!((x.length() - 1.0).abs() < EPS);
        assert!((y.length() - 1.0).abs() < EPS);
        assert!((z.length() - 1.0).abs() < EPS);
        assert!(x.dot(y).abs() < EPS && y.dot(z).abs() < EPS && x.dot(z).abs() < EPS);
    }
}

#[test]
fn all_world_maps_are_proper_rotations() {
    // Both world→Bevy maps are pure rotations (det = +1): NED is re-expressed as a proper rotation by
    // its axis assignment, not a handedness-flipping reflection.
    assert!((WorldConvention::Enu.to_bevy_mat3().determinant() - 1.0).abs() < EPS);
    assert!((WorldConvention::Ned.to_bevy_mat3().determinant() - 1.0).abs() < EPS);
    // The full body→Bevy composition stays a proper rotation for the canonical pairs too.
    let flu_enu = conv(BodyConvention::Flu, WorldConvention::Enu);
    let frd_ned = conv(BodyConvention::Frd, WorldConvention::Ned);
    let det = |c: FrameConvention| (c.world.to_bevy_mat3() * c.body.to_world_mat3()).determinant();
    assert!((det(flu_enu) - 1.0).abs() < EPS);
    assert!((det(frd_ned) - 1.0).abs() < EPS);
}

#[test]
fn convention_mapping_defaults() {
    use crate::schema::model::enums::{BodyFrame, WorldFrame};
    assert_eq!(
        WorldConvention::from_schema(Some(WorldFrame::NED)),
        WorldConvention::Ned
    );
    assert_eq!(
        WorldConvention::from_schema(Some(WorldFrame::ENU)),
        WorldConvention::Enu
    );
    assert_eq!(WorldConvention::from_schema(None), WorldConvention::Enu);
    assert_eq!(
        BodyConvention::from_schema(Some(BodyFrame::FRD)),
        BodyConvention::Frd
    );
    assert_eq!(
        BodyConvention::from_schema(Some(BodyFrame::FLU)),
        BodyConvention::Flu
    );
    assert_eq!(BodyConvention::from_schema(None), BodyConvention::Flu);
}

// ── item 3: kinematic-tree extraction ────────────────────────────────────────

fn parse(xml: &str) -> Hcdf {
    Hcdf::from_xml_str(xml).expect("test HCDF should parse")
}

/// Index of a comp by name in the doc (helper for asserting edges).
fn idx(h: &Hcdf, name: &str) -> usize {
    h.comp.iter().position(|c| c.name == name).unwrap()
}

const SERIAL_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="chain" body-frame="FLU" world-frame="ENU">
  <comp name="base" role="parent"/>
  <comp name="link1"/>
  <comp name="link2"/>
  <joint name="j1" type="fixed"><parent comp="base"/><child comp="link1"/><origin xyz="0 0 0.1"/></joint>
  <joint name="j2" type="revolute"><parent comp="link1"/><child comp="link2"/><origin xyz="0 0 0.2"/></joint>
</hcdf>"#;

#[test]
fn serial_chain_edges_and_root() {
    let h = parse(SERIAL_CHAIN);
    let tree = build_kinematic_tree(&h);
    assert_eq!(tree.roots, vec![idx(&h, "base")]);
    assert_eq!(tree.edges.len(), 2);
    assert!(tree.constraints.is_empty());
    // base -> link1 -> link2
    let e1 = &tree.edges[0];
    assert_eq!((e1.parent, e1.child), (idx(&h, "base"), idx(&h, "link1")));
    let e2 = &tree.edges[1];
    assert_eq!((e2.parent, e2.child), (idx(&h, "link1"), idx(&h, "link2")));
}

const MULTI_ROOT: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="multi" body-frame="FRD" world-frame="NED">
  <comp name="frame" role="parent"/>
  <comp name="navq95"/>
  <comp name="floating"/>
  <joint name="j" type="fixed"><parent comp="frame"/><child comp="navq95"/><origin xyz="0 0 0.02"/></joint>
</hcdf>"#;

#[test]
fn unjointed_comps_are_roots() {
    let h = parse(MULTI_ROOT);
    let tree = build_kinematic_tree(&h);
    // both "frame" and "floating" have no incoming edge.
    let mut roots = tree.roots.clone();
    roots.sort_unstable();
    let mut expected = vec![idx(&h, "frame"), idx(&h, "floating")];
    expected.sort_unstable();
    assert_eq!(roots, expected);
    assert_eq!(tree.edges.len(), 1);
}

const PARALLEL_LOOP: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="loop" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="a"/>
  <comp name="b"/>
  <comp name="ee"/>
  <joint name="j1" type="fixed"><parent comp="base"/><child comp="a"/></joint>
  <joint name="j2" type="fixed"><parent comp="base"/><child comp="b"/></joint>
  <joint name="j3" type="revolute"><parent comp="a"/><child comp="ee"/></joint>
  <joint name="j4" type="revolute"><parent comp="b"/><child comp="ee"/></joint>
</hcdf>"#;

#[test]
fn multi_parent_picks_one_edge_and_records_constraint() {
    let h = parse(PARALLEL_LOOP);
    let tree = build_kinematic_tree(&h);
    // "ee" is targeted by two joints (j3, j4): exactly one becomes a primary edge, the other a link.
    let ee = idx(&h, "ee");
    let primary_edges_to_ee = tree.edges.iter().filter(|e| e.child == ee).count();
    assert_eq!(
        primary_edges_to_ee, 1,
        "ee must have exactly one tree parent"
    );
    let constraints_to_ee = tree.constraints.iter().filter(|c| c.child == ee).count();
    assert_eq!(
        constraints_to_ee, 1,
        "the other joint becomes a constraint link"
    );
    // only "base" is a root.
    assert_eq!(tree.roots, vec![idx(&h, "base")]);
    // no panic, acyclic: every non-root child has a primary edge.
    assert!(tree.primary_edge_of.contains_key(&ee));
}

const EXPLICIT_LOOP: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="explicitloop" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="a"/>
  <joint name="j1" type="fixed"><parent comp="base"/><child comp="a"/></joint>
  <joint name="jloop" type="revolute"><parent comp="a"/><child comp="base"/><loop><predecessor>a</predecessor><successor>base</successor></loop></joint>
</hcdf>"#;

#[test]
fn explicit_loop_joint_is_constraint_not_edge() {
    let h = parse(EXPLICIT_LOOP);
    let tree = build_kinematic_tree(&h);
    // base stays a root; the <loop> joint does not become a tree edge.
    assert_eq!(tree.roots, vec![idx(&h, "base")]);
    assert_eq!(tree.edges.len(), 1);
    assert_eq!(tree.constraints.len(), 1);
}

#[test]
fn unknown_comp_references_are_skipped() {
    let xml = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="bad">
  <comp name="base"/>
  <joint name="j"><parent comp="ghost"/><child comp="base"/></joint>
</hcdf>"#;
    let h = parse(xml);
    let tree = build_kinematic_tree(&h);
    // ghost is unknown → joint skipped; base remains a root, no edges, no panic.
    assert_eq!(tree.roots, vec![idx(&h, "base")]);
    assert!(tree.edges.is_empty());
}

// A quad-style FRD/NED doc (kept inline so the test is self-contained, no cross-repo path. The real
// hcformat examples are round-tripped by the hcdformat crate's own test suite).
const DRONE_FRD_NED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="quad" body-frame="FRD" world-frame="NED">
  <comp name="frame" role="parent"/>
  <comp name="fc"/>
  <comp name="m0"/>
  <comp name="m1"/>
  <comp name="m2"/>
  <comp name="m3"/>
  <joint name="jfc" type="fixed"><parent comp="frame"/><child comp="fc"/><origin xyz="0 0 -0.02"/></joint>
  <joint name="j0" type="revolute"><parent comp="frame"/><child comp="m0"/><origin xyz="0.1 0.1 0"/></joint>
  <joint name="j1" type="revolute"><parent comp="frame"/><child comp="m1"/><origin xyz="-0.1 0.1 0"/></joint>
  <joint name="j2" type="revolute"><parent comp="frame"/><child comp="m2"/><origin xyz="-0.1 -0.1 0"/></joint>
  <joint name="j3" type="revolute"><parent comp="frame"/><child comp="m3"/><origin xyz="0.1 -0.1 0"/></joint>
</hcdf>"#;

#[test]
fn drone_frd_ned_parses_and_builds() {
    // An FRD/NED quad: verify it goes through convention detection + tree extraction with no panic
    // and produces a sane spanning tree (frame as the root, an edge per child).
    let h = parse(DRONE_FRD_NED);
    let convention = FrameConvention::from_hcdf(&h);
    assert_eq!(convention.world, WorldConvention::Ned);
    assert_eq!(convention.body, BodyConvention::Frd);
    let tree = build_kinematic_tree(&h);
    assert_eq!(tree.edges.len(), 5, "five children, five edges");
    assert!(tree.constraints.is_empty());
    // every child has exactly one primary parent.
    for e in &tree.edges {
        assert!(tree.primary_edge_of.contains_key(&e.child));
    }
    // "frame" is the structural root.
    assert_eq!(tree.roots, vec![idx(&h, "frame")]);
}

// ── isolate-selection visibility: the pure `item_visibility` truth table ──────
//
// One toggleable item shows iff its display kind is enabled AND (isolate is off OR nothing is selected
// OR the selection owns the item). These cover every cell of that table headless (no World/GPU).

use crate::scene::item_visibility;

/// A stand-in owner entity (`item_visibility` only compares `Entity`s, never touches a World).
fn ent(i: u32) -> Entity {
    Entity::from_raw_u32(i).expect("valid test entity index")
}

#[test]
fn item_visibility_isolate_off_follows_kind() {
    let owner = ent(1);
    // Isolate off → selection is irrelevant; visibility just mirrors the kind toggle.
    for selected in [None, Some(owner), Some(ent(2))] {
        assert_eq!(
            item_visibility(true, false, selected, owner, None),
            Visibility::Inherited
        );
        assert_eq!(
            item_visibility(false, false, selected, owner, None),
            Visibility::Hidden
        );
    }
}

#[test]
fn item_visibility_isolate_on_no_selection_shows_all() {
    let owner = ent(1);
    // Isolate on but nothing selected → behaves like isolate off (shows all enabled kinds). This is
    // why deselect auto-reverts: clearing `Selected` makes isolate a no-op.
    assert_eq!(
        item_visibility(true, true, None, owner, None),
        Visibility::Inherited
    );
    assert_eq!(
        item_visibility(false, true, None, owner, None),
        Visibility::Hidden
    );
}

#[test]
fn item_visibility_isolate_on_selected_is_owner() {
    let owner = ent(1);
    // Isolate on + the item belongs to the selection → visible when the kind is enabled.
    assert_eq!(
        item_visibility(true, true, Some(owner), owner, None),
        Visibility::Inherited
    );
    // Kind disabled still hides it (isolate never overrides a global-off toggle).
    assert_eq!(
        item_visibility(false, true, Some(owner), owner, None),
        Visibility::Hidden
    );
}

#[test]
fn item_visibility_isolate_on_selected_other_hides_even_if_enabled() {
    let owner = ent(1);
    let other = ent(2);
    // Isolate on + a DIFFERENT comp selected → hidden even though the kind is enabled.
    assert_eq!(
        item_visibility(true, true, Some(other), owner, None),
        Visibility::Hidden
    );
    assert_eq!(
        item_visibility(false, true, Some(other), owner, None),
        Visibility::Hidden
    );
}

// ── comp-set isolate: the pure `isolate_hides` / `isolate_hides_edge` rules ────
//
// `IsolateSet = Some(set)` supersedes the single-selection path: an item shows iff its owner is in the
// set (kind toggle still applies), and a tree edge shows iff BOTH endpoints are in the set. `None`
// falls back to the single-`Selected` behaviour. These cover both modes headless (no World/GPU).

use crate::scene::{isolate_hides, isolate_hides_edge};

#[test]
fn isolate_set_none_matches_single_selection_path() {
    let owner = ent(1);
    let other = ent(2);
    // None ⇒ identical to the isolate-flag behaviour: no selection ⇒ never hidden; a different comp
    // selected + isolate on ⇒ hidden; the selected comp itself ⇒ shown.
    assert!(!isolate_hides(true, None, None, owner));
    assert!(isolate_hides(true, Some(other), None, owner));
    assert!(!isolate_hides(true, Some(owner), None, owner));
    assert!(!isolate_hides(false, Some(other), None, owner)); // isolate off ⇒ no-op
}

#[test]
fn isolate_set_some_shows_only_members_ignoring_flag() {
    let a = ent(1);
    let b = ent(2);
    let c = ent(3);
    let set: HashSet<Entity> = [a, b].into_iter().collect();
    // In-set comps show, out-of-set comps hide, regardless of the isolate flag or the single selection.
    assert!(!isolate_hides(false, Some(c), Some(&set), a));
    assert!(!isolate_hides(true, None, Some(&set), b));
    assert!(isolate_hides(false, Some(a), Some(&set), c));
    // An empty set is "isolate to nothing": everything hides.
    let empty: HashSet<Entity> = HashSet::new();
    assert!(isolate_hides(true, Some(a), Some(&empty), a));
}

#[test]
fn item_visibility_respects_isolate_set() {
    let a = ent(1);
    let c = ent(3);
    let set: HashSet<Entity> = [a].into_iter().collect();
    // Kind on + owner in set ⇒ shown; owner out of set ⇒ hidden; kind off always hides.
    assert_eq!(
        item_visibility(true, false, Some(c), a, Some(&set)),
        Visibility::Inherited
    );
    assert_eq!(
        item_visibility(true, false, None, c, Some(&set)),
        Visibility::Hidden
    );
    assert_eq!(
        item_visibility(false, false, None, a, Some(&set)),
        Visibility::Hidden
    );
}

#[test]
fn isolate_hides_edge_needs_both_endpoints_in_set() {
    let p = ent(1);
    let c = ent(2);
    let outsider = ent(3);
    let both: HashSet<Entity> = [p, c].into_iter().collect();
    // Both endpoints in the set ⇒ edge shows; only one in ⇒ edge hides.
    assert!(!isolate_hides_edge(false, None, Some(&both), p, c));
    let only_child: HashSet<Entity> = [c].into_iter().collect();
    assert!(isolate_hides_edge(false, None, Some(&only_child), p, c));
    // None path: keep the edge only when its CHILD is the single selection (isolate on).
    assert!(!isolate_hides_edge(true, Some(c), None, p, c));
    assert!(isolate_hides_edge(true, Some(outsider), None, p, c));
    assert!(!isolate_hides_edge(false, Some(outsider), None, p, c)); // isolate off ⇒ no-op
}

// ── per-link overrides: the pure `effective_kind_enabled` truth table ─────────
//
// A per-link override only ever applies to the item OWNED by the currently selected comp; every other
// comp (and any kind the selection hasn't overridden) follows the live global toggle. These cover
// every cell headless (no World/GPU); the result feeds `item_visibility` in place of the raw global.

use crate::scene::effective_kind_enabled;

#[test]
fn effective_kind_not_selected_owner_always_global() {
    // Not the selected owner → the override is irrelevant; the global value always wins.
    for global in [true, false] {
        for ov in [None, Some(true), Some(false)] {
            assert_eq!(
                effective_kind_enabled(global, ov, false),
                global,
                "non-selected owner must follow global (global={global}, override={ov:?})"
            );
        }
    }
}

#[test]
fn effective_kind_selected_owner_no_override_follows_global() {
    // Selected owner but no override recorded for this kind → follows the live global toggle.
    assert!(effective_kind_enabled(true, None, true));
    assert!(!effective_kind_enabled(false, None, true));
}

#[test]
fn effective_kind_selected_owner_override_wins() {
    // Selected owner WITH an override → the override wins over the global toggle (both directions).
    assert!(
        effective_kind_enabled(false, Some(true), true),
        "override-on beats global-off"
    );
    assert!(
        !effective_kind_enabled(true, Some(false), true),
        "override-off beats global-on"
    );
    // And it agrees with global when the override happens to match it.
    assert!(effective_kind_enabled(true, Some(true), true));
    assert!(!effective_kind_enabled(false, Some(false), true));
}

// ── per-sensor viz overrides: `SensorVizOverrides::visible` targets one sensor ─
//
// The comp inspector's per-sensor checkboxes write `(comp, sensor) -> visible` here; the sensor-viz
// systems AND the result with the global Sensors toggle. An entry force-hides EXACTLY the one keyed
// sensor; every other (comp, sensor) pair defaults on. Pure lookup, headless (no World/GPU).

use crate::pick::{SensorVizOverrides, SensorVizState};

#[test]
fn sensor_viz_overrides_default_all_on() {
    // Empty map (the standalone-viewer state) ⇒ every sensor defaults visible.
    let ov = SensorVizOverrides::default();
    assert!(ov.visible("cam_link", "front_cam"));
    assert!(ov.visible("any_comp", "any_sensor"));
}

#[test]
fn sensor_viz_overrides_hide_targets_exactly_one_sensor() {
    let mut ov = SensorVizOverrides::default();
    ov.0.insert(
        ("cam_link".to_string(), "front_cam".to_string()),
        SensorVizState {
            visible: false,
            ..Default::default()
        },
    );
    // The keyed sensor is hidden…
    assert!(!ov.visible("cam_link", "front_cam"));
    // …but a different sensor on the SAME comp stays on…
    assert!(ov.visible("cam_link", "depth_cam"));
    // …and the same sensor NAME on a different comp stays on (the key is the full pair)…
    assert!(ov.visible("other_link", "front_cam"));
    // …and an unrelated pair stays on.
    assert!(ov.visible("lidar_link", "scan"));
}

#[test]
fn sensor_viz_overrides_explicit_true_is_on() {
    let mut ov = SensorVizOverrides::default();
    ov.0.insert(
        ("imu_link".to_string(), "imu0".to_string()),
        SensorVizState {
            visible: true,
            ..Default::default()
        },
    );
    // An explicit `visible: true` reads the same as the default-on absence.
    assert!(ov.visible("imu_link", "imu0"));
    // Default `SensorVizState` is on, matching the "absent ⇒ on" convention.
    assert!(SensorVizState::default().visible);
}

#[test]
fn sensor_viz_full_extent_defaults_off_and_targets_one_sensor() {
    // Absent ⇒ capped (full_extent false), matching the "absent ⇒ follow the global" convention.
    assert!(!SensorVizState::default().full_extent);
    let mut ov = SensorVizOverrides::default();
    assert!(!ov.full_extent("lidar_link", "scan"));
    ov.0.insert(
        ("lidar_link".to_string(), "scan".to_string()),
        SensorVizState {
            visible: true,
            full_extent: true,
        },
    );
    // The keyed sensor now draws uncapped…
    assert!(ov.full_extent("lidar_link", "scan"));
    // …but a different sensor (even on the same comp) still follows the cap…
    assert!(!ov.full_extent("lidar_link", "other"));
    // …and toggling full_extent leaves `visible` independent.
    assert!(ov.visible("lidar_link", "scan"));
}

#[test]
fn sensor_viz_effective_full_extent_is_global_or_per_sensor() {
    // The scene ORs the global toggle with the per-sensor override; model that composition directly.
    let mut ov = SensorVizOverrides::default();
    ov.0.insert(
        ("a".to_string(), "s".to_string()),
        SensorVizState {
            visible: true,
            full_extent: true,
        },
    );
    let effective = |global: bool, comp: &str, sensor: &str| global || ov.full_extent(comp, sensor);
    // Global OFF: only the per-sensor override draws uncapped.
    assert!(effective(false, "a", "s"));
    assert!(!effective(false, "b", "s"));
    // Global ON: every sensor draws uncapped regardless of per-sensor state.
    assert!(effective(true, "b", "s"));
}

// ── selection bounds: the pure geometry-only AABB helpers ────────────────────
//
// The highlight box must wrap ONLY the comp's real geometry (visual + shown collision), computed in
// comp-LOCAL space so joint articulation cannot swell it. These cover the include filter and the
// local-space union headless (no World/GPU).

use crate::scene::{include_in_selection_bounds, union_local_aabbs};
use bevy::math::Affine3A;

#[test]
fn selection_bounds_include_only_shown_geometry() {
    // Visual and collision items count while shown…
    assert!(include_in_selection_bounds(true, false, false));
    assert!(include_in_selection_bounds(false, true, false));
    // …but not when hidden (a toggled-off overlay must not size the box)…
    assert!(!include_in_selection_bounds(true, false, true));
    assert!(!include_in_selection_bounds(false, true, true));
    // …and annotation glyphs (sensor frusta, connectors, CG marker, selection triad) and child-comp
    // subtrees (anything without a geometry marker) never count, shown or hidden.
    assert!(!include_in_selection_bounds(false, false, false));
    assert!(!include_in_selection_bounds(false, false, true));
}

#[test]
fn union_local_aabbs_empty_is_none() {
    // No measurable mesh AABBs (e.g. glTF still loading) → None, so the caller can fall back.
    assert!(union_local_aabbs(Vec::<(Affine3A, Vec3, Vec3)>::new()).is_none());
}

#[test]
fn union_local_aabbs_identity_entry_is_exact() {
    let center = Vec3::new(1.0, 2.0, 3.0);
    let he = Vec3::new(0.5, 1.0, 2.0);
    let (min, max) = union_local_aabbs([(Affine3A::IDENTITY, center, he)]).unwrap();
    assert!(close(min, center - he), "min {min:?}");
    assert!(close(max, center + he), "max {max:?}");
}

#[test]
fn union_local_aabbs_applies_relative_transform() {
    // A mesh 90°-rotated about Z relative to the comp swaps its X/Y extents in comp-local space;
    // a translation offsets the bound. Both must come from the transformed corners.
    let rel = Affine3A::from_rotation_translation(
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        Vec3::new(5.0, 0.0, 0.0),
    );
    let (min, max) = union_local_aabbs([(rel, Vec3::ZERO, Vec3::new(2.0, 1.0, 0.5))]).unwrap();
    assert!(close(min, Vec3::new(4.0, -2.0, -0.5)), "min {min:?}");
    assert!(close(max, Vec3::new(6.0, 2.0, 0.5)), "max {max:?}");
}

#[test]
fn union_local_aabbs_unions_disjoint_boxes() {
    let a = (
        Affine3A::from_translation(Vec3::new(-2.0, 0.0, 0.0)),
        Vec3::ZERO,
        Vec3::splat(0.5),
    );
    let b = (
        Affine3A::from_translation(Vec3::new(3.0, 1.0, 0.0)),
        Vec3::ZERO,
        Vec3::splat(0.5),
    );
    let (min, max) = union_local_aabbs([a, b]).unwrap();
    assert!(close(min, Vec3::new(-2.5, -0.5, -0.5)), "min {min:?}");
    assert!(close(max, Vec3::new(3.5, 1.5, 0.5)), "max {max:?}");
}

#[test]
fn union_local_aabbs_invariant_under_joint_articulation() {
    // Articulating a joint moves the comp and its meshes TOGETHER: the mesh's affine RELATIVE to the
    // comp (the union's input) is q-independent, so the box cannot grow as a wheel spins (the old
    // world-space union did). Model it: mesh world = comp world · L with L fixed; rel = comp⁻¹ · mesh.
    let local = Affine3A::from_rotation_translation(
        Quat::from_rotation_x(0.3),
        Vec3::new(0.0, -0.015, 0.0),
    );
    let he = Vec3::new(0.0365, 0.015, 0.0365);
    let mut results = Vec::new();
    for q in [0.0_f32, 0.8, -1.6] {
        let comp = Affine3A::from_quat(Quat::from_rotation_y(q));
        let rel = comp.inverse() * (comp * local);
        results.push(union_local_aabbs([(rel, Vec3::ZERO, he)]).unwrap());
    }
    for (min, max) in &results[1..] {
        assert!(
            close(*min, results[0].0),
            "min drifted under articulation: {min:?}"
        );
        assert!(
            close(*max, results[0].1),
            "max drifted under articulation: {max:?}"
        );
    }
}

// ── multi-highlight: HighlightSet plumbing + the pure bounds→box transform ────
//
// `draw_highlight` paints the plain `Selected` yellow, then one bounds box per `HighlightSet` entry in
// its own color. These cover the shared box-transform math and the set/color plumbing headless.

use crate::pick::HighlightSet;
use crate::scene::highlight_box_transform;

#[test]
fn highlight_set_defaults_empty_and_carries_colors() {
    // Empty default ⇒ draw_highlight adds nothing beyond the yellow `Selected` box (today's behaviour).
    assert!(HighlightSet::default().0.is_empty());
    // Entries carry their per-comp color through untouched (parent=orange, child=cyan in the embedder).
    let orange = Color::srgb(1.0, 0.5, 0.0);
    let cyan = Color::srgb(0.0, 0.8, 1.0);
    let set = HighlightSet(vec![(ent(1), orange), (ent(2), cyan)]);
    assert_eq!(set.0.len(), 2);
    assert_eq!(set.0[0], (ent(1), orange));
    assert_eq!(set.0[1].1, cyan);
}

#[test]
fn highlight_box_transform_hugs_measured_bounds() {
    // A comp-local box (-1,-1,-1)..(1,1,1): the cube centers on the box center and scales to the full
    // extent plus a hairline pad, composed under the comp transform.
    let comp = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
    let t = highlight_box_transform(comp, Some((Vec3::splat(-1.0), Vec3::splat(1.0))));
    assert!(
        close(t.translation, Vec3::new(10.0, 0.0, 0.0)),
        "t {:?}",
        t.translation
    );
    // half=1, pad=0.02 → scale=(1.02)*2=2.04 each axis.
    assert!(close(t.scale, Vec3::splat(2.04)), "scale {:?}", t.scale);
}

#[test]
fn highlight_box_transform_pad_floors_for_tiny_box() {
    // Off-center, degenerate (zero-extent) box: pad floors at 0.001 so it still gets a visible margin.
    let t = highlight_box_transform(
        Transform::IDENTITY,
        Some((Vec3::splat(2.0), Vec3::splat(2.0))),
    );
    assert!(
        close(t.translation, Vec3::splat(2.0)),
        "center {:?}",
        t.translation
    );
    // half=0 → pad=max(0,0.001)=0.001 → scale=0.002.
    assert!(close(t.scale, Vec3::splat(0.002)), "scale {:?}", t.scale);
}

#[test]
fn highlight_box_transform_fallback_when_unmeasured() {
    // No measurable geometry (glTF still loading): a fixed 0.08 box at the comp origin, comp scale
    // discarded (matches the old single-box fallback).
    let comp = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0)).with_scale(Vec3::splat(5.0));
    let t = highlight_box_transform(comp, None);
    assert!(
        close(t.translation, Vec3::new(1.0, 2.0, 3.0)),
        "t {:?}",
        t.translation
    );
    assert!(close(t.scale, Vec3::splat(0.08)), "scale {:?}", t.scale);
}

// ── item 4: joint articulation kinematics (pure helpers) ─────────────────────
//
// joint_local_transform applies the joint motion AFTER the fixed origin (URDF convention), and
// resolve_q maps the commanded-position source of truth through mimic + limit + continuous handling.
use crate::joints::{
    joint_local_transform, make_articulated_joint, resolve_all, resolve_q, ArticulatedJoint,
    JointKind, JointMimic, JointPositions,
};

const PI: f32 = std::f32::consts::PI;
const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;

/// Build a bare [`ArticulatedJoint`] for resolve_q tests (the child entity is unused there).
fn aj(name: &str, kind: JointKind, lower: Option<f32>, upper: Option<f32>) -> ArticulatedJoint {
    ArticulatedJoint {
        name: name.to_string(),
        child: Entity::PLACEHOLDER,
        origin: Transform::IDENTITY,
        axis: Vec3::Z,
        axis2: Vec3::Y,
        kind,
        lower: [lower, None, None],
        upper: [upper, None, None],
        mimic: None,
    }
}

#[test]
fn revolute_rotates_about_axis_after_identity_origin() {
    // +Z by π/2 with identity origin: the child's +X should rotate onto +Y.
    let t = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Revolute,
        &[HALF_PI],
    );
    assert!(
        close(t.rotation * Vec3::X, Vec3::Y),
        "got {:?}",
        t.rotation * Vec3::X
    );
    assert!(
        close(t.translation, Vec3::ZERO),
        "revolute must not translate"
    );
}

#[test]
fn continuous_rotates_like_revolute() {
    // Continuous shares revolute's rotation kinematics (it only differs in being unbounded).
    let t = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Continuous,
        &[HALF_PI],
    );
    assert!(
        close(t.rotation * Vec3::X, Vec3::Y),
        "got {:?}",
        t.rotation * Vec3::X
    );
}

#[test]
fn prismatic_translates_along_axis() {
    // +Z by d with identity origin: the child slides to (0,0,d) with no rotation.
    let d = 0.37;
    let t = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Prismatic,
        &[d],
    );
    assert!(
        close(t.translation, Vec3::new(0.0, 0.0, d)),
        "got {:?}",
        t.translation
    );
    assert!(
        t.rotation.abs_diff_eq(Quat::IDENTITY, EPS),
        "prismatic must not rotate"
    );
}

#[test]
fn motion_composes_after_origin() {
    // Non-identity origin: the joint motion is applied in the origin's frame (origin · motion).
    // Origin translates +X by 2 and rotates +90° about Z; a prismatic +X by 1 then moves along the
    // origin's rotated X (= world +Y), landing at (2,1,0), not (3,0,0).
    let origin = Transform::from_translation(Vec3::new(2.0, 0.0, 0.0))
        .with_rotation(Quat::from_rotation_z(HALF_PI));
    let t = joint_local_transform(origin, Vec3::X, Vec3::Y, JointKind::Prismatic, &[1.0]);
    assert!(
        close(t.translation, Vec3::new(2.0, 1.0, 0.0)),
        "got {:?}",
        t.translation
    );
    // And it equals the explicit mul_transform composition (the documented order).
    let expected = origin.mul_transform(Transform::from_translation(Vec3::X));
    assert!(close(t.translation, expected.translation));
    assert!(t.rotation.abs_diff_eq(expected.rotation, EPS));
}

#[test]
fn fixed_and_other_hold_at_origin() {
    let origin = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0))
        .with_rotation(Quat::from_rotation_y(0.5));
    for kind in [JointKind::Fixed, JointKind::Other] {
        // q is ignored entirely for non-articulating kinds.
        let t = joint_local_transform(origin, Vec3::Z, Vec3::Y, kind, &[1.234]);
        assert!(
            close(t.translation, origin.translation),
            "{kind:?} moved translation"
        );
        assert!(
            t.rotation.abs_diff_eq(origin.rotation, EPS),
            "{kind:?} moved rotation"
        );
    }
}

#[test]
fn axis_is_normalized_on_parse() {
    // A non-unit axis "0 0 2" must normalize to +Z, so a π/2 revolute behaves identically to unit +Z.
    let joint = crate::schema::Joint {
        name: Some("j".into()),
        type_: Some(crate::schema::model::enums::JointType::Revolute),
        axis: Some(crate::schema::model::joint::Axis {
            xyz: Some("0 0 2".into()),
        }),
        ..Default::default()
    };
    let j = make_articulated_joint(&joint, Entity::PLACEHOLDER, Transform::IDENTITY);
    assert!(close(j.axis, Vec3::Z), "axis not normalized: {:?}", j.axis);
    let t = joint_local_transform(j.origin, j.axis, j.axis2, j.kind, &[HALF_PI]);
    assert!(
        close(t.rotation * Vec3::X, Vec3::Y),
        "got {:?}",
        t.rotation * Vec3::X
    );
}

#[test]
fn zero_length_axis_falls_back_to_x() {
    // "0 0 0" (and a missing axis) collapse to the URDF default +X.
    let zero = crate::schema::Joint {
        type_: Some(crate::schema::model::enums::JointType::Revolute),
        axis: Some(crate::schema::model::joint::Axis {
            xyz: Some("0 0 0".into()),
        }),
        ..Default::default()
    };
    assert!(close(
        make_articulated_joint(&zero, Entity::PLACEHOLDER, Transform::IDENTITY).axis,
        Vec3::X
    ));
    let none = crate::schema::Joint {
        type_: Some(crate::schema::model::enums::JointType::Revolute),
        ..Default::default()
    };
    assert!(close(
        make_articulated_joint(&none, Entity::PLACEHOLDER, Transform::IDENTITY).axis,
        Vec3::X
    ));
}

#[test]
fn resolve_q_clamps_revolute_with_both_limits() {
    let j = aj("j", JointKind::Revolute, Some(-0.5), Some(0.5));
    let mut p = JointPositions::default();
    p.set_dof("j", 0, 2.0); // way past the upper limit.
    assert!((resolve_q(&j, &p) - 0.5).abs() < EPS, "must clamp to upper");
    p.set_dof("j", 0, -2.0);
    assert!((resolve_q(&j, &p) + 0.5).abs() < EPS, "must clamp to lower");
    p.set_dof("j", 0, 0.1);
    assert!(
        (resolve_q(&j, &p) - 0.1).abs() < EPS,
        "in-range value passes through"
    );
}

#[test]
fn resolve_q_does_not_clamp_continuous_or_one_sided() {
    // Continuous is unbounded even if limits happen to be present.
    let mut cont = aj("c", JointKind::Continuous, Some(-0.5), Some(0.5));
    cont.kind = JointKind::Continuous;
    let mut p = JointPositions::default();
    p.set_dof("c", 0, 10.0);
    assert!(
        (resolve_q(&cont, &p) - 10.0).abs() < EPS,
        "continuous must not clamp"
    );
    // A revolute with only ONE bound present is not clamped (both must be present).
    let one = aj("o", JointKind::Revolute, Some(-0.5), None);
    p.set_dof("o", 0, 10.0);
    assert!(
        (resolve_q(&one, &p) - 10.0).abs() < EPS,
        "one-sided limit must not clamp"
    );
}

#[test]
fn resolve_q_missing_position_is_zero() {
    let j = aj("absent", JointKind::Revolute, None, None);
    let p = JointPositions::default();
    assert!(
        resolve_q(&j, &p).abs() < EPS,
        "missing command defaults to zero pose"
    );
}

#[test]
fn slider_range_sentinel_huge_limits_fall_back_to_pi() {
    use crate::ui::joint_slider_range;
    // A sanely-bounded joint keeps its declared range.
    assert_eq!(
        joint_slider_range(JointKind::Revolute, Some(-1.0), Some(2.0)),
        -1.0..=2.0
    );
    // SDF "revolute standing in for continuous": sentinel ±1e16 limits (also the f32-overflowed
    // f64::MAX case, which parses to None) must get the continuous-style ±π slider, not a ±1e16 one.
    assert_eq!(
        joint_slider_range(JointKind::Revolute, Some(-1e16), Some(1e16)),
        -PI..=PI
    );
    // Even one out-of-scale bound falls back (a half-sentinel range is just as undraggable).
    assert_eq!(
        joint_slider_range(JointKind::Prismatic, Some(0.0), Some(1e16)),
        -PI..=PI
    );
    // Continuous is always ±π, whatever limits happen to be present.
    assert_eq!(
        joint_slider_range(JointKind::Continuous, Some(-1.0), Some(1.0)),
        -PI..=PI
    );
    // Missing or inverted bounds keep the existing limitless fallback.
    assert_eq!(
        joint_slider_range(JointKind::Revolute, Some(-0.5), None),
        -PI..=PI
    );
    assert_eq!(
        joint_slider_range(JointKind::Revolute, Some(1.0), Some(-1.0)),
        -PI..=PI
    );
}

#[test]
fn resolve_q_mimic_follows_source() {
    // mimic: q = multiplier * q(source) + offset.
    let mut j = aj("follower", JointKind::Revolute, None, None);
    j.mimic = Some(JointMimic {
        source: "driver".into(),
        multiplier: -2.0,
        offset: 0.25,
    });
    let mut p = JointPositions::default();
    p.set_dof("driver", 0, 1.0);
    assert!(
        (resolve_q(&j, &p) - (-1.75)).abs() < EPS,
        "got {}",
        resolve_q(&j, &p)
    );
    // The follower's OWN map entry is ignored when it mimics.
    p.set_dof("follower", 0, 99.0);
    assert!(
        (resolve_q(&j, &p) - (-1.75)).abs() < EPS,
        "mimic must ignore own command"
    );
}

#[test]
fn resolve_all_chains_mimic_of_mimic() {
    // C is directly driven; B mimics C; A mimics B. A must follow C's motion through B, even though
    // neither A nor B is ever written into JointPositions (mimic joints get no slider entry).
    //   q(C) = 0.5
    //   q(B) = 3*q(C) + 0.1 = 1.6
    //   q(A) = 2*q(B) + 0.2 = 3.4
    let c = aj("c", JointKind::Revolute, None, None);
    let mut b = aj("b", JointKind::Revolute, None, None);
    b.mimic = Some(JointMimic {
        source: "c".into(),
        multiplier: 3.0,
        offset: 0.1,
    });
    let mut a = aj("a", JointKind::Revolute, None, None);
    a.mimic = Some(JointMimic {
        source: "b".into(),
        multiplier: 2.0,
        offset: 0.2,
    });
    let joints = vec![a, b, c];

    let mut p = JointPositions::default();
    p.set_dof("c", 0, 0.5);
    let r = resolve_all(&joints, &p);
    assert!((r["c"][0] - 0.5).abs() < EPS, "c got {}", r["c"][0]);
    assert!((r["b"][0] - 1.6).abs() < EPS, "b got {}", r["b"][0]);
    assert!(
        (r["a"][0] - 3.4).abs() < EPS,
        "a (mimic-of-mimic) got {}",
        r["a"][0]
    );
}

#[test]
fn resolve_all_chain_applies_source_limits_before_following() {
    // A mimic follows its source's RESOLVED q, i.e. AFTER the source's own clamp. C is clamped to
    // [−1, 1]; commanded 5.0 → resolved 1.0. B mimics C with multiplier 2 → q(B) = 2.0 (not 10.0).
    let c = aj("c", JointKind::Revolute, Some(-1.0), Some(1.0));
    let mut b = aj("b", JointKind::Revolute, None, None);
    b.mimic = Some(JointMimic {
        source: "c".into(),
        multiplier: 2.0,
        offset: 0.0,
    });
    let joints = vec![b, c];

    let mut p = JointPositions::default();
    p.set_dof("c", 0, 5.0); // clamps to 1.0
    let r = resolve_all(&joints, &p);
    assert!((r["c"][0] - 1.0).abs() < EPS, "c clamp got {}", r["c"][0]);
    assert!(
        (r["b"][0] - 2.0).abs() < EPS,
        "b must follow c's clamped value, got {}",
        r["b"][0]
    );
}

#[test]
fn resolve_all_breaks_mimic_cycle_without_infinite_loop() {
    // A↔B mutual mimic is malformed; resolution must terminate (cycle contribution treated as 0.0)
    // rather than recurse forever. We only assert it returns finite values for both joints.
    let mut a = aj("a", JointKind::Revolute, None, None);
    a.mimic = Some(JointMimic {
        source: "b".into(),
        multiplier: 1.0,
        offset: 0.0,
    });
    let mut b = aj("b", JointKind::Revolute, None, None);
    b.mimic = Some(JointMimic {
        source: "a".into(),
        multiplier: 1.0,
        offset: 0.0,
    });
    let joints = vec![a, b];

    let p = JointPositions::default();
    let r = resolve_all(&joints, &p);
    assert!(
        r["a"][0].is_finite() && r["b"][0].is_finite(),
        "cycle must resolve to finite values"
    );
}

#[test]
fn resolve_all_self_mimic_terminates() {
    // A joint mimicking itself is degenerate; resolution must terminate (self-reference contributes
    // 0.0 once revisited) instead of recursing forever.
    let mut a = aj("a", JointKind::Revolute, None, None);
    a.mimic = Some(JointMimic {
        source: "a".into(),
        multiplier: 1.0,
        offset: 0.3,
    });
    let joints = vec![a];
    let p = JointPositions::default();
    let r = resolve_all(&joints, &p);
    // The entry recurses once into itself, then the cycle guard fires: inner = 1.0*0.0 + 0.3 = 0.3,
    // outer = 1.0*0.3 + 0.3 = 0.6. The only contract that matters is that it TERMINATES with a finite
    // value; 0.6 is that deterministic terminating result.
    assert!(
        r["a"][0].is_finite(),
        "self-mimic must terminate finitely, got {}",
        r["a"][0]
    );
    assert!(
        (r["a"][0] - 0.6).abs() < EPS,
        "self-mimic got {}",
        r["a"][0]
    );
}

#[test]
fn movable_classification_excludes_fixed_and_other() {
    assert!(JointKind::Revolute.is_movable());
    assert!(JointKind::Continuous.is_movable());
    assert!(JointKind::Prismatic.is_movable());
    assert!(JointKind::Screw { pitch: 0.01 }.is_movable());
    // The multi-DOF kinds are movable too, with their DOF counts.
    assert!(JointKind::Cylindrical.is_movable());
    assert!(JointKind::Universal.is_movable());
    assert!(JointKind::Planar.is_movable());
    assert!(JointKind::Ball.is_movable());
    assert_eq!(JointKind::Revolute.dof_count(), 1);
    assert_eq!(JointKind::Cylindrical.dof_count(), 2);
    assert_eq!(JointKind::Universal.dof_count(), 2);
    assert_eq!(JointKind::Planar.dof_count(), 2);
    assert_eq!(JointKind::Ball.dof_count(), 3);
    assert_eq!(JointKind::Fixed.dof_count(), 0);
    assert!(!JointKind::Fixed.is_movable());
    assert!(!JointKind::Other.is_movable());
    // The typed schema kinds map across; free/unknown/absent ⇒ Other.
    use crate::schema::model::enums::JointType;
    assert_eq!(
        JointKind::from_schema(Some(JointType::Revolute), None),
        JointKind::Revolute
    );
    assert_eq!(
        JointKind::from_schema(Some(JointType::Planar), None),
        JointKind::Planar
    );
    assert_eq!(
        JointKind::from_schema(Some(JointType::Ball), None),
        JointKind::Ball
    );
    assert_eq!(
        JointKind::from_schema(Some(JointType::Free), None),
        JointKind::Other
    );
    assert_eq!(JointKind::from_schema(None, None), JointKind::Other);
    // Screw carries its parsed pitch (metres/rev); absent/unparsable ⇒ SCREW_PITCH_DEFAULT.
    assert_eq!(
        JointKind::from_schema(Some(JointType::Screw), Some("0.002")),
        JointKind::Screw { pitch: 0.002 }
    );
    assert_eq!(
        JointKind::from_schema(Some(JointType::Screw), None),
        JointKind::Screw {
            pitch: crate::joints::SCREW_PITCH_DEFAULT
        }
    );
}

#[test]
fn full_turn_revolute_returns_near_identity() {
    // A continuous joint commanded a full 2π lands back at the origin orientation (sanity on wrap).
    let t = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Continuous,
        &[2.0 * PI],
    );
    assert!(
        close(t.rotation * Vec3::X, Vec3::X),
        "got {:?}",
        t.rotation * Vec3::X
    );
}

// ── item 5: multi-DOF joint kinematics ───────────────────────────────────────
//
// cylindrical (rotate + independent slide), universal (two crossed-axis rotations, order-sensitive),
// planar (translation in a normal-derived in-plane basis), ball (intrinsic roll/pitch/yaw), plus the
// make_articulated_joint DOF-limit mapping (cylindrical's single <limit> bounds BOTH DOFs).

#[test]
fn cylindrical_rotates_then_slides_independently() {
    // axis +Z: q0 = π/2 rotation about Z (X→Y), q1 = 0.5 m slide along Z. Rotation and translation are
    // independent (unlike screw), so the slide is exactly q1 with no pitch coupling.
    let t = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Cylindrical,
        &[HALF_PI, 0.5],
    );
    assert!(
        close(t.rotation * Vec3::X, Vec3::Y),
        "cylindrical rotation {:?}",
        t.rotation * Vec3::X
    );
    assert!(
        close(t.translation, Vec3::new(0.0, 0.0, 0.5)),
        "cylindrical slide {:?}",
        t.translation
    );
}

#[test]
fn universal_composes_axis_then_axis2_in_order() {
    // axis +Z (q0), axis2 +X (q1), both π/2. Intrinsic order (q0 about axis, THEN q1 about axis2 in the
    // rotated frame) ⇒ the net rotation cycles X→Y→Z→X. The reversed order would NOT give this cycle,
    // so these three mappings pin the composition order.
    let t = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::X,
        JointKind::Universal,
        &[HALF_PI, HALF_PI],
    );
    assert!(
        close(t.rotation * Vec3::X, Vec3::Y),
        "X→Y, got {:?}",
        t.rotation * Vec3::X
    );
    assert!(
        close(t.rotation * Vec3::Y, Vec3::Z),
        "Y→Z, got {:?}",
        t.rotation * Vec3::Y
    );
    assert!(
        close(t.rotation * Vec3::Z, Vec3::X),
        "Z→X, got {:?}",
        t.rotation * Vec3::Z
    );
    // Explicitly distinct from the reversed composition (axis2 first, then axis).
    let reversed = Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, HALF_PI)).mul_transform(
        Transform::from_rotation(Quat::from_axis_angle(Vec3::X, HALF_PI)),
    );
    // (sanity: our transform equals origin·R(axis)·R(axis2), not R(axis2)·R(axis))
    assert!(
        t.rotation.abs_diff_eq(reversed.rotation, EPS),
        "universal must be axis-then-axis2"
    );
    let swapped = Transform::from_rotation(Quat::from_axis_angle(Vec3::X, HALF_PI)).mul_transform(
        Transform::from_rotation(Quat::from_axis_angle(Vec3::Z, HALF_PI)),
    );
    assert!(
        !t.rotation.abs_diff_eq(swapped.rotation, EPS),
        "order must matter: axis2-then-axis differs"
    );
}

#[test]
fn planar_translates_in_normal_derived_in_plane_basis() {
    // Normal +Z: the two DOFs translate along an orthonormal in-plane basis (u = Z×X = +Y, v = Z×u =
    // −X). q0=0.3 along u, q1=0.4 along v ⇒ (−0.4, 0.3, 0); strictly in the plane (z = 0), no rotation.
    let t = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Planar,
        &[0.3, 0.4],
    );
    assert!(
        close(t.translation, Vec3::new(-0.4, 0.3, 0.0)),
        "planar translation {:?}",
        t.translation
    );
    // In-plane: the motion has no component along the normal.
    assert!(
        t.translation.dot(Vec3::Z).abs() < EPS,
        "planar motion must stay in the plane"
    );
    assert!(
        t.rotation.abs_diff_eq(Quat::IDENTITY, EPS),
        "planar must not rotate"
    );
}

#[test]
fn ball_composes_intrinsic_roll_pitch_yaw() {
    // Each single DOF is a rotation about the joint frame's X (roll), Y (pitch), Z (yaw).
    let roll = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Ball,
        &[HALF_PI, 0.0, 0.0],
    );
    assert!(
        close(roll.rotation * Vec3::Y, Vec3::Z),
        "roll: Y→Z, got {:?}",
        roll.rotation * Vec3::Y
    );
    let pitch = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Ball,
        &[0.0, HALF_PI, 0.0],
    );
    assert!(
        close(pitch.rotation * Vec3::Z, Vec3::X),
        "pitch: Z→X, got {:?}",
        pitch.rotation * Vec3::Z
    );
    let yaw = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Ball,
        &[0.0, 0.0, HALF_PI],
    );
    assert!(
        close(yaw.rotation * Vec3::X, Vec3::Y),
        "yaw: X→Y, got {:?}",
        yaw.rotation * Vec3::X
    );
    // Intrinsic composition (roll THEN pitch about the rolled frame): [π/2, π/2, 0] maps Z→X, which a
    // pure roll ([π/2,0,0], Z→−Y) does not, pinning the order.
    let rp = joint_local_transform(
        Transform::IDENTITY,
        Vec3::Z,
        Vec3::Y,
        JointKind::Ball,
        &[HALF_PI, HALF_PI, 0.0],
    );
    assert!(
        close(rp.rotation * Vec3::Z, Vec3::X),
        "intrinsic roll-then-pitch: Z→X, got {:?}",
        rp.rotation * Vec3::Z
    );
    assert!(
        !close(rp.rotation * Vec3::Z, roll.rotation * Vec3::Z),
        "composition order must matter"
    );
    // Ball never translates.
    assert!(close(rp.translation, Vec3::ZERO), "ball must not translate");
}

#[test]
fn make_articulated_joint_maps_per_dof_limits() {
    use crate::schema::model::enums::JointType;
    use crate::schema::model::joint::{Axis, JointLimit};
    // Cylindrical: <limit2> (rotation, rad) → DOF0; <limit> (translation, m) → DOF1, SEPARATE
    // unit-distinct bounds, not one shared range.
    let cyl = crate::schema::Joint {
        type_: Some(JointType::Cylindrical),
        axis: Some(Axis {
            xyz: Some("0 0 1".into()),
        }),
        limit: Some(JointLimit {
            lower: Some("0".into()), // translation → DOF1
            upper: Some("2".into()),
            ..Default::default()
        }),
        limit2: Some(JointLimit {
            lower: Some("-1".into()), // rotation → DOF0
            upper: Some("1".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let j = make_articulated_joint(&cyl, Entity::PLACEHOLDER, Transform::IDENTITY);
    assert_eq!(
        (j.lower[0], j.upper[0]),
        (Some(-1.0), Some(1.0)),
        "cylindrical DOF0 rotation from <limit2>"
    );
    assert_eq!(
        (j.lower[1], j.upper[1]),
        (Some(0.0), Some(2.0)),
        "cylindrical DOF1 translation from <limit>"
    );

    // Cylindrical rotation <limit2> is OPTIONAL: absent => DOF0 unbounded, translation still bounded.
    let cyl_norot = crate::schema::Joint {
        type_: Some(JointType::Cylindrical),
        axis: Some(Axis {
            xyz: Some("0 0 1".into()),
        }),
        limit: Some(JointLimit {
            lower: Some("0".into()),
            upper: Some("2".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let jn = make_articulated_joint(&cyl_norot, Entity::PLACEHOLDER, Transform::IDENTITY);
    assert_eq!(
        (jn.lower[0], jn.upper[0]),
        (None, None),
        "rotation unbounded"
    );
    assert_eq!((jn.lower[1], jn.upper[1]), (Some(0.0), Some(2.0)));

    // Universal: <limit> → DOF0, <limit2> → DOF1 (independent bounds).
    let uni = crate::schema::Joint {
        type_: Some(JointType::Universal),
        axis: Some(Axis {
            xyz: Some("0 0 1".into()),
        }),
        axis2: Some(Axis {
            xyz: Some("1 0 0".into()),
        }),
        limit: Some(JointLimit {
            lower: Some("-0.5".into()),
            upper: Some("0.5".into()),
            ..Default::default()
        }),
        limit2: Some(JointLimit {
            lower: Some("-1.0".into()),
            upper: Some("1.5".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let u = make_articulated_joint(&uni, Entity::PLACEHOLDER, Transform::IDENTITY);
    assert_eq!((u.lower[0], u.upper[0]), (Some(-0.5), Some(0.5)));
    assert_eq!((u.lower[1], u.upper[1]), (Some(-1.0), Some(1.5)));
    assert!(
        close(u.axis, Vec3::Z) && close(u.axis2, Vec3::X),
        "universal axes parsed"
    );

    // Ball: twist → DOF0 (rad, from <twist_limit>); swing2 → DOF1, swing1 → DOF2 (symmetric ±half-angle).
    use crate::schema::model::joint::SwingLimit;
    let ball = crate::schema::Joint {
        type_: Some(JointType::Ball),
        swing_limit: Some(SwingLimit {
            swing1: Some("0.4".into()),
            swing2: Some("0.2".into()),
            ..Default::default()
        }),
        twist_limit: Some(JointLimit {
            lower: Some("-0.3".into()),
            upper: Some("0.3".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let b = make_articulated_joint(&ball, Entity::PLACEHOLDER, Transform::IDENTITY);
    assert_eq!(
        (b.lower[0], b.upper[0]),
        (Some(-0.3), Some(0.3)),
        "twist DOF0"
    );
    assert_eq!(
        (b.lower[1], b.upper[1]),
        (Some(-0.2), Some(0.2)),
        "pitch DOF1 = ±swing2"
    );
    assert_eq!(
        (b.lower[2], b.upper[2]),
        (Some(-0.4), Some(0.4)),
        "yaw DOF2 = ±swing1"
    );
}

// ── hierarchy_rows: the shared comp-tree panel's pure layout helper ─────
//
// The whole point of the Hierarchy panel is reaching GEOMETRY-LESS comps (reference links / frames)
// that 3D mesh picking can never select. These cover the contract `hierarchy_panel` renders: a
// depth-first, indented row per comp, EVERY comp present exactly once (geometry-less included), correct
// depth and `has_geometry` flags, and floating geometry-less comps surfaced as their own roots.

use crate::hierarchy::hierarchy_rows;

/// base_footprint (no geometry) → base_link (visual) → wheel (collision), plus a stray geometry-less
/// `imu_link` that no joint references (a frame mount). Exercises every row variant the panel draws.
const HIERARCHY_DOC: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="hier" body-frame="FLU" world-frame="ENU">
  <comp name="base_footprint"/>
  <comp name="base_link">
    <visual name="body"><geometry><box><size>0.2 0.2 0.1</size></box></geometry></visual>
  </comp>
  <comp name="wheel">
    <collision name="wheel_col"><geometry><cylinder><radius>0.05</radius><length>0.04</length></cylinder></geometry></collision>
  </comp>
  <comp name="imu_link"/>
  <joint name="j_foot" type="fixed"><parent comp="base_footprint"/><child comp="base_link"/><origin xyz="0 0 0.05"/></joint>
  <joint name="j_wheel" type="continuous"><parent comp="base_link"/><child comp="wheel"/><origin xyz="0.1 0 0"/></joint>
</hcdf>"#;

/// Find a row by comp name (each comp appears exactly once, asserted separately).
fn row<'a>(
    rows: &'a [crate::hierarchy::HierarchyRow],
    name: &str,
) -> &'a crate::hierarchy::HierarchyRow {
    rows.iter()
        .find(|r| r.name == name)
        .unwrap_or_else(|| panic!("no row for {name}"))
}

#[test]
fn hierarchy_rows_orders_root_first_then_children_indented() {
    let h = parse(HIERARCHY_DOC);
    let rows = hierarchy_rows(&h);
    // The serial chain root comes first, then its descendants depth-first, deepest-last.
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    let foot = names.iter().position(|&n| n == "base_footprint").unwrap();
    let base = names.iter().position(|&n| n == "base_link").unwrap();
    let wheel = names.iter().position(|&n| n == "wheel").unwrap();
    assert!(
        foot < base && base < wheel,
        "depth-first chain order, got {names:?}"
    );
    // Depths: root=0, its child=1, grandchild=2.
    assert_eq!(row(&rows, "base_footprint").depth, 0);
    assert_eq!(row(&rows, "base_link").depth, 1);
    assert_eq!(row(&rows, "wheel").depth, 2);
}

#[test]
fn hierarchy_rows_flags_geometry_correctly() {
    let h = parse(HIERARCHY_DOC);
    let rows = hierarchy_rows(&h);
    // base_footprint + imu_link are reference frames (no visual/collision): the rows picking can't reach.
    assert!(
        !row(&rows, "base_footprint").has_geometry,
        "base_footprint is geometry-less"
    );
    assert!(
        !row(&rows, "imu_link").has_geometry,
        "imu_link is geometry-less"
    );
    // a visual OR a collision counts as geometry.
    assert!(
        row(&rows, "base_link").has_geometry,
        "base_link has a visual"
    );
    assert!(row(&rows, "wheel").has_geometry, "wheel has a collision");
}

#[test]
fn hierarchy_rows_lists_every_comp_exactly_once_including_geometryless() {
    let h = parse(HIERARCHY_DOC);
    let rows = hierarchy_rows(&h);
    // EVERY comp appears, none twice. comp_index is the authoritative identity.
    assert_eq!(rows.len(), h.comp.len(), "one row per comp, no more");
    let mut seen: Vec<usize> = rows.iter().map(|r| r.comp_index).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), h.comp.len(), "each comp listed exactly once");
    // The geometry-less ones are genuinely present (not silently dropped).
    assert!(rows.iter().any(|r| r.name == "base_footprint"));
    assert!(rows.iter().any(|r| r.name == "imu_link"));
}

#[test]
fn hierarchy_rows_surfaces_unjointed_geometryless_comp_as_root() {
    let h = parse(HIERARCHY_DOC);
    let rows = hierarchy_rows(&h);
    // imu_link is referenced by no joint → it's a root (depth 0), still selectable from the panel.
    assert_eq!(
        row(&rows, "imu_link").depth,
        0,
        "a comp with no incoming edge is a depth-0 root"
    );
}

// ── include flattening through the load pipeline ─────────────────────────────
//
// A parent HCDF that <include>s a module must, after the load path runs, expose the included module's
// (prefixed) comp, exactly the name a `CompEntity` carries (scene.rs builds `CompEntity { name }`
// straight from `comp.name`). We drive the real `load_hcdf_system` headlessly via a minimal App so the
// assertion exercises the actual integration point, not a hand-rolled flatten call.

use crate::doc::{HcdfDoc, LoadHcdf, SchemaStatus};

/// The set of comp names that `scene::rebuild_on_change` would stamp onto `CompEntity`s for `doc`.
fn comp_entity_names(doc: &HcdfDoc) -> Vec<String> {
    doc.0
        .as_ref()
        .map(|h| h.comp.iter().map(|c| c.name.clone()).collect())
        .unwrap_or_default()
}

#[test]
fn include_is_flattened_on_load_and_prefixed_comp_is_present() {
    // Lay down a module + a parent that includes it, in a temp dir, then load the parent by Path.
    let dir = std::env::temp_dir().join(format!("hcdviz_include_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("rotor.hcdf"),
        r#"<hcdf name="rotor-module" version="1.0"><comp name="rotor"/></hcdf>"#,
    )
    .unwrap();
    let parent = dir.join("airframe.hcdf");
    std::fs::write(
        &parent,
        r#"<hcdf name="airframe" version="1.0">
             <comp name="hub"/>
             <include uri="rotor.hcdf" name="left"/>
           </hcdf>"#,
    )
    .unwrap();

    // Minimal App carrying just the load pipeline's contract (resources + message + the system).
    let mut app = App::new();
    app.init_resource::<HcdfDoc>()
        .init_resource::<SchemaStatus>()
        .add_message::<LoadHcdf>();
    app.add_systems(Update, crate::doc::load_hcdf_system);
    app.world_mut()
        .write_message(LoadHcdf::Path(parent.clone()));
    app.update();

    let doc = app.world().resource::<HcdfDoc>();
    let names = comp_entity_names(doc);
    // The parent comp and the PREFIXED included comp are both in the CompEntity set.
    assert!(
        names.iter().any(|n| n == "hub"),
        "parent comp present; got {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "left/rotor"),
        "prefixed included comp present after flatten; got {names:?}"
    );
    // The include was consumed (flattened), not left dangling.
    assert!(
        doc.0.as_ref().unwrap().include.is_empty(),
        "include resolved away"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── warn-on-open raw-bytes XSD validation ─────────────────────────────────────
//
// Every open funnels its RAW xml through `load_hcdf_system`, which must flag a schema-invalid doc in
// `SchemaStatus::open_warning` while still loading whatever parsed (non-blocking by design). Driven
// headlessly through the real load pipeline, like the include test above.

#[test]
fn text_content_pose_doc_warns_on_open_and_still_loads() {
    let mut app = App::new();
    app.init_resource::<HcdfDoc>()
        .init_resource::<SchemaStatus>()
        .add_message::<LoadHcdf>();
    app.add_systems(Update, crate::doc::load_hcdf_system);

    // A legacy text-content pose: the typed parse ACCEPTS it (the pose silently comes back empty),
    // so only the raw-bytes XSD check can reveal it: exactly the class warn-on-open exists for.
    let legacy = r#"<hcdf name="legacy" version="1.0">
          <comp name="base"><frame name="mount"><pose>0 0 0.1 0 0 0</pose></frame></comp>
        </hcdf>"#;
    app.world_mut()
        .write_message(LoadHcdf::Xml(legacy.to_string()));
    app.update();

    let status = app.world().resource::<SchemaStatus>();
    assert!(
        status.open_warning.is_some(),
        "a text-content pose is schema-invalid and must set the open warning; status: {:?}",
        status.message
    );
    // Non-blocking: the doc still loaded with what parsed (comp present, pose silently empty).
    let doc = app.world().resource::<HcdfDoc>();
    let names = comp_entity_names(doc);
    assert!(
        names.iter().any(|n| n == "base"),
        "the doc still opens; got {names:?}"
    );

    // A subsequent VALID open clears the warning: the check is one-shot PER OPEN, never sticky.
    let valid = r#"<hcdf name="ok" version="1.0"><comp name="base"/></hcdf>"#;
    app.world_mut()
        .write_message(LoadHcdf::Xml(valid.to_string()));
    app.update();
    assert!(
        app.world()
            .resource::<SchemaStatus>()
            .open_warning
            .is_none(),
        "a schema-valid open must clear the previous warning"
    );
}

// ── sensor axis-align rotation (legacy dendrite semantics) ───────────────────
//
// Legacy dendrite's `DeviceAxisAlign::to_rotation_matrix()` returned the remap as ROWS
// `[x; y; z]` (row i = the body-frame direction raw axis i maps to); hcdviz's
// `axis_align_rotation` is its transpose (COLUMNS), so `m * raw_axis` applies the remap
// directly. These tests pin that equivalence against the legacy test vectors.

use crate::scene::axis_align_rotation;
use crate::schema::model::enums::AxisValue;
use crate::schema::model::AxisAlign;

#[test]
fn axis_align_absent_attributes_are_identity() {
    // The XSD defaults each attribute to "no remap"; an empty <axis-align/> is the identity.
    let m = axis_align_rotation(&AxisAlign::default());
    assert!(close(m * Vec3::X, Vec3::X));
    assert!(close(m * Vec3::Y, Vec3::Y));
    assert!(close(m * Vec3::Z, Vec3::Z));
}

#[test]
fn axis_align_remap_matches_legacy_rotation_matrix() {
    // The legacy dendrite-core test vector (a 90° yaw mount): x="Y" y="-X" z="Z" gave rows
    // [[0,1,0], [-1,0,0], [0,0,1]]. Ours must be exactly that matrix transposed.
    let a = AxisAlign {
        x: Some(AxisValue::Y),
        y: Some(AxisValue::NegX),
        z: Some(AxisValue::Z),
    };
    let m = axis_align_rotation(&a);
    let legacy_rows = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    for (i, row) in legacy_rows.into_iter().enumerate() {
        assert!(
            close(m.col(i), Vec3::from_array(row)),
            "legacy row {i} must be our column {i}: got {:?}, legacy {row:?}",
            m.col(i)
        );
    }
    // Applying the remap sends each raw axis where the legacy arrows pointed…
    assert!(close(m * Vec3::X, Vec3::Y));
    assert!(close(m * Vec3::Y, Vec3::NEG_X));
    assert!(close(m * Vec3::Z, Vec3::Z));
    // …and a signed-permutation remap is a proper rotation (det +1), as legacy assumed.
    assert!((m.determinant() - 1.0).abs() < EPS);
}

#[test]
fn axis_align_neg_y_remap_lands_raw_x_on_minus_y() {
    // x="-Y" y="X" z="Z": the raw X axis lands on body −Y (an IMU yawed −90° on the board).
    let a = AxisAlign {
        x: Some(AxisValue::NegY),
        y: Some(AxisValue::X),
        z: Some(AxisValue::Z),
    };
    let m = axis_align_rotation(&a);
    assert!(close(m * Vec3::X, Vec3::NEG_Y));
    assert!(close(m * Vec3::Y, Vec3::X));
    assert!(close(m * Vec3::Z, Vec3::Z));
    assert!((m.determinant() - 1.0).abs() < EPS);
}

#[test]
fn parsed_driver_axis_align_reaches_rotation() {
    // End-to-end through the real XML parse: the `<axis-align>` literals ("-Y" etc.) come back as
    // typed AxisValues and feed the same rotation the display draws from.
    let h = parse(
        r#"<?xml version="1.0"?>
<hcdf version="1.0" name="imu" body-frame="FLU" world-frame="ENU">
  <comp name="fmu">
    <sensor name="imu0">
      <inertial type="accel_gyro">
        <pose xyz="0.01 0 0.002"/>
        <driver name="icm45686"><axis-align x="-Y" y="X" z="Z"/></driver>
      </inertial>
    </sensor>
  </comp>
</hcdf>"#,
    );
    let align = h.comp[0].sensor[0].inertial[0]
        .driver
        .as_ref()
        .expect("driver parsed")
        .axis_align
        .as_ref()
        .expect("axis-align parsed");
    let m = axis_align_rotation(align);
    assert!(close(m * Vec3::X, Vec3::NEG_Y));
    assert!(close(m * Vec3::Y, Vec3::X));
    assert!(close(m * Vec3::Z, Vec3::Z));
}

// ── visual toggle groups: pure collection + show/hide decision ────────────────
//
// `<visual toggle="…">` groups (legacy per-group show/hide, e.g. a `case` over a bare PCB) are
// collected per doc and ANDed into the visual visibility decision. Both halves are pure and
// covered headless here (no World/GPU).

use crate::scene::{collect_toggle_groups, toggle_group_visible};
use bevy::platform::collections::HashSet;

#[test]
fn collect_toggle_groups_distinct_sorted_across_comps() {
    // Two comps; duplicate group names, an ungrouped visual, and an empty toggle to be ignored.
    let h = parse(
        r#"<?xml version="1.0"?>
<hcdf version="1.0" name="pcb" body-frame="FLU" world-frame="ENU">
  <comp name="board">
    <visual name="pcb_vis"><geometry><box size="0.1 0.1 0.01"/></geometry></visual>
    <visual name="case_top" toggle="case"><geometry><box size="0.1 0.1 0.02"/></geometry></visual>
    <visual name="lid" toggle="lid"><geometry><box size="0.1 0.1 0.005"/></geometry></visual>
    <visual name="blank" toggle=""><geometry><box size="0.01 0.01 0.01"/></geometry></visual>
  </comp>
  <comp name="mount">
    <visual name="case_bottom" toggle="case"><geometry><box size="0.1 0.1 0.02"/></geometry></visual>
  </comp>
</hcdf>"#,
    );
    // Distinct, sorted, empty-string dropped, duplicates ("case" in both comps) collapsed.
    assert_eq!(
        collect_toggle_groups(&h),
        vec!["case".to_string(), "lid".to_string()]
    );
}

#[test]
fn collect_toggle_groups_empty_when_none_declared() {
    let h = parse(
        r#"<?xml version="1.0"?>
<hcdf version="1.0" name="plain" body-frame="FLU" world-frame="ENU">
  <comp name="body"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
</hcdf>"#,
    );
    assert!(collect_toggle_groups(&h).is_empty());
}

#[test]
fn toggle_group_visible_truth_table() {
    let hidden: HashSet<String> = ["case".to_string()].into_iter().collect();
    // Ungrouped items are always shown, whatever is hidden.
    assert!(toggle_group_visible(None, &hidden));
    assert!(toggle_group_visible(None, &HashSet::default()));
    // A grouped item hides iff ITS group is in the hide set.
    assert!(!toggle_group_visible(Some("case"), &hidden));
    assert!(toggle_group_visible(Some("lid"), &hidden));
    // Default (nothing hidden) shows every group: the all-visible load state.
    assert!(toggle_group_visible(Some("case"), &HashSet::default()));
}

// ── selection hook: the embedder-only SelectedJoint out-param ─────────────────

#[test]
fn selected_joint_default_is_none() {
    // The standalone viewer never registers `SelectedJoint`; when an embedder does, it must start empty
    // (nothing selected) so `joints_panel` renders no active highlight until the user clicks a name.
    let sel = crate::pick::SelectedJoint::default();
    assert!(sel.0.is_none());
}

// ── egui style tuning: long combos must be visibly long ───────────────────────

#[test]
fn style_tuning_makes_long_combos_visible() {
    // The regression: egui's default combo popup (~200 px, auto-hiding scrollbar) showed ~4 rows of
    // the TEN-entry joint-type combo with no cue that it scrolls: the enum masqueraded as short.
    // The tuned style must fit the ten joint types outright and make any overflow visible (solid,
    // non-floating scrollbars).
    let ctx = bevy_egui::egui::Context::default();
    crate::ui::tune_egui_style(&ctx);
    let style = ctx.global_style();
    assert!(
        style.spacing.combo_height >= 400.0,
        "combo popups must fit ~16 rows, got {}",
        style.spacing.combo_height
    );
    assert!(
        !style.spacing.scroll.floating,
        "scrollbars must be solid (always visible), not floating/auto-hidden"
    );
}

#[test]
fn connectivity_enum_labels_match_schema_kebab_case() {
    use crate::ui::camel_case_to_kebab;

    assert_eq!(camel_case_to_kebab("PowerDelivery"), "power-delivery");
    assert_eq!(camel_case_to_kebab("GuidedOptical"), "guided-optical");
    assert_eq!(camel_case_to_kebab("ConductedRf"), "conducted-rf");
    assert_eq!(camel_case_to_kebab("WaveguideOpening"), "waveguide-opening");
}

#[test]
fn connectivity_instance_qualifier_distinguishes_included_objects() {
    use crate::schema::connectivity::{IdentityPart, ObjectIdentity, ObjectKind};
    use crate::schema::model::connectivity::{DocumentIdentity, IncludeInstanceId};
    use crate::ui::connectivity_instance_qualifier;

    let document = DocumentIdentity::new("memory://hcdviz/include-label").unwrap();
    let root = ObjectIdentity::new(
        document.clone(),
        IncludeInstanceId::root(),
        ObjectKind::Position,
        vec![IdentityPart::new("position", "1")],
    );
    assert_eq!(connectivity_instance_qualifier(&root), None);

    let included = ObjectIdentity::new(
        document,
        IncludeInstanceId::root().named_child("left", 0),
        ObjectKind::Position,
        vec![IdentityPart::new("position", "1")],
    );
    let qualifier = connectivity_instance_qualifier(&included).unwrap();
    assert!(qualifier.contains("left"));
    assert!(qualifier.contains("#0"));
}
