//! Headless integration test for slider-driven JOINT ARTICULATION (no GPU, no window).
//!
//! Builds a real 2-joint chain from HCDF, writes the source-of-truth [`JointPositions`] map (exactly
//! what a slider (or a future ROS/topic listener) would write), runs the app so [`articulate`] fires,
//! and asserts each driven child entity's local `Transform` equals the pure
//! [`hcdviz::joints::joint_local_transform`]. Also checks mimic following and the reload reset.
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::doc::HcdfDoc;
use hcdviz::joints::{
    joint_local_transform, ArticulatedJoints, JointKind, JointPositions, JointsPlugin,
};
use hcdviz::pick::Selected;
use hcdviz::scene::{CompEntity, ScenePlugin};
use hcdviz::schema::Hcdf;
use std::collections::HashMap;
use std::sync::Arc;

const EPS: f32 = 1e-4;

// base -> link1 (revolute about +Z, ±π/2 limit, origin +X 1m) -> link2 (prismatic along +Z, origin +Z
// 0.5m). A mimic joint drives link3 off link1 (multiplier 2, offset 0).
const CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="chain" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="link1"><visual name="v1"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <comp name="link2"><visual name="v2"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <comp name="link3"><visual name="v3"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="shoulder" type="revolute">
    <parent comp="base"/><child comp="link1"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent comp="link1"/><child comp="link2"/>
    <origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1"/>
  </joint>
  <joint name="finger" type="revolute">
    <parent comp="link1"/><child comp="link3"/>
    <origin xyz="0 1 0"/><axis xyz="0 0 1"/>
    <mimic joint="shoulder" multiplier="2" offset="0"/>
  </joint>
</hcdf>"#;

fn build_app(xml: &str) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin)
        .add_plugins(JointsPlugin);
    app.world_mut().resource_mut::<HcdfDoc>().0 = Some(Arc::new(Hcdf::from_xml_str(xml).unwrap()));
    app.update(); // rebuild_on_change builds the scene + ArticulatedJoints
    app.update(); // flush
    app
}

/// Map comp name → its spawned entity.
fn comp_entities(app: &mut App) -> HashMap<String, Entity> {
    let world = app.world_mut();
    let mut q = world.query::<(Entity, &CompEntity)>();
    q.iter(world).map(|(e, c)| (c.name.clone(), e)).collect()
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < EPS
}

#[test]
fn articulation_drives_child_transforms() {
    let mut app = build_app(CHAIN);

    // The catalogue must contain the three tree-edge joints (shoulder, slide, finger).
    {
        let aj = app.world().resource::<ArticulatedJoints>();
        assert_eq!(
            aj.0.len(),
            3,
            "expected 3 catalogued joints, got {}",
            aj.0.len()
        );
    }

    // Command the revolute and the prismatic via the single source of truth (as a slider would).
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("shoulder", 0, std::f32::consts::FRAC_PI_2);
        p.set_dof("slide", 0, 0.4);
    }
    app.update(); // articulate fires (JointPositions changed)

    let names = comp_entities(&mut app);

    // link1 (revolute +Z by π/2, origin +X 1): rotation maps +X→+Y, translation stays at origin.
    let t1 = *app
        .world()
        .entity(names["link1"])
        .get::<Transform>()
        .unwrap();
    let expect1 = joint_local_transform(
        Transform::from_translation(Vec3::X),
        Vec3::Z,
        Vec3::Y,
        JointKind::Revolute,
        &[std::f32::consts::FRAC_PI_2],
    );
    assert!(
        close(t1.translation, expect1.translation),
        "link1 t {:?}",
        t1.translation
    );
    assert!(
        t1.rotation.abs_diff_eq(expect1.rotation, EPS),
        "link1 r {:?}",
        t1.rotation
    );
    assert!(
        close(t1.rotation * Vec3::X, Vec3::Y),
        "link1 +X did not rotate to +Y"
    );

    // link2 (prismatic +Z by 0.4, origin +Z 0.5): translation = (0,0,0.9), no rotation.
    let t2 = *app
        .world()
        .entity(names["link2"])
        .get::<Transform>()
        .unwrap();
    assert!(
        close(t2.translation, Vec3::new(0.0, 0.0, 0.9)),
        "link2 t {:?}",
        t2.translation
    );
    assert!(
        t2.rotation.abs_diff_eq(Quat::IDENTITY, EPS),
        "link2 must not rotate"
    );
}

#[test]
fn mimic_joint_follows_multiplier_times_source() {
    let mut app = build_app(CHAIN);
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("shoulder", 0, 0.3); // finger mimics: q = 2*0.3 + 0 = 0.6.
    }
    app.update();

    let names = comp_entities(&mut app);
    let t3 = *app
        .world()
        .entity(names["link3"])
        .get::<Transform>()
        .unwrap();
    let expect3 = joint_local_transform(
        Transform::from_translation(Vec3::Y),
        Vec3::Z,
        Vec3::Y,
        JointKind::Revolute,
        &[0.6],
    );
    assert!(
        t3.rotation.abs_diff_eq(expect3.rotation, EPS),
        "mimic rotation {:?}",
        t3.rotation
    );
    assert!(
        close(t3.translation, Vec3::Y),
        "mimic origin translation {:?}",
        t3.translation
    );
}

// base -> link1 (revolute "shoulder", origin +X 1) -> link2 (revolute "mid" mimics shoulder ×2,
// origin +Z 1) -> link3 (revolute "tip" mimics mid ×3, origin +Y 1). A mimic-of-mimic chain:
// shoulder is the ONLY directly-driven joint; mid and tip are never written into JointPositions.
const MIMIC_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="mimicchain" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="link1"><visual name="v1"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <comp name="link2"><visual name="v2"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <comp name="link3"><visual name="v3"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="shoulder" type="revolute">
    <parent comp="base"/><child comp="link1"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
  </joint>
  <joint name="mid" type="revolute">
    <parent comp="link1"/><child comp="link2"/>
    <origin xyz="0 0 1"/><axis xyz="0 0 1"/>
    <mimic joint="shoulder" multiplier="2" offset="0"/>
  </joint>
  <joint name="tip" type="revolute">
    <parent comp="link2"/><child comp="link3"/>
    <origin xyz="0 1 0"/><axis xyz="0 0 1"/>
    <mimic joint="mid" multiplier="3" offset="0"/>
  </joint>
</hcdf>"#;

#[test]
fn mimic_of_mimic_follows_resolved_source() {
    // shoulder = 0.1 → mid = 2*0.1 = 0.2 → tip = 3*0.2 = 0.6. The tip must follow the RESOLVED mid
    // (whose name is never in JointPositions), not mid's absent raw command (which would give 0.0).
    let mut app = build_app(MIMIC_CHAIN);
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("shoulder", 0, 0.1);
    }
    app.update();

    let names = comp_entities(&mut app);

    let t2 = *app
        .world()
        .entity(names["link2"])
        .get::<Transform>()
        .unwrap();
    let expect2 = joint_local_transform(
        Transform::from_translation(Vec3::Z),
        Vec3::Z,
        Vec3::Y,
        JointKind::Revolute,
        &[0.2],
    );
    assert!(
        t2.rotation.abs_diff_eq(expect2.rotation, EPS),
        "mid rotation {:?}",
        t2.rotation
    );

    let t3 = *app
        .world()
        .entity(names["link3"])
        .get::<Transform>()
        .unwrap();
    let expect3 = joint_local_transform(
        Transform::from_translation(Vec3::Y),
        Vec3::Z,
        Vec3::Y,
        JointKind::Revolute,
        &[0.6],
    );
    assert!(
        t3.rotation.abs_diff_eq(expect3.rotation, EPS),
        "tip (mimic-of-mimic) rotation {:?}, expected q=0.6",
        t3.rotation
    );
}

#[test]
fn revolute_limit_is_clamped_on_apply() {
    let mut app = build_app(CHAIN);
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("shoulder", 0, 5.0); // way past +π/2 limit; must clamp to +π/2.
    }
    app.update();

    let names = comp_entities(&mut app);
    let t1 = *app
        .world()
        .entity(names["link1"])
        .get::<Transform>()
        .unwrap();
    let clamped = joint_local_transform(
        Transform::from_translation(Vec3::X),
        Vec3::Z,
        Vec3::Y,
        JointKind::Revolute,
        &[std::f32::consts::FRAC_PI_2],
    );
    assert!(
        t1.rotation.abs_diff_eq(clamped.rotation, EPS),
        "limit not clamped: {:?}",
        t1.rotation
    );
}

#[test]
fn reload_clears_joint_positions_and_resets_pose() {
    let mut app = build_app(CHAIN);
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("shoulder", 0, 1.0);
        p.set_dof("slide", 0, 0.5);
    }
    app.update();
    assert!(
        !app.world().resource::<JointPositions>().0.is_empty(),
        "precondition: posed"
    );

    // Reload a fresh document (Changed<HcdfDoc>): reset_on_reload must clear the commanded positions.
    // Assert the pose after EXACTLY ONE update: on the reload frame rebuild_on_change respawns the
    // entities, reset_on_reload clears the commands, and articulate (ordered .after(reset_on_reload))
    // must apply the ZERO pose to the fresh entities the SAME frame. A missing ordering would let
    // articulate read the stale commands and flash the previous pose for one frame; this single-update
    // assertion is what makes that transient visible.
    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(CHAIN).unwrap()));
    app.update();

    assert!(
        app.world().resource::<JointPositions>().0.is_empty(),
        "reload must clear JointPositions to the zero pose"
    );

    // With no commands, the reloaded link1 rests at its joint origin (+X 1, no rotation) on the very
    // first frame, never the previous articulated pose.
    let names = comp_entities(&mut app);
    let t1 = *app
        .world()
        .entity(names["link1"])
        .get::<Transform>()
        .unwrap();
    assert!(
        close(t1.translation, Vec3::X),
        "reset translation {:?}",
        t1.translation
    );
    assert!(
        t1.rotation.abs_diff_eq(Quat::IDENTITY, EPS),
        "reset rotation {:?}",
        t1.rotation
    );
}

// base -> nut (SCREW about +Z, pitch 0.5 m/rev, origin +X 1, no limits). A full revolution must
// advance the nut exactly one pitch along the axis while completing one turn.
const SCREW_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="screw" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="nut"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="lead" type="screw" thread_pitch="0.5">
    <parent comp="base"/><child comp="nut"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
  </joint>
</hcdf>"#;

#[test]
fn screw_full_turn_advances_one_pitch_and_rotates_a_full_turn() {
    let pitch = 0.5_f32;
    let mut app = build_app(SCREW_CHAIN);

    // The screw joint is catalogued as movable and carries its parsed pitch.
    {
        let aj = app.world().resource::<ArticulatedJoints>();
        assert_eq!(aj.0.len(), 1, "expected the single screw joint");
        assert_eq!(aj.0[0].kind, JointKind::Screw { pitch });
    }

    // Command one full revolution.
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("lead", 0, std::f32::consts::TAU);
    }
    app.update();

    let names = comp_entities(&mut app);
    let t = *app.world().entity(names["nut"]).get::<Transform>().unwrap();

    // Matches the pure kinematics exactly.
    let expect = joint_local_transform(
        Transform::from_translation(Vec3::X),
        Vec3::Z,
        Vec3::Y,
        JointKind::Screw { pitch },
        &[std::f32::consts::TAU],
    );
    assert!(
        close(t.translation, expect.translation),
        "screw t {:?}",
        t.translation
    );

    // Advanced exactly one pitch along +Z from the joint origin (+X 1) → (1, 0, pitch).
    assert!(
        close(t.translation, Vec3::new(1.0, 0.0, pitch)),
        "screw did not advance one pitch: {:?}",
        t.translation
    );
    // A full turn returns every axis to itself (net identity rotation).
    assert!(
        close(t.rotation * Vec3::X, Vec3::X),
        "screw X not full-turned"
    );
    assert!(
        close(t.rotation * Vec3::Y, Vec3::Y),
        "screw Y not full-turned"
    );
}

#[test]
fn screw_half_turn_advances_half_pitch() {
    let pitch = 0.5_f32;
    let mut app = build_app(SCREW_CHAIN);
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("lead", 0, std::f32::consts::PI); // half revolution
    }
    app.update();

    let names = comp_entities(&mut app);
    let t = *app.world().entity(names["nut"]).get::<Transform>().unwrap();
    // Half a turn ⇒ half a pitch of advance; the box also flipped 180° about +Z (X → −X).
    assert!(
        close(t.translation, Vec3::new(1.0, 0.0, pitch * 0.5)),
        "screw half-turn advance {:?}",
        t.translation
    );
    assert!(
        close(t.rotation * Vec3::X, Vec3::NEG_X),
        "screw half-turn rotation"
    );
}

// base -> slider (CYLINDRICAL about/along +Z, origin +X 1). By convention: <limit> is the
// TRANSLATION bound (m, DOF 1) and <limit2> is the ROTATION bound (rad, DOF 0): separate, unit-distinct.
// Here translation is bounded 0..1 m and rotation −0.5..0.5 rad.
const CYL_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="cyl" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="slider"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="cyl" type="cylindrical">
    <parent comp="base"/><child comp="slider"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1"/>
    <limit2 lower="-0.5" upper="0.5"/>
  </joint>
</hcdf>"#;

#[test]
fn cylindrical_translation_from_limit_rotation_from_limit2() {
    let mut app = build_app(CYL_CHAIN);

    // The cylindrical joint is a 2-DOF movable joint: DOF 0 (rotation) clamps from <limit2> (rad), DOF 1
    // (translation) from <limit> (m), NOT one shared bound.
    {
        let aj = app.world().resource::<ArticulatedJoints>();
        assert_eq!(aj.0.len(), 1, "expected the single cylindrical joint");
        assert_eq!(aj.0[0].kind, JointKind::Cylindrical);
        assert_eq!(aj.0[0].kind.dof_count(), 2);
        // DOF 0 = rotation ← <limit2> (−0.5..0.5 rad).
        assert_eq!(
            (aj.0[0].lower[0], aj.0[0].upper[0]),
            (Some(-0.5), Some(0.5)),
            "cyl DOF-0 rotation bound must come from <limit2>"
        );
        // DOF 1 = translation ← <limit> (0..1 m).
        assert_eq!(
            (aj.0[0].lower[1], aj.0[0].upper[1]),
            (Some(0.0), Some(1.0)),
            "cyl DOF-1 translation bound must come from <limit>"
        );
    }

    // Command DOF 0 (rotate +0.3 rad about +Z) and DOF 1 (slide +0.5 along +Z) via the one source of truth.
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("cyl", 0, 0.3);
        p.set_dof("cyl", 1, 0.5);
    }
    app.update();

    let names = comp_entities(&mut app);
    let t = *app
        .world()
        .entity(names["slider"])
        .get::<Transform>()
        .unwrap();
    let expect = joint_local_transform(
        Transform::from_translation(Vec3::X),
        Vec3::Z,
        Vec3::Y,
        JointKind::Cylindrical,
        &[0.3, 0.5],
    );
    assert!(
        close(t.translation, expect.translation),
        "cyl t {:?}",
        t.translation
    );
    assert!(
        close(t.translation, Vec3::new(1.0, 0.0, 0.5)),
        "cyl slide from origin +X 1 along +Z by 0.5 {:?}",
        t.translation
    );

    // Over-command BOTH DOFs: rotation clamps to +0.5 rad (from <limit2>), translation to +1 m (from
    // <limit>): the two clamp INDEPENDENTLY from their own bound.
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("cyl", 0, 5.0);
        p.set_dof("cyl", 1, 5.0);
    }
    app.update();
    let names = comp_entities(&mut app);
    let t = *app
        .world()
        .entity(names["slider"])
        .get::<Transform>()
        .unwrap();
    assert!(
        close(t.translation, Vec3::new(1.0, 0.0, 1.0)),
        "cyl translation must clamp to +1 m from <limit> {:?}",
        t.translation
    );
    let expect_rot = joint_local_transform(
        Transform::from_translation(Vec3::X),
        Vec3::Z,
        Vec3::Y,
        JointKind::Cylindrical,
        &[0.5, 1.0],
    );
    assert!(
        t.rotation.abs_diff_eq(expect_rot.rotation, EPS),
        "cyl rotation must clamp to +0.5 rad from <limit2> {:?}",
        t.rotation
    );
}

// base -> slider (CYLINDRICAL, origin +X 1) with ONLY a translation <limit> and NO <limit2>: the rotation
// DOF must be UNBOUNDED (the telescope-that-spins case), translation still clamped.
const CYL_NOROT_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="cylnorot" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="slider"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="cyl" type="cylindrical">
    <parent comp="base"/><child comp="slider"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1"/>
  </joint>
</hcdf>"#;

#[test]
fn cylindrical_rotation_unbounded_when_limit2_absent() {
    let mut app = build_app(CYL_NOROT_CHAIN);
    {
        let aj = app.world().resource::<ArticulatedJoints>();
        // Rotation (DOF 0) is unbounded; translation (DOF 1) clamps 0..1.
        assert_eq!((aj.0[0].lower[0], aj.0[0].upper[0]), (None, None));
        assert_eq!((aj.0[0].lower[1], aj.0[0].upper[1]), (Some(0.0), Some(1.0)));
    }
    // A big rotation command rides through unclamped; the translation clamps to +1 m.
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("cyl", 0, 5.0);
        p.set_dof("cyl", 1, 5.0);
    }
    app.update();
    let names = comp_entities(&mut app);
    let t = *app
        .world()
        .entity(names["slider"])
        .get::<Transform>()
        .unwrap();
    assert!(
        close(t.translation, Vec3::new(1.0, 0.0, 1.0)),
        "cyl translation clamps to +1 m {:?}",
        t.translation
    );
    let expect = joint_local_transform(
        Transform::from_translation(Vec3::X),
        Vec3::Z,
        Vec3::Y,
        JointKind::Cylindrical,
        &[5.0, 1.0],
    );
    assert!(
        t.rotation.abs_diff_eq(expect.rotation, EPS),
        "cyl rotation must ride through unclamped (5 rad) {:?}",
        t.rotation
    );
}

// base -> plate (PLANAR, plane normal +Z, origin at world). <limit> is the x-box (−1..1 m, DOF 0),
// <limit2> the y-box (−0.5..0.5 m, DOF 1). The in-plane basis for normal +Z (is u=+X? plane_basis
// derives it deterministically); we only assert the per-DOF CLAMP behaviour, not the basis direction.
const PLANAR_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="planar" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="plate"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="slide2d" type="planar">
    <parent comp="base"/><child comp="plate"/>
    <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1"/>
    <limit2 lower="-0.5" upper="0.5"/>
  </joint>
</hcdf>"#;

#[test]
fn planar_x_from_limit_y_from_limit2() {
    let mut app = build_app(PLANAR_CHAIN);
    {
        let aj = app.world().resource::<ArticulatedJoints>();
        assert_eq!(aj.0[0].kind, JointKind::Planar);
        assert_eq!(aj.0[0].kind.dof_count(), 2);
        // DOF 0 (x) ← <limit>, DOF 1 (y) ← <limit2>.
        assert_eq!(
            (aj.0[0].lower[0], aj.0[0].upper[0]),
            (Some(-1.0), Some(1.0)),
            "planar DOF-0 (x) bound from <limit>"
        );
        assert_eq!(
            (aj.0[0].lower[1], aj.0[0].upper[1]),
            (Some(-0.5), Some(0.5)),
            "planar DOF-1 (y) bound from <limit2>"
        );
    }
    // Over-command both in-plane axes; each clamps to its own box independently. The resulting
    // translation must match the pure kinematics fed the CLAMPED coordinates.
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("slide2d", 0, 5.0);
        p.set_dof("slide2d", 1, -5.0);
    }
    app.update();
    let names = comp_entities(&mut app);
    let t = *app
        .world()
        .entity(names["plate"])
        .get::<Transform>()
        .unwrap();
    let expect = joint_local_transform(
        Transform::default(),
        Vec3::Z,
        Vec3::Y,
        JointKind::Planar,
        &[1.0, -0.5], // x clamped to +1, y clamped to −0.5
    );
    assert!(
        close(t.translation, expect.translation),
        "planar clamped translation {:?} vs {:?}",
        t.translation,
        expect.translation
    );
}

// base -> socket (BALL, origin +X 1) with an ELLIPTIC swing cone (swing1=0.4 about Z→yaw, swing2=0.2
// about Y→pitch) and a twist bound (−0.3..0.3 about X→roll). Each slider clamps to its mapped bound.
const BALL_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="ball" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="socket"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="hip" type="ball">
    <parent comp="base"/><child comp="socket"/>
    <origin xyz="1 0 0"/>
    <swing_limit swing1="0.4" swing2="0.2"/>
    <twist_limit lower="-0.3" upper="0.3"/>
  </joint>
</hcdf>"#;

#[test]
fn ball_swing_twist_clamp_per_slider() {
    let mut app = build_app(BALL_CHAIN);
    {
        let aj = app.world().resource::<ArticulatedJoints>();
        assert_eq!(aj.0[0].kind, JointKind::Ball);
        assert_eq!(aj.0[0].kind.dof_count(), 3);
        // DOF 0 (roll/twist about X) ← <twist_limit> (−0.3..0.3).
        assert_eq!(
            (aj.0[0].lower[0], aj.0[0].upper[0]),
            (Some(-0.3), Some(0.3)),
            "ball twist (DOF 0) from <twist_limit>"
        );
        // DOF 1 (pitch about Y) ← swing2 half-angle 0.2, symmetric ±0.2.
        assert_eq!(
            (aj.0[0].lower[1], aj.0[0].upper[1]),
            (Some(-0.2), Some(0.2)),
            "ball pitch (DOF 1) symmetric ±swing2"
        );
        // DOF 2 (yaw about Z) ← swing1 half-angle 0.4, symmetric ±0.4.
        assert_eq!(
            (aj.0[0].lower[2], aj.0[0].upper[2]),
            (Some(-0.4), Some(0.4)),
            "ball yaw (DOF 2) symmetric ±swing1"
        );
    }
    // Over-command all three; each clamps to its mapped bound.
    {
        let mut p = app.world_mut().resource_mut::<JointPositions>();
        p.set_dof("hip", 0, 5.0);
        p.set_dof("hip", 1, 5.0);
        p.set_dof("hip", 2, -5.0);
    }
    app.update();
    let names = comp_entities(&mut app);
    let t = *app
        .world()
        .entity(names["socket"])
        .get::<Transform>()
        .unwrap();
    let expect = joint_local_transform(
        Transform::from_translation(Vec3::X),
        Vec3::X,
        Vec3::Y,
        JointKind::Ball,
        &[0.3, 0.2, -0.4], // twist +0.3, pitch +0.2, yaw −0.4
    );
    assert!(
        t.rotation.abs_diff_eq(expect.rotation, EPS),
        "ball clamped rotation {:?} vs {:?}",
        t.rotation,
        expect.rotation
    );
}

// A CIRCULAR ball cone: swing1 only (swing2 omitted) => both swing DOFs reuse swing1; no twist bound =>
// twist DOF unbounded.
const BALL_CIRCULAR_CHAIN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="ballc" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="socket"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="hip" type="ball">
    <parent comp="base"/><child comp="socket"/>
    <origin xyz="1 0 0"/>
    <swing_limit swing1="0.5"/>
  </joint>
</hcdf>"#;

#[test]
fn ball_circular_cone_reuses_swing1_and_twist_unbounded() {
    let app = build_app(BALL_CIRCULAR_CHAIN);
    let aj = app.world().resource::<ArticulatedJoints>();
    // Twist (DOF 0) unbounded (no <twist_limit>).
    assert_eq!((aj.0[0].lower[0], aj.0[0].upper[0]), (None, None));
    // Both swing DOFs reuse swing1=0.5 => ±0.5.
    assert_eq!(
        (aj.0[0].lower[1], aj.0[0].upper[1]),
        (Some(-0.5), Some(0.5)),
        "circular cone pitch reuses swing1"
    );
    assert_eq!(
        (aj.0[0].lower[2], aj.0[0].upper[2]),
        (Some(-0.5), Some(0.5)),
        "circular cone yaw uses swing1"
    );
}
