//! Headless integration tests for FRAME `@relative-to` resolution: the scene builder places each
//! `<frame>` at its composed comp-relative pose. A frame chained relative to a SIBLING frame lands at
//! the composed transform; a frame relative to its own comp (or an unresolvable ref) stays comp-
//! relative; a frame relative to another comp / a joint composes through the rest-pose placement. Also
//! locks the `tcp` UI-text mapping. No GPU / no window; frames are children of their comp, so their
//! LOCAL `Transform` IS the resolved pose (no `GlobalTransform` propagation needed).
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::doc::HcdfDoc;
use hcdviz::pick::Selected;
use hcdviz::scene::{frame_type_label, FrameMarker, ScenePlugin};
use hcdviz::schema::Hcdf;
use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

const EPS: f32 = 1e-5;

// One board with: A (comp-relative, 0.1 m +X then yaw 90°), B (relative-to sibling A, +0.2 m along A's
// rotated X), a self-comp ref, an unresolvable ref, and a typed tcp frame.
const FRAMES: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="frames" body-frame="FLU" world-frame="ENU">
  <comp name="board">
    <visual name="v"><geometry><box size="0.1 0.1 0.02"/></geometry></visual>
    <frame name="A"><pose xyz="0.1 0 0" rpy="0 0 1.5707963"/></frame>
    <frame name="B" relative-to="A"><pose xyz="0.2 0 0"/></frame>
    <frame name="own" relative-to="board"><pose xyz="0.05 0 0"/></frame>
    <frame name="dangling" relative-to="nope"><pose xyz="0.07 0 0"/></frame>
    <frame name="tool" type="tcp"><pose xyz="0 0 0.03"/></frame>
  </comp>
</hcdf>"#;

// Cross-comp: a frame on `base` expressed relative to the child comp `arm` (and one relative to the
// joint `j`). The joint origin lifts arm +0.5 m along +Z, so both frames land at (0,0,0.5) in base.
const FRAMES_X: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="framesx" body-frame="FLU" world-frame="ENU">
  <comp name="base">
    <frame name="at_arm" relative-to="arm"><pose xyz="0 0 0"/></frame>
    <frame name="at_joint" relative-to="j"><pose xyz="0 0 0"/></frame>
  </comp>
  <comp name="arm"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="j" type="fixed"><parent comp="base"/><child comp="arm"/><origin xyz="0 0 0.5"/></joint>
</hcdf>"#;

fn app_with_scene() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin);
    app
}

fn load(app: &mut App, xml: &str) {
    app.world_mut().resource_mut::<HcdfDoc>().0 = Some(Arc::new(Hcdf::from_xml_str(xml).unwrap()));
    app.update(); // rebuild_on_change builds the scene + frame markers
    app.update(); // flush
}

/// The LOCAL transform of the frame marker named `label` (frames are children of their comp, so the
/// local `Transform` is exactly the `@relative-to`-resolved pose the builder wrote).
fn frame_local(app: &mut App, label: &str) -> Transform {
    let world = app.world_mut();
    let mut q = world.query::<(&FrameMarker, &Transform)>();
    q.iter(world)
        .find(|(m, _)| m.label == label)
        .map(|(_, t)| *t)
        .unwrap_or_else(|| panic!("frame {label:?} not spawned"))
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < EPS
}

#[test]
fn sibling_relative_frame_lands_at_the_composed_pose() {
    let mut app = app_with_scene();
    load(&mut app, FRAMES);

    // A: raw comp-relative pose, 0.1 m +X, yaw 90° (X→Y).
    let a = frame_local(&mut app, "A");
    assert!(
        close(a.translation, Vec3::new(0.1, 0.0, 0.0)),
        "A t {:?}",
        a.translation
    );
    assert!(
        a.rotation
            .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, FRAC_PI_2), EPS),
        "A r {:?}",
        a.rotation
    );

    // B relative-to A: T_B = T_A ∘ pose_B. A's yaw sends B's +0.2 X onto +Y, added to A's 0.1 +X ⇒
    // (0.1, 0.2, 0), and B inherits A's rotation. This is the composed-pose assertion.
    let b = frame_local(&mut app, "B");
    assert!(
        close(b.translation, Vec3::new(0.1, 0.2, 0.0)),
        "B must land at the composed pose (0.1, 0.2, 0), got {:?}",
        b.translation
    );
    assert!(
        b.rotation
            .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, FRAC_PI_2), EPS),
        "B inherits A's rotation, got {:?}",
        b.rotation
    );
}

#[test]
fn own_comp_and_unresolvable_refs_stay_comp_relative() {
    let mut app = app_with_scene();
    load(&mut app, FRAMES);

    // relative-to the OWN comp == comp-relative (the raw pose): the real test-minimal case.
    let own = frame_local(&mut app, "own");
    assert!(
        close(own.translation, Vec3::new(0.05, 0.0, 0.0)),
        "own {:?}",
        own.translation
    );
    assert!(own.rotation.abs_diff_eq(Quat::IDENTITY, EPS));

    // An unresolvable ref keeps the earlier behaviour (comp-relative pose): the viewer stays lenient.
    let dangling = frame_local(&mut app, "dangling");
    assert!(
        close(dangling.translation, Vec3::new(0.07, 0.0, 0.0)),
        "dangling {:?}",
        dangling.translation
    );
}

#[test]
fn cross_comp_and_joint_refs_compose_through_placement() {
    let mut app = app_with_scene();
    load(&mut app, FRAMES_X);
    // Frame on `base` relative to child comp `arm`: arm sits +0.5 m along +Z of base (the joint
    // origin), so the frame lands there in base's own frame.
    let at_arm = frame_local(&mut app, "at_arm");
    assert!(
        close(at_arm.translation, Vec3::new(0.0, 0.0, 0.5)),
        "at_arm must compose through arm's placement, got {:?}",
        at_arm.translation
    );
    // Relative to the JOINT `j` resolves to the same child-link frame.
    let at_joint = frame_local(&mut app, "at_joint");
    assert!(
        close(at_joint.translation, Vec3::new(0.0, 0.0, 0.5)),
        "at_joint (joint ref = child link frame) {:?}",
        at_joint.translation
    );
}

#[test]
fn tcp_frame_type_is_spelled_out() {
    // Pure mapping (the UI text rule).
    assert_eq!(
        frame_type_label(Some("tcp")).as_deref(),
        Some("TCP (tool center point)")
    );
    assert_eq!(
        frame_type_label(Some("optical")).as_deref(),
        Some("optical")
    );
    assert_eq!(frame_type_label(None), None);

    // And the spawned tcp frame marker carries the spelled-out label the frames display renders.
    let mut app = app_with_scene();
    load(&mut app, FRAMES);
    let world = app.world_mut();
    let mut q = world.query::<&FrameMarker>();
    let tool = q
        .iter(world)
        .find(|m| m.label == "tool")
        .expect("tool frame spawned");
    assert_eq!(tool.type_label.as_deref(), Some("TCP (tool center point)"));
}
