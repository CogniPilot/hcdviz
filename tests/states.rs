//! Headless integration tests for KINEMATIC STATES: a `<state default="true">` is applied to
//! [`JointPositions`] on load (the viewer no longer always zero-poses), a reload RE-applies it (never
//! zero), and [`apply_state`] seeds the command map the states dropdown drives. No GPU / no window.
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::doc::HcdfDoc;
use hcdviz::joints::{apply_state, default_state, JointPositions, JointsPlugin};
use hcdviz::pick::Selected;
use hcdviz::scene::{CompEntity, ScenePlugin};
use hcdviz::schema::Hcdf;
use std::sync::Arc;

const EPS: f32 = 1e-4;

// base -> arm (revolute "shoulder" about +Z, origin +X 1). A DEFAULT state "home" poses shoulder to
// 0.5 rad; a second named state "raised" poses it to 1.0 rad (the dropdown-apply path).
const STATED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="stated" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="arm"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="shoulder" type="revolute">
    <parent comp="base"/><child comp="arm"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2"/>
  </joint>
  <state name="home" default="true">
    <joint-position joint="shoulder" value="0.5"/>
  </state>
  <state name="raised">
    <joint-position joint="shoulder" value="1.0"/>
  </state>
</hcdf>"#;

// Same tree, no <state>: load must land in the zero pose (the unchanged behaviour).
const NO_STATE: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="nostate" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="arm"><visual name="v"><geometry><box size="0.1 0.1 0.1"/></geometry></visual></comp>
  <joint name="shoulder" type="revolute">
    <parent comp="base"/><child comp="arm"/>
    <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
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
    app.update(); // rebuild_on_change + reset_on_reload (applies the default state) + articulate
    app.update(); // flush
    app
}

fn arm_entity(app: &mut App) -> Entity {
    let world = app.world_mut();
    let mut q = world.query::<(Entity, &CompEntity)>();
    q.iter(world)
        .find(|(_, c)| c.name == "arm")
        .map(|(e, _)| e)
        .expect("arm comp spawned")
}

#[test]
fn default_state_applied_on_load() {
    let mut app = build_app(STATED);
    // The default state "home" seeds JointPositions with shoulder=0.5 (NOT the zero pose).
    assert!(
        (app.world().resource::<JointPositions>().dof("shoulder", 0) - 0.5).abs() < EPS,
        "default state must seed shoulder=0.5 on load, got {}",
        app.world().resource::<JointPositions>().dof("shoulder", 0)
    );
    // And the arm is actually posed there: revolute +Z by 0.5 rad.
    let arm = arm_entity(&mut app);
    let t = *app.world().entity(arm).get::<Transform>().unwrap();
    assert!(
        t.rotation
            .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, 0.5), EPS),
        "arm must render at the default-state pose: {:?}",
        t.rotation
    );
}

#[test]
fn no_default_state_loads_zero_pose() {
    let app = build_app(NO_STATE);
    assert!(
        app.world().resource::<JointPositions>().0.is_empty(),
        "a doc with no default state loads the zero pose (empty command map)"
    );
}

#[test]
fn reload_reapplies_default_state_not_zero() {
    let mut app = build_app(STATED);
    // Move the shoulder away from the default via the one source of truth (as a slider would).
    app.world_mut()
        .resource_mut::<JointPositions>()
        .set_dof("shoulder", 0, -1.5);
    app.update();
    assert!(
        (app.world().resource::<JointPositions>().dof("shoulder", 0) + 1.5).abs() < EPS,
        "precondition: moved off the default"
    );

    // Reload the SAME doc → reset_on_reload must RE-APPLY the default state (0.5), never zero it.
    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(STATED).unwrap()));
    app.update();
    assert!(
        (app.world().resource::<JointPositions>().dof("shoulder", 0) - 0.5).abs() < EPS,
        "reload must re-apply the default state (0.5), not the zero pose, got {}",
        app.world().resource::<JointPositions>().dof("shoulder", 0)
    );
    let arm = arm_entity(&mut app);
    let t = *app.world().entity(arm).get::<Transform>().unwrap();
    assert!(
        t.rotation
            .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, 0.5), EPS),
        "arm must re-pose at the default state after reload: {:?}",
        t.rotation
    );
}

#[test]
fn apply_state_seeds_command_map_and_replaces_prior_commands() {
    // The dropdown-apply path: `apply_state` writes a named state's positions, REPLACING whatever was
    // commanded (a named state is a whole pose).
    let h = Hcdf::from_xml_str(STATED).unwrap();
    let raised = h
        .state
        .iter()
        .find(|s| s.name.as_deref() == Some("raised"))
        .expect("raised state present");
    let mut positions = JointPositions::default();
    positions.set_dof("shoulder", 0, -0.3); // a prior command the whole-pose state replaces
    positions.set_dof("ghost", 0, 9.0); // an unrelated command the whole-pose state clears
    apply_state(&mut positions, raised);
    assert!(
        (positions.dof("shoulder", 0) - 1.0).abs() < EPS,
        "raised state must set shoulder=1.0"
    );
    assert!(
        !positions.0.contains_key("ghost"),
        "a named state is a whole pose: prior unrelated commands are cleared"
    );
}

#[test]
fn default_state_helper_picks_the_marked_state() {
    let h = Hcdf::from_xml_str(STATED).unwrap();
    let d = default_state(&h).expect("home is the default state");
    assert_eq!(d.name.as_deref(), Some("home"));
    let h2 = Hcdf::from_xml_str(NO_STATE).unwrap();
    assert!(
        default_state(&h2).is_none(),
        "a doc with no default state resolves to None"
    );
}
