//! Headless integration test for imported-sensor rendering (no GPU, no window).
//!
//! Builds the REAL scene from an HCDF that mirrors what the importers now populate on `b3rb`: optical
//! camera/lidar sensors that carry `<fov>` frustum geometry, and FLUID sensors (barometer/airspeed)
//! that carry none. Asserts that every sensor becomes a `SensorMarker` entity, that the optical FoVs
//! spawn frustum MESH markers (the drawable frustum), and that the fluid sensors spawn triad-only
//! markers (no FoV mesh): the "labeled glyph like em/rf/force" treatment. All meshes here are
//! primitives/frustums, so the build is synchronous (no async glTF).
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::display::DisplayRegistry;
use hcdviz::doc::HcdfDoc;
use hcdviz::pick::{
    IsolateSelection, IsolateSet, Selected, SelectionOverrides, SensorVizOverrides, SensorVizState,
};
use hcdviz::scene::{sync_sensor_visibility, ScenePlugin, SensorMarker, ID_SENSORS};
use hcdviz::schema::Hcdf;
use std::collections::HashMap;
use std::sync::Arc;

// camera_link + lidar_link carry optical sensors with <fov> frustums; baro_link + pitot_link carry
// FLUID sensors (no FoV). Fixed joints tie them into one tree so the scene builds a single skeleton.
const SENSORS: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="sensors" body-frame="FLU" world-frame="ENU">
  <comp name="camera_link">
    <sensor name="cam">
      <optical type="camera">
        <fov name="view" color="0 1 0 0.3">
          <geometry>
            <frustum shape="pyramidal"><near>0.05</near><far>10.0</far><hfov>1.2</hfov><vfov>0.9</vfov></frustum>
          </geometry>
        </fov>
      </optical>
    </sensor>
  </comp>
  <comp name="lidar_link">
    <sensor name="lidar">
      <optical type="lidar">
        <fov name="scan">
          <geometry>
            <frustum shape="conical"><near>0.1</near><far>12.0</far><fov>0.5</fov></frustum>
          </geometry>
        </fov>
        <lidar-params>
          <range><max unit="m">12.0</max></range>
          <scan-pattern>
            <horizontal samples="360" min-angle="-3.14159" max-angle="3.14159"/>
            <vertical samples="16" min-angle="-0.26" max-angle="0.26"/>
          </scan-pattern>
        </lidar-params>
      </optical>
    </sensor>
  </comp>
  <comp name="baro_link">
    <sensor name="baro"><fluid type="barometer"><driver name="bmp390"/></fluid></sensor>
  </comp>
  <comp name="pitot_link">
    <sensor name="pitot"><fluid type="airspeed"><driver name="ms4525do"/></fluid></sensor>
  </comp>
  <joint name="j0" type="fixed"><parent comp="camera_link"/><child comp="lidar_link"/></joint>
  <joint name="j1" type="fixed"><parent comp="camera_link"/><child comp="baro_link"/></joint>
  <joint name="j2" type="fixed"><parent comp="camera_link"/><child comp="pitot_link"/></joint>
</hcdf>"#;

/// Every `SensorMarker` entity keyed by its spawned `Name`, with a flag for whether it carries a mesh
/// (mesh ⇒ a drawable FoV frustum; no mesh ⇒ a triad-only pose marker).
fn sensor_markers(app: &mut App) -> HashMap<String, bool> {
    let world = app.world_mut();
    let mut q = world.query_filtered::<(&Name, Option<&Mesh3d>), With<SensorMarker>>();
    q.iter(world)
        .map(|(n, m)| (n.as_str().to_string(), m.is_some()))
        .collect()
}

fn build_scene() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<IsolateSelection>()
        .init_resource::<SelectionOverrides>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin);
    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(SENSORS).unwrap()));
    app.update(); // rebuild_on_change builds the scene
    app.update(); // flush the command queue so children materialize
    app
}

#[test]
fn optical_fov_frustums_and_fluid_markers_spawn() {
    let mut app = build_scene();
    let markers = sensor_markers(&mut app);

    // Every sensor gets a pose marker (triad + label), optical and fluid alike.
    for node in ["sensor:cam", "sensor:lidar", "sensor:baro", "sensor:pitot"] {
        assert!(
            markers.contains_key(node),
            "missing pose marker {node}: {markers:?}"
        );
        assert!(
            !markers[node],
            "pose marker {node} must be triad-only (no mesh)"
        );
    }

    // The optical <fov> frustums become drawable mesh markers (pyramidal + conical both resolve).
    for fov in ["fov:cam", "fov:lidar"] {
        assert!(
            markers.contains_key(fov),
            "missing FoV frustum mesh {fov}: {markers:?}"
        );
        assert!(markers[fov], "FoV marker {fov} must carry a frustum mesh");
    }

    // The lidar's <lidar-params><scan-pattern> becomes two drawable scan-extent mesh markers: the
    // translucent filled annulus/sector/band (scan:) plus its thin boundary ring (scan-ring:).
    for scan in ["scan:lidar", "scan-ring:lidar"] {
        assert!(
            markers.contains_key(scan),
            "missing lidar scan mesh {scan}: {markers:?}"
        );
        assert!(markers[scan], "scan marker {scan} must carry a mesh");
    }

    // FLUID sensors have no field of view, so no `fov:` mesh marker is spawned for them.
    assert!(
        !markers.contains_key("fov:baro"),
        "barometer must not spawn a FoV mesh: {markers:?}"
    );
    assert!(
        !markers.contains_key("fov:pitot"),
        "airspeed must not spawn a FoV mesh: {markers:?}"
    );

    // Exactly the eight markers above: 4 triad-only pose nodes + 2 frustum meshes + 2 lidar scan meshes
    // (fill + boundary ring); regression guard on any stray or missing sensor entity.
    assert_eq!(
        markers.len(),
        8,
        "expected 8 sensor markers, got {markers:?}"
    );
    assert_eq!(
        markers.values().filter(|has_mesh| **has_mesh).count(),
        4,
        "exactly 4 drawable meshes (2 FoV frustums + lidar fill + lidar ring)"
    );
}

/// Show iff the named entity's `Visibility` is not `Hidden`: the sync systems write `Inherited` when
/// shown, `Hidden` when suppressed. `None` when no entity carries that `Name`.
fn shown_by_name(app: &mut App, name: &str) -> Option<bool> {
    let world = app.world_mut();
    let mut q = world.query::<(&Name, &Visibility)>();
    q.iter(world)
        .find(|(n, _)| n.as_str() == name)
        .map(|(_, v)| *v != Visibility::Hidden)
}

/// Drives the REAL `sync_sensor_visibility` system headless: with the global Sensors display ON, a
/// per-sensor `SensorVizOverrides` entry must hide EXACTLY the targeted sensor's FoV/scan mesh entities
/// (they all carry the sensor NAME as their `SensorMarker` label) and leave every other sensor visible.
#[test]
fn per_sensor_override_hides_exactly_the_targeted_sensor() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<IsolateSelection>()
        .init_resource::<IsolateSet>()
        .init_resource::<SelectionOverrides>()
        .init_resource::<SensorVizOverrides>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin)
        // Drive the mesh-visibility sync directly (the immediate-mode triad draw needs a Gizmo store
        // MinimalPlugins lacks); this is the exact system SensorsDisplay wires in the real app.
        .add_systems(Update, sync_sensor_visibility);
    // Force the global Sensors toggle ON so shown-ness reflects only the per-sensor override.
    app.world_mut()
        .resource_mut::<DisplayRegistry>()
        .set_enabled(ID_SENSORS, true);
    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(SENSORS).unwrap()));
    app.update(); // build the scene
    app.update(); // flush command queue + run sync_sensor_visibility

    // No override yet: every FoV/scan mesh is visible under the global toggle.
    assert_eq!(shown_by_name(&mut app, "fov:cam"), Some(true));
    assert_eq!(shown_by_name(&mut app, "fov:lidar"), Some(true));
    assert_eq!(shown_by_name(&mut app, "scan:lidar"), Some(true));

    // Hide EXACTLY the camera sensor's viz.
    app.world_mut()
        .resource_mut::<SensorVizOverrides>()
        .0
        .insert(
            ("camera_link".to_string(), "cam".to_string()),
            SensorVizState {
                visible: false,
                ..Default::default()
            },
        );
    app.update();

    // The camera's FoV frustum is now hidden; the lidar's FoV + scan are untouched.
    assert_eq!(
        shown_by_name(&mut app, "fov:cam"),
        Some(false),
        "the targeted sensor's FoV must hide"
    );
    assert_eq!(
        shown_by_name(&mut app, "fov:lidar"),
        Some(true),
        "a different sensor's FoV must stay visible"
    );
    assert_eq!(
        shown_by_name(&mut app, "scan:lidar"),
        Some(true),
        "a different sensor's scan must stay visible"
    );

    // Flip the override back on: the camera FoV returns.
    app.world_mut()
        .resource_mut::<SensorVizOverrides>()
        .0
        .insert(
            ("camera_link".to_string(), "cam".to_string()),
            SensorVizState {
                visible: true,
                ..Default::default()
            },
        );
    app.update();
    assert_eq!(
        shown_by_name(&mut app, "fov:cam"),
        Some(true),
        "clearing the override restores the FoV"
    );
}

/// Max radial distance (XY) of the mesh carried by the entity named `name`.
fn mesh_xy_radius(app: &mut App, name: &str) -> f32 {
    use bevy::mesh::VertexAttributeValues;
    let world = app.world_mut();
    let mut q = world.query::<(&Name, &Mesh3d)>();
    let handle = q
        .iter(world)
        .find(|(n, _)| n.as_str() == name)
        .map(|(_, m)| m.0.clone())
        .unwrap_or_else(|| panic!("no mesh entity named {name}"));
    let meshes = world.resource::<Assets<Mesh>>();
    let mesh = meshes.get(&handle).expect("mesh asset present");
    let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
    else {
        panic!("mesh {name} has no positions");
    };
    pos.iter()
        .map(|p| (p[0] * p[0] + p[1] * p[1]).sqrt())
        .fold(0.0f32, f32::max)
}

/// End-to-end: the 12 m lidar draws CAPPED (≈2.5 m) by default, and flipping the global
/// [`SensorVizGlobal::full_extent`] toggle re-resolves the scan mesh to its TRUE 12 m radius on the next
/// rebuild (the mesh cache keys on the effective extent). Exercises the full plumbing:
/// `SensorVizGlobal` → `rebuild_on_change` → `resolve_lidar_scan_mesh`.
#[test]
fn lidar_display_cap_toggles_with_global_full_extent() {
    use hcdviz::geometry::LIDAR_DISPLAY_CAP_M;
    use hcdviz::pick::SensorVizGlobal;

    let mut app = build_scene();

    let capped = mesh_xy_radius(&mut app, "scan:lidar");
    assert!(
        (capped - LIDAR_DISPLAY_CAP_M).abs() < 1e-1,
        "the 12 m lidar draws capped at {LIDAR_DISPLAY_CAP_M} m, got {capped}"
    );

    // Flip the global full-extent toggle and force a rebuild (dendrite drives this via a republish; here
    // we mark the doc changed directly). The scan mesh re-resolves to the true range.
    app.world_mut()
        .resource_mut::<SensorVizGlobal>()
        .full_extent = true;
    app.world_mut().resource_mut::<HcdfDoc>().set_changed();
    app.update();
    app.update();

    let full = mesh_xy_radius(&mut app, "scan:lidar");
    assert!(
        (full - 12.0).abs() < 5e-1,
        "full-extent draws the true 12 m range, got {full}"
    );
}
