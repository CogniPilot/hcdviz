//! Headless integration test for collision MESH rendering (no GPU, no window).
//!
//! Before the fix a collision `<mesh uri>` always loaded a `Handle<Mesh>`, whose ONLY loader is STL, so
//! a `.glb`/`.gltf` collision URI silently resolved to nothing and unsupported extensions drew nothing at
//! all. The fix dispatches by extension: glTF/GLB via the SCENE loader (a `WorldAssetRoot` under the
//! `CollisionItem` kind, exactly like a visual model), `.stl` via the Mesh loader, and anything else via a
//! LOUD fallback bounds box. This test builds the REAL scene from an HCDF carrying one collision of each
//! flavor and asserts each spawns the right `CollisionItem` shape.
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use bevy::world_serialization::{WorldAsset, WorldAssetRoot};
use hcdviz::display::DisplayRegistry;
use hcdviz::doc::HcdfDoc;
use hcdviz::pick::{IsolateSelection, Selected, SelectionOverrides};
use hcdviz::scene::{
    collision_mesh_kind, CollisionItem, CollisionMeshKind, OwnerComp, ScenePlugin,
};
use hcdviz::schema::Hcdf;
use std::collections::HashMap;
use std::sync::Arc;

// One comp, three collision meshes: a GLB (scene loader), an STL (Mesh loader), and a `.obj` (no loader
// → fallback box). URIs point at assets that do not exist on disk; the asset_server still hands back a
// handle and the entity spawns synchronously, which is all this structural test observes.
const COLLISIONS: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="collisions" body-frame="FLU" world-frame="ENU">
  <comp name="chassis">
    <collision name="glb_col">
      <geometry><mesh uri="assets/wheel.glb"/></geometry>
    </collision>
    <collision name="stl_col">
      <geometry><mesh uri="assets/hull.stl"/></geometry>
    </collision>
    <collision name="bad_col">
      <geometry><mesh uri="assets/legacy.obj"/></geometry>
    </collision>
  </comp>
</hcdf>"#;

/// Every `CollisionItem` entity keyed by its `Name`, with flags for the two spawn shapes it could carry:
/// a `WorldAssetRoot` (glTF scene subtree root) and/or a `Mesh3d` (STL / primitive / fallback box).
struct ColItem {
    has_scene_root: bool,
    has_mesh: bool,
    has_owner: bool,
}

fn collision_items(app: &mut App) -> HashMap<String, ColItem> {
    let world = app.world_mut();
    let mut q = world.query_filtered::<(
        &Name,
        Has<WorldAssetRoot>,
        Has<Mesh3d>,
        Has<OwnerComp>,
    ), With<CollisionItem>>();
    q.iter(world)
        .map(|(n, scene, mesh, owner)| {
            (
                n.as_str().to_string(),
                ColItem {
                    has_scene_root: scene,
                    has_mesh: mesh,
                    has_owner: owner,
                },
            )
        })
        .collect()
}

fn build_scene() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        // The GLB collision spawns a `WorldAssetRoot` (Handle<WorldAsset>); its asset type must be
        // registered for the load to allocate a handle (the real app wires this via world_serialization).
        .init_asset::<WorldAsset>()
        .init_resource::<Selected>()
        .init_resource::<IsolateSelection>()
        .init_resource::<SelectionOverrides>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin);
    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(COLLISIONS).unwrap()));
    app.update(); // rebuild_on_change builds the scene
    app.update(); // flush the command queue so children materialize
    app
}

#[test]
fn collision_kind_dispatch_by_extension() {
    // Case-insensitive, basename-only (a directory in the path never fools the extension check).
    assert_eq!(
        collision_mesh_kind("a/b/wheel.glb"),
        CollisionMeshKind::Scene
    );
    assert_eq!(collision_mesh_kind("WHEEL.GLTF"), CollisionMeshKind::Scene);
    assert_eq!(
        collision_mesh_kind("assets/hull.STL"),
        CollisionMeshKind::Stl
    );
    assert_eq!(
        collision_mesh_kind("legacy.obj"),
        CollisionMeshKind::Unsupported
    );
    assert_eq!(collision_mesh_kind("noext"), CollisionMeshKind::Unsupported);
    // A `.gltf` directory name must not win over the real (extension-less) basename.
    assert_eq!(
        collision_mesh_kind("foo.gltf/thing"),
        CollisionMeshKind::Unsupported
    );
}

#[test]
fn each_collision_mesh_spawns_under_the_collision_kind() {
    let mut app = build_scene();
    let items = collision_items(&mut app);

    // All three collisions became `CollisionItem` entities owned by their comp, none silently dropped.
    for name in [
        "collision:glb_col",
        "collision:stl_col",
        "collision:bad_col",
    ] {
        let it = items
            .get(name)
            .unwrap_or_else(|| panic!("missing collision item {name}: have {:?}", items.keys()));
        assert!(
            it.has_owner,
            "{name} must carry OwnerComp for isolate/toggle"
        );
    }

    // GLB collision → the SCENE loader: a WorldAssetRoot subtree root, NOT a single Handle<Mesh>.
    let glb = &items["collision:glb_col"];
    assert!(
        glb.has_scene_root,
        "GLB collision must spawn a WorldAssetRoot scene root"
    );
    assert!(
        !glb.has_mesh,
        "GLB collision root is a scene root, not a Mesh3d"
    );

    // STL collision → the Mesh loader path (translucent-tinted single mesh).
    let stl = &items["collision:stl_col"];
    assert!(stl.has_mesh, "STL collision must spawn a Mesh3d");
    assert!(!stl.has_scene_root, "STL collision is not a scene subtree");

    // Unsupported `.obj` → the LOUD fallback: a translucent bounds box (a Mesh3d), never nothing.
    let bad = &items["collision:bad_col"];
    assert!(
        bad.has_mesh,
        "unsupported-ext collision must spawn a fallback box Mesh3d"
    );
    assert!(
        !bad.has_scene_root,
        "the fallback box is a primitive mesh, not a scene"
    );

    // Exactly the three collision items: regression guard on a stray or missing spawn.
    assert_eq!(
        items.len(),
        3,
        "expected 3 collision items, got {:?}",
        items.keys()
    );
}
