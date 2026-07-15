//! Headless integration test for picking RESOLUTION (no GPU, no window).
//!
//! Builds the REAL scene from a nested HCDF using primitive visuals (synchronous, no async glTF),
//! then asserts that each comp's visual mesh resolves to its OWN comp via the same `owning_comp` walk
//! `on_click` uses. This is the rigorous, GUI-free check that sub-components are individually
//! selectable and don't collapse to the root comp.
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::doc::HcdfDoc;
use hcdviz::pick::{owning_comp, Selected};
use hcdviz::scene::{CompEntity, ScenePlugin};
use hcdviz::schema::Hcdf;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

// world (no visual, the root) -> body (box) -> arm (box). Mirrors openarm's `world -> body -> arms`.
const NESTED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="nested" body-frame="FLU" world-frame="ENU">
  <comp name="world"/>
  <comp name="body">
    <visual name="body_vis"><geometry><box size="0.10 0.10 0.50"/></geometry></visual>
  </comp>
  <comp name="arm">
    <visual name="arm_vis"><geometry><box size="0.30 0.05 0.05"/></geometry></visual>
  </comp>
  <joint name="j1" type="fixed"><parent comp="world"/><child comp="body"/></joint>
  <joint name="j2" type="revolute"><parent comp="body"/><child comp="arm"/><origin xyz="0 0 0.5"/></joint>
</hcdf>"#;

#[test]
fn each_visual_resolves_to_its_own_comp() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin);

    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(NESTED).unwrap()));
    app.update(); // rebuild_on_change builds the scene
    app.update(); // flush

    let world = app.world_mut();

    let mut cq = world.query::<(Entity, &CompEntity)>();
    let comp_name: HashMap<Entity, String> =
        cq.iter(world).map(|(e, c)| (e, c.name.clone())).collect();
    let comp_set: HashSet<Entity> = comp_name.keys().copied().collect();

    let mut pq = world.query::<(Entity, &ChildOf)>();
    let parent_of: HashMap<Entity, Entity> = pq.iter(world).map(|(e, c)| (e, c.parent())).collect();

    let mut mq = world.query_filtered::<Entity, With<Mesh3d>>();
    let meshes: Vec<Entity> = mq.iter(world).collect();

    assert!(
        meshes.len() >= 2,
        "expected the body + arm box visuals, found {} mesh entities",
        meshes.len()
    );

    let resolved: Vec<String> = meshes
        .iter()
        .map(|&m| {
            let comp = owning_comp(m, |e| parent_of.get(&e).copied(), |e| comp_set.contains(&e))
                .expect("a visual mesh must resolve to some comp");
            comp_name[&comp].clone()
        })
        .collect();

    // Each visual must select its OWN comp; nothing should collapse to the root "world".
    assert!(
        resolved.contains(&"body".to_string()),
        "body visual didn't resolve to body: {resolved:?}"
    );
    assert!(
        resolved.contains(&"arm".to_string()),
        "arm visual didn't resolve to arm: {resolved:?}"
    );
    assert!(
        !resolved.iter().any(|n| n == "world"),
        "a visual wrongly resolved to the root `world` (the reported bug): {resolved:?}"
    );
}

// openarm uses glTF `<model>` visuals (ARM A), which spawn a WorldAssetRoot HOLDER under the comp;
// the real glTF mesh then loads async as a child of that holder. This checks the SYNCHRONOUS part we
// own: that each comp's visual holder is parented to its OWN comp, using `VisualItem` (the marker on
// both glTF holders and primitive visuals). No GPU/real asset needed: `asset_server.load` just returns
// a handle.
const NESTED_GLTF: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="nested_gltf" body-frame="FLU" world-frame="ENU">
  <comp name="world"/>
  <comp name="body"><visual name="body_vis"><model uri="body.glb"/></visual></comp>
  <comp name="left"><visual name="left_vis"><model uri="left.glb"/></visual></comp>
  <comp name="right"><visual name="right_vis"><model uri="right.glb"/></visual></comp>
  <joint name="j0" type="fixed"><parent comp="world"/><child comp="body"/></joint>
  <joint name="j1" type="fixed"><parent comp="body"/><child comp="left"/></joint>
  <joint name="j2" type="fixed"><parent comp="body"/><child comp="right"/></joint>
</hcdf>"#;

#[test]
fn gltf_visual_holders_attach_to_their_own_comp() {
    use hcdviz::scene::VisualItem;
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_asset::<bevy::world_serialization::WorldAsset>() // glTF <model> loads as a WorldAsset in 0.19
        .init_resource::<Selected>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin);
    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(NESTED_GLTF).unwrap()));
    app.update();
    app.update();

    let world = app.world_mut();
    let mut cq = world.query::<(Entity, &CompEntity)>();
    let comp_name: HashMap<Entity, String> =
        cq.iter(world).map(|(e, c)| (e, c.name.clone())).collect();
    let comp_set: HashSet<Entity> = comp_name.keys().copied().collect();
    let mut pq = world.query::<(Entity, &ChildOf)>();
    let parent_of: HashMap<Entity, Entity> = pq.iter(world).map(|(e, c)| (e, c.parent())).collect();
    let mut hq = world.query_filtered::<Entity, With<VisualItem>>();
    let holders: Vec<Entity> = hq.iter(world).collect();

    assert_eq!(
        holders.len(),
        3,
        "expected 3 glTF visual holders (body/left/right)"
    );
    let resolved: Vec<String> = holders
        .iter()
        .map(|&h| {
            let c = owning_comp(h, |e| parent_of.get(&e).copied(), |e| comp_set.contains(&e))
                .expect("a visual holder must resolve to a comp");
            comp_name[&c].clone()
        })
        .collect();
    let mut sorted = resolved.clone();
    sorted.sort();
    assert_eq!(
        sorted,
        vec!["body", "left", "right"],
        "each glTF visual holder must resolve to its OWN comp, not collapse to root: {resolved:?}"
    );
}

/// Smoke test: a real `Pointer<Click>` on a deep leaf fires `on_click` and selects the hit's owning
/// comp (not `None`/the root). This exercises the live observer wiring + button gate + original-target
/// resolution end-to-end.
///
/// NOTE: it does NOT reproduce event *bubbling*; Bevy's pick-event propagation only runs inside the
/// full picking runtime (DefaultPlugins), not under MinimalPlugins, so the observer fires once here.
/// The bubbling bug's core (resolving the ORIGINAL hit → nearest comp, never an ancestor/root) is
/// locked by `owning_comp_stops_at_nearest_comp_through_gltf_wrapper`, and the live fix was verified at
/// runtime via the `HCDVIZ_DEBUG_PICK` probe.
#[test]
fn real_click_selects_the_hit_comp() {
    use bevy::camera::{ManualTextureViewHandle, NormalizedRenderTarget};
    use bevy::picking::backend::HitData;
    use bevy::picking::events::{Click, Pointer};
    use bevy::picking::pointer::{Location, PointerButton, PointerId};
    use std::time::Duration;

    #[derive(Resource)]
    struct PendingClick(Option<Pointer<Click>>);

    let mut app = App::new();
    app.add_plugins(MinimalPlugins).init_resource::<Selected>();
    app.add_observer(hcdviz::pick::on_click);

    // world (root comp) -> body (comp) -> leaf (the picked glTF-mesh-like entity, no comp)
    let (comp_body, leaf) = {
        let world = app.world_mut();
        let comp_world = world
            .spawn(CompEntity {
                comp_index: 0,
                name: "world".into(),
            })
            .id();
        let comp_body = world
            .spawn(CompEntity {
                comp_index: 1,
                name: "body".into(),
            })
            .id();
        let leaf = world.spawn_empty().id();
        world.entity_mut(comp_world).add_child(comp_body);
        world.entity_mut(comp_body).add_child(leaf);
        (comp_body, leaf)
    };

    // A real propagating left-click targeted at the leaf (location/hit are dummies the observer ignores).
    let location = Location {
        target: NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0)),
        position: Vec2::ZERO,
    };
    let click = Click {
        button: PointerButton::Primary,
        hit: HitData::new(Entity::PLACEHOLDER, 0.0, None, None),
        duration: Duration::ZERO,
        count: 1,
    };
    app.insert_resource(PendingClick(Some(Pointer::new(
        PointerId::Mouse,
        location,
        click,
        leaf,
    ))));
    // Trigger from INSIDE a system (as bevy_picking does) and run the schedule, so the event
    // auto-propagates up the ChildOf chain and the observer fires for every ancestor.
    app.add_systems(
        Update,
        |mut commands: Commands, mut pending: ResMut<PendingClick>| {
            if let Some(ev) = pending.0.take() {
                commands.trigger(ev);
            }
        },
    );
    app.update();

    assert_eq!(
        app.world().resource::<Selected>().0,
        Some(comp_body),
        "a click bubbling up from the leaf must select the clicked link (body), not the root (world)"
    );
}

/// Isolate-selection integration test: build the REAL scene from the nested HCDF (world→body→arm, both
/// with primitive box visuals, synchronous, no async glTF), turn on `IsolateSelection` with `body`
/// selected, run the live `VisualDisplay` visibility system, and assert ONLY the body's `VisualItem`
/// shows (Inherited) while the arm's is Hidden. Then turn isolate off and assert both show again.
#[test]
fn isolate_selection_shows_only_selected_comps_visual() {
    use bevy::asset::AssetPlugin;
    use hcdviz::display::{AddDisplayExt, DisplayRegistry};
    use hcdviz::pick::{IsolateSelection, IsolateSet, SelectionOverrides};
    use hcdviz::scene::{OwnerComp, VisualDisplay, VisualItem};

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<IsolateSelection>()
        .init_resource::<IsolateSet>()
        .init_resource::<SelectionOverrides>() // sync_visual_visibility now reads per-link overrides
        .init_resource::<DisplayRegistry>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin)
        .add_display(VisualDisplay); // registers "Visual" (default-on) + adds sync_visual_visibility

    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(NESTED).unwrap()));
    app.update(); // rebuild_on_change builds the scene (also clears Selected)
    app.update(); // flush command queue (children/visuals materialize)

    // Map each comp name → entity so we can select "body".
    let (body, arm) = {
        let world = app.world_mut();
        let mut cq = world.query::<(Entity, &CompEntity)>();
        let by_name: HashMap<String, Entity> =
            cq.iter(world).map(|(e, c)| (c.name.clone(), e)).collect();
        (by_name["body"], by_name["arm"])
    };

    // Returns the (single) VisualItem visibility owned by `comp`.
    fn visual_vis_of(app: &mut App, comp: Entity) -> Visibility {
        let world = app.world_mut();
        let mut q = world.query_filtered::<(&OwnerComp, &Visibility), With<VisualItem>>();
        let found: Vec<Visibility> = q
            .iter(world)
            .filter(|(o, _)| o.0 == comp)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(
            found.len(),
            1,
            "expected exactly one VisualItem for the comp, got {}",
            found.len()
        );
        found[0]
    }

    // Isolate ON with body selected → only body's visual is shown.
    app.world_mut().resource_mut::<Selected>().0 = Some(body);
    app.world_mut().resource_mut::<IsolateSelection>().0 = true;
    app.update();
    assert_eq!(
        visual_vis_of(&mut app, body),
        Visibility::Inherited,
        "body visual must stay visible"
    );
    assert_eq!(
        visual_vis_of(&mut app, arm),
        Visibility::Hidden,
        "arm visual must be isolated away"
    );

    // Isolate OFF → both visuals shown again (global Visual toggle is on).
    app.world_mut().resource_mut::<IsolateSelection>().0 = false;
    app.update();
    assert_eq!(
        visual_vis_of(&mut app, body),
        Visibility::Inherited,
        "body visual visible with isolate off"
    );
    assert_eq!(
        visual_vis_of(&mut app, arm),
        Visibility::Inherited,
        "arm visual visible with isolate off"
    );
}

/// Per-link override integration test: build the REAL scene (world→body→arm, primitive box visuals,
/// synchronous, no async glTF), global Visual ON, select `body`, then force a per-link override
/// `{ID_VISUAL: false}` for `body` ONLY. Running the live `VisualDisplay` system must HIDE body's visual
/// while leaving arm's visible (Inherited), proving overrides scope to the selected comp and never leak
/// to other comps. Then simulate deselect (Selected=None) + run the real reset system: overrides clear
/// and body's visual returns (follows the still-ON global). Isolate is OFF throughout.
#[test]
fn per_link_override_hides_only_selected_comp_and_clears_on_deselect() {
    use bevy::asset::AssetPlugin;
    use hcdviz::display::{AddDisplayExt, DisplayRegistry};
    use hcdviz::pick::{
        reset_overrides_on_selection_change, IsolateSelection, IsolateSet, SelectionOverrides,
    };
    use hcdviz::scene::{OwnerComp, VisualDisplay, VisualItem, ID_VISUAL};

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<IsolateSelection>()
        .init_resource::<IsolateSet>()
        .init_resource::<SelectionOverrides>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin)
        .add_display(VisualDisplay); // registers "Visual" (default-on) + adds sync_visual_visibility

    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(NESTED).unwrap()));
    app.update(); // rebuild_on_change builds the scene (also clears Selected)
    app.update(); // flush command queue (children/visuals materialize)

    let (body, arm) = {
        let world = app.world_mut();
        let mut cq = world.query::<(Entity, &CompEntity)>();
        let by_name: HashMap<String, Entity> =
            cq.iter(world).map(|(e, c)| (c.name.clone(), e)).collect();
        (by_name["body"], by_name["arm"])
    };

    // Returns the (single) VisualItem visibility owned by `comp`.
    fn visual_vis_of(app: &mut App, comp: Entity) -> Visibility {
        let world = app.world_mut();
        let mut q = world.query_filtered::<(&OwnerComp, &Visibility), With<VisualItem>>();
        let found: Vec<Visibility> = q
            .iter(world)
            .filter(|(o, _)| o.0 == comp)
            .map(|(_, v)| *v)
            .collect();
        assert_eq!(
            found.len(),
            1,
            "expected exactly one VisualItem for the comp, got {}",
            found.len()
        );
        found[0]
    }

    // Select body and force a per-link Visual=OFF override for body only (global Visual is still ON).
    app.world_mut().resource_mut::<Selected>().0 = Some(body);
    {
        let mut ov = app.world_mut().resource_mut::<SelectionOverrides>();
        ov.comp = Some(body);
        ov.kinds.insert(ID_VISUAL, false);
    }
    app.update();
    assert_eq!(
        visual_vis_of(&mut app, body),
        Visibility::Hidden,
        "body's visual must be hidden by its per-link override"
    );
    assert_eq!(
        visual_vis_of(&mut app, arm),
        Visibility::Inherited,
        "a per-link override must NOT affect a non-selected comp (arm)"
    );

    // Deselect and run the REAL reset system → overrides clear, body's visual returns (follows global).
    app.add_systems(Update, reset_overrides_on_selection_change);
    app.world_mut().resource_mut::<Selected>().0 = None;
    app.update();
    assert!(
        app.world()
            .resource::<SelectionOverrides>()
            .kinds
            .is_empty(),
        "deselect must clear all per-link overrides"
    );
    assert_eq!(
        app.world().resource::<SelectionOverrides>().comp,
        None,
        "deselect must clear the override owner"
    );
    assert_eq!(
        visual_vis_of(&mut app, body),
        Visibility::Inherited,
        "body's visual must reappear after the override clears (global Visual still on)"
    );
    assert_eq!(
        visual_vis_of(&mut app, arm),
        Visibility::Inherited,
        "arm stays visible throughout"
    );
}

/// Pure resolution check over an openarm-like chain INCLUDING the glTF wrapper level
/// (mesh → WorldAssetRoot → comp → comp → basis → root). The walk must stop at the nearest comp.
#[test]
fn owning_comp_stops_at_nearest_comp_through_gltf_wrapper() {
    let mut w = World::new();
    let world_root = w.spawn_empty().id();
    let body_basis = w.spawn_empty().id();
    let comp_world = w.spawn_empty().id(); // the bare root comp "world"
    let comp_body = w.spawn_empty().id(); // its child comp, owns a glTF visual
    let gltf_root = w.spawn_empty().id(); // WorldAssetRoot entity (NOT a comp)
    let gltf_node = w.spawn_empty().id(); // glTF node (NOT a comp)
    let mesh = w.spawn_empty().id(); // the actual picked glTF mesh

    let parent_of: HashMap<Entity, Entity> = [
        (body_basis, world_root),
        (comp_world, body_basis),
        (comp_body, comp_world),
        (gltf_root, comp_body),
        (gltf_node, gltf_root),
        (mesh, gltf_node),
    ]
    .into_iter()
    .collect();
    let comps: HashSet<Entity> = [comp_world, comp_body].into_iter().collect();
    let resolve = |e| owning_comp(e, |x| parent_of.get(&x).copied(), |x| comps.contains(&x));

    // Clicking the glTF mesh selects its OWN comp (body), NOT the root (world). This is the core of
    // the bubbling-bug fix: resolution starts from the ORIGINAL hit, so it can never reach the root.
    assert_eq!(resolve(mesh), Some(comp_body));
    assert_ne!(
        resolve(mesh),
        Some(comp_world),
        "the hit must never resolve to the root comp"
    );
    assert_eq!(resolve(gltf_root), Some(comp_body));
    assert_eq!(resolve(comp_body), Some(comp_body));
    assert_eq!(resolve(comp_world), Some(comp_world));
    // Above all comps there is nothing to select.
    assert_eq!(resolve(body_basis), None);
    assert_eq!(resolve(world_root), None);
}
