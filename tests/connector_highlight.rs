//! Headless integration test for mesh-linked connectivity endpoint highlighting.
//!
//! Hand-builds the spawned-scene shape (comp → `visual:main` holder → glTF node `eth0` → primitive
//! mesh) plus an endpoint render entity carrying a `ConnectorMeshLink`, then drives the live
//! `ConnectorHighlightPlugin` through `ActiveConnector`: selecting the endpoint swaps the linked
//! mesh's material for a tinted clone, deselecting restores the ORIGINAL handle, toggling the
//! Connectivity display off/on clears and re-applies, and a link resolved only after its glTF
//! "loads" (the meshes spawn late) still gets highlighted.
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::connectivity::{
    CanonicalConnectivityGeneration, CanonicalConnectivityPickMapping, CanonicalConnectivityPlugin,
    CanonicalConnectivityRenderOwner, CanonicalConnectivityUpdate, SelectedConnectivityObject,
};
use hcdviz::connector::{
    ActiveConnector, ConnectivitySelectionPulse, ConnectorHighlightPlugin, ConnectorMeshLink,
};
use hcdviz::display::DisplayRegistry;
use hcdviz::pick::Selected;
use hcdviz::scene::{CompEntity, OwnerComp, VisualItem, ID_CONNECTIVITY};
use hcdviz::schema::connectivity::{
    normalize_connectivity, IdentityPart, ObjectIdentity, ObjectKind, StableObjectId,
};
use hcdviz::schema::model::connectivity as canonical;
use hcdviz::schema::model::connectivity::{DocumentIdentity, IncludeInstanceId};

const ETH_RGBA: [f32; 4] = [0.2, 0.8, 0.2, 1.0];

fn app_with_plugin() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<DisplayRegistry>()
        .add_plugins(ConnectorHighlightPlugin);
    // What `add_display(ConnectivityDisplay)` would register, enabled so the highlight is active.
    app.world_mut().resource_mut::<DisplayRegistry>().register(
        ID_CONNECTIVITY,
        "Connectivity".into(),
        true,
    );
    app
}

/// comp → holder(`visual:main`) → node(`eth0`) → primitive mesh with `original`; plus the glyph.
/// Returns (glyph, primitive mesh entity).
fn spawn_linked_scene(app: &mut App, original: Handle<StandardMaterial>) -> (Entity, Entity) {
    let world = app.world_mut();
    let comp = world.spawn_empty().id();
    let holder = world.spawn((Name::new("visual:main"), VisualItem)).id();
    let node = world.spawn(Name::new("eth0")).id();
    let prim = world
        .spawn((Name::new("eth0_mesh"), MeshMaterial3d(original)))
        .id();
    world.entity_mut(comp).add_child(holder);
    world.entity_mut(holder).add_child(node);
    world.entity_mut(node).add_child(prim);
    let glyph = world
        .spawn((
            ConnectorMeshLink::new(Some("main"), Some("eth0"), ETH_RGBA).expect("mesh present"),
            OwnerComp(comp),
        ))
        .id();
    (glyph, prim)
}

fn material_of(app: &mut App, e: Entity) -> Handle<StandardMaterial> {
    app.world()
        .entity(e)
        .get::<MeshMaterial3d<StandardMaterial>>()
        .expect("has material")
        .0
        .clone()
}

#[test]
fn select_tints_linked_mesh_and_deselect_restores() {
    let mut app = app_with_plugin();
    let original = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let (glyph, prim) = spawn_linked_scene(&mut app, original.clone());
    app.update(); // first-run doc tick settles with nothing active

    // Select the glyph → the primitive's material handle is swapped for a tinted CLONE.
    app.world_mut()
        .resource_mut::<ActiveConnector>()
        .set_selected(Some(glyph));
    app.update();
    let tinted = material_of(&mut app, prim);
    assert_ne!(
        tinted, original,
        "selection must swap in a highlight material"
    );
    {
        let materials = app.world().resource::<Assets<StandardMaterial>>();
        let m = materials.get(&tinted).expect("tinted material exists");
        assert!(m.emissive.red > 0.0, "highlight carries an emissive glow");
        // The ORIGINAL asset is untouched (it may be shared across the whole GLB).
        let o = materials.get(&original).expect("original still lives");
        assert_eq!(o.emissive, StandardMaterial::default().emissive);
    }

    // Deselect → the exact original handle returns.
    app.world_mut()
        .resource_mut::<ActiveConnector>()
        .clear_selected_targets();
    app.update();
    assert_eq!(
        material_of(&mut app, prim),
        original,
        "deselect must restore the original"
    );
}

#[test]
fn display_off_clears_and_reenable_reapplies() {
    let mut app = app_with_plugin();
    let original = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let (glyph, prim) = spawn_linked_scene(&mut app, original.clone());
    app.update();
    app.world_mut()
        .resource_mut::<ActiveConnector>()
        .set_selected(Some(glyph));
    app.update();
    assert_ne!(material_of(&mut app, prim), original);

    // Connectivity display off → glyphs hide, so the tint must clear with them.
    app.world_mut()
        .resource_mut::<DisplayRegistry>()
        .set_enabled(ID_CONNECTIVITY, false);
    app.update();
    assert_eq!(
        material_of(&mut app, prim),
        original,
        "display-off must restore the original"
    );

    // Back on → the surviving selection re-applies.
    app.world_mut()
        .resource_mut::<DisplayRegistry>()
        .set_enabled(ID_CONNECTIVITY, true);
    app.update();
    assert_ne!(
        material_of(&mut app, prim),
        original,
        "re-enable must re-apply the highlight"
    );
}

#[test]
fn hover_tints_and_out_restores() {
    let mut app = app_with_plugin();
    let original = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let (glyph, prim) = spawn_linked_scene(&mut app, original.clone());
    app.update();

    app.world_mut().resource_mut::<ActiveConnector>().hovered = Some(glyph);
    app.update();
    assert_ne!(
        material_of(&mut app, prim),
        original,
        "hover must tint the linked mesh"
    );

    app.world_mut().resource_mut::<ActiveConnector>().hovered = None;
    app.update();
    assert_eq!(
        material_of(&mut app, prim),
        original,
        "hover-out must restore"
    );
}

#[test]
fn switching_same_named_nodes_between_visuals_moves_the_highlight() {
    let mut app = app_with_plugin();
    let original_a = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let original_b = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let (link_a, link_b, primitive_a, primitive_b) = {
        let world = app.world_mut();
        let comp = world.spawn_empty().id();
        let visual_a = world.spawn((Name::new("visual:a"), VisualItem)).id();
        let visual_b = world.spawn((Name::new("visual:b"), VisualItem)).id();
        let node_a = world.spawn(Name::new("pin")).id();
        let node_b = world.spawn(Name::new("pin")).id();
        let primitive_a = world.spawn(MeshMaterial3d(original_a.clone())).id();
        let primitive_b = world.spawn(MeshMaterial3d(original_b.clone())).id();
        world.entity_mut(comp).add_children(&[visual_a, visual_b]);
        world.entity_mut(visual_a).add_child(node_a);
        world.entity_mut(visual_b).add_child(node_b);
        world.entity_mut(node_a).add_child(primitive_a);
        world.entity_mut(node_b).add_child(primitive_b);
        let link_a = world
            .spawn((
                ConnectorMeshLink::new(Some("a"), Some("pin"), ETH_RGBA).expect("visual A link"),
                OwnerComp(comp),
            ))
            .id();
        let link_b = world
            .spawn((
                ConnectorMeshLink::new(Some("b"), Some("pin"), ETH_RGBA).expect("visual B link"),
                OwnerComp(comp),
            ))
            .id();
        (link_a, link_b, primitive_a, primitive_b)
    };
    app.update();

    app.world_mut()
        .resource_mut::<ActiveConnector>()
        .set_selected(Some(link_a));
    app.update();
    assert_ne!(material_of(&mut app, primitive_a), original_a);
    assert_eq!(material_of(&mut app, primitive_b), original_b);

    app.world_mut()
        .resource_mut::<ActiveConnector>()
        .set_selected(Some(link_b));
    app.update();
    assert_eq!(
        material_of(&mut app, primitive_a),
        original_a,
        "switching visual scope must restore the previously selected node"
    );
    assert_ne!(
        material_of(&mut app, primitive_b),
        original_b,
        "the same node name in the newly selected visual must receive the highlight"
    );
}

#[test]
fn late_loading_gltf_still_gets_highlighted() {
    let mut app = app_with_plugin();
    // Only the comp + empty holder exist at selection time: the "GLB still loading" state.
    let (comp, holder, glyph) = {
        let world = app.world_mut();
        let comp = world.spawn_empty().id();
        let holder = world.spawn((Name::new("visual:main"), VisualItem)).id();
        world.entity_mut(comp).add_child(holder);
        let glyph = world
            .spawn((
                ConnectorMeshLink::new(Some("main"), Some("eth0"), ETH_RGBA).expect("mesh present"),
                OwnerComp(comp),
            ))
            .id();
        (comp, holder, glyph)
    };
    let _ = comp;
    app.update();
    app.world_mut()
        .resource_mut::<ActiveConnector>()
        .set_selected(Some(glyph));
    app.update(); // resolves nothing yet, and must not panic or wedge

    // The glTF "finishes loading": node + primitive (with Mesh3d, whose Added tick is the retry
    // trigger) appear under the holder.
    let (original, prim) = {
        let mesh = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Cuboid::new(0.01, 0.01, 0.01));
        let original = app
            .world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .add(StandardMaterial::default());
        let world = app.world_mut();
        let node = world.spawn(Name::new("eth0")).id();
        let prim = world
            .spawn((
                Name::new("eth0_mesh"),
                Mesh3d(mesh),
                MeshMaterial3d(original.clone()),
            ))
            .id();
        world.entity_mut(holder).add_child(node);
        world.entity_mut(node).add_child(prim);
        (original, prim)
    };
    app.update();
    assert_ne!(
        material_of(&mut app, prim),
        original,
        "the still-selected link must resolve once the glTF meshes spawn"
    );
}

#[test]
fn exact_model_part_path_and_fallback_tint_the_declared_submesh() {
    let mut app = app_with_plugin();
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::new(0.01, 0.01, 0.01));
    let original = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let (glyph, prim) = {
        let world = app.world_mut();
        let comp = world.spawn_empty().id();
        let root = world.spawn(Name::new("model-root")).id();
        let fallback = world.spawn(Name::new("J1_fallback")).id();
        let prim = world
            .spawn((
                Name::new("J1_primitive"),
                Mesh3d(mesh),
                MeshMaterial3d(original.clone()),
            ))
            .id();
        world.entity_mut(root).add_child(fallback);
        world.entity_mut(fallback).add_child(prim);
        let glyph = world
            .spawn((
                ConnectorMeshLink::exact(root, Some("missing/J1"), Some("J1_fallback"), ETH_RGBA),
                OwnerComp(comp),
            ))
            .id();
        (glyph, prim)
    };
    app.update();
    app.world_mut()
        .resource_mut::<ActiveConnector>()
        .set_selected(Some(glyph));
    app.update();
    assert_ne!(
        material_of(&mut app, prim),
        original,
        "the exact-root resolver must use the authored fallback after the path misses"
    );

    app.world_mut()
        .resource_mut::<DisplayRegistry>()
        .set_enabled(ID_CONNECTIVITY, false);
    app.update();
    assert_ne!(
        material_of(&mut app, prim),
        original,
        "an explicit exact model-part selection must remain visible with passive endpoints off"
    );
}

#[test]
fn expanded_two_channel_port_tints_and_restores_both_exact_pin_meshes() {
    let mut app = app_with_plugin();
    let mesh = app
        .world_mut()
        .resource_mut::<Assets<Mesh>>()
        .add(Cuboid::new(0.01, 0.01, 0.01));
    let original_1 = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let original_2 = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let (pin_1_render, pin_2_render, prim_1, prim_2) = {
        let world = app.world_mut();
        let comp = world.spawn_empty().id();
        let root = world.spawn(Name::new("model-root")).id();
        let pin_1 = world.spawn(Name::new("pin-1")).id();
        let pin_2 = world.spawn(Name::new("pin-2")).id();
        let prim_1 = world
            .spawn((Mesh3d(mesh.clone()), MeshMaterial3d(original_1.clone())))
            .id();
        let prim_2 = world
            .spawn((Mesh3d(mesh), MeshMaterial3d(original_2.clone())))
            .id();
        world.entity_mut(root).add_children(&[pin_1, pin_2]);
        world.entity_mut(pin_1).add_child(prim_1);
        world.entity_mut(pin_2).add_child(prim_2);
        let pin_1_render = world
            .spawn((
                ConnectorMeshLink::exact(root, Some("pin-1"), None, ETH_RGBA),
                OwnerComp(comp),
            ))
            .id();
        let pin_2_render = world
            .spawn((
                ConnectorMeshLink::exact(root, Some("pin-2"), None, ETH_RGBA),
                OwnerComp(comp),
            ))
            .id();
        (pin_1_render, pin_2_render, prim_1, prim_2)
    };
    app.update();

    {
        let mut active = app.world_mut().resource_mut::<ActiveConnector>();
        active.set_selected_targets(vec![pin_1_render, pin_2_render]);
    }
    app.update();
    assert_ne!(material_of(&mut app, prim_1), original_1);
    assert_ne!(material_of(&mut app, prim_2), original_2);

    {
        let mut active = app.world_mut().resource_mut::<ActiveConnector>();
        active.clear_selected_targets();
    }
    app.update();
    assert_eq!(material_of(&mut app, prim_1), original_1);
    assert_eq!(material_of(&mut app, prim_2), original_2);
}

fn stable_object(kind: ObjectKind, field: &str, value: &str) -> StableObjectId {
    ObjectIdentity::new(
        DocumentIdentity::new("memory://hcdviz/direct-pick").unwrap(),
        IncludeInstanceId::root(),
        kind,
        vec![IdentityPart::new(field, value)],
    )
    .stable_id()
}

fn interaction_app(connector_observer_first: bool) -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<Selected>();
    if connector_observer_first {
        app.add_plugins(ConnectorHighlightPlugin)
            .add_observer(hcdviz::pick::on_click);
    } else {
        app.add_observer(hcdviz::pick::on_click)
            .add_plugins(ConnectorHighlightPlugin);
    }
    app
}

fn canonical_component(name: &str) -> canonical::ComponentConnectivity {
    canonical::ComponentConnectivity {
        component: name.to_owned(),
        ports: Vec::new(),
        connectors: vec![canonical::Connector {
            name: "J1".to_owned(),
            family: None,
            positions: vec![canonical::Position {
                name: "1".to_owned(),
                kind: canonical::PositionKind::Pin,
                role: None,
                local_group: None,
                representation: None,
            }],
            representation: None,
        }],
        antennas: Vec::new(),
        functions: Vec::new(),
        paths: Vec::new(),
        junctions: Vec::new(),
        terminations: Vec::new(),
    }
}

struct CanonicalInteractionScene {
    app: App,
    first_component: Entity,
    second_component: Entity,
    first_pin: StableObjectId,
    generation: CanonicalConnectivityGeneration,
}

fn canonical_interaction_scene() -> CanonicalInteractionScene {
    let mut document = canonical::ConnectivityDocument::new(
        DocumentIdentity::new("memory://hcdviz/canonical-pick").unwrap(),
    );
    for component in ["first", "second"] {
        document.scopes[0]
            .components
            .push(canonical_component(component));
        document.scopes[0]
            .structural_anchors
            .push(canonical::StructuralAnchors {
                component: component.to_owned(),
                visuals: Vec::new(),
                frames: Vec::new(),
            });
    }
    let graph = normalize_connectivity(&document).unwrap();
    let first_pin = graph
        .resolver(IncludeInstanceId::root())
        .position(&canonical::PositionRef::local_component("first", "J1", "1"))
        .unwrap()
        .id()
        .clone();

    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<Selected>()
        .add_plugins(CanonicalConnectivityPlugin)
        .add_plugins(ConnectorHighlightPlugin)
        .add_observer(hcdviz::pick::on_click);
    app.world_mut().resource_mut::<DisplayRegistry>().register(
        ID_CONNECTIVITY,
        "Connectivity".into(),
        true,
    );
    app.update();
    let generation = *app.world().resource::<CanonicalConnectivityGeneration>();
    let first_component = app
        .world_mut()
        .spawn(CompEntity {
            comp_index: 0,
            name: "first".to_owned(),
        })
        .id();
    let second_component = app
        .world_mut()
        .spawn(CompEntity {
            comp_index: 1,
            name: "second".to_owned(),
        })
        .id();
    app.world_mut()
        .write_message(CanonicalConnectivityUpdate::ready(generation, graph));
    app.update();

    CanonicalInteractionScene {
        app,
        first_component,
        second_component,
        first_pin,
        generation,
    }
}

struct PickScene {
    comp: Entity,
    shell_primitive: Entity,
    pin_primitive: Entity,
    connector: StableObjectId,
    position: StableObjectId,
    pin_render: Entity,
}

fn spawn_exact_pick_scene(app: &mut App) -> PickScene {
    let connector = stable_object(ObjectKind::Connector, "connector", "J1");
    let position = stable_object(ObjectKind::Position, "position", "1");
    let world = app.world_mut();
    let comp = world
        .spawn(CompEntity {
            comp_index: 0,
            name: "base".to_owned(),
        })
        .id();
    let root = world.spawn_empty().id();
    let shell = world.spawn_empty().id();
    let shell_primitive = world.spawn_empty().id();
    let pin = world.spawn_empty().id();
    let pin_primitive = world.spawn_empty().id();
    world.entity_mut(comp).add_child(root);
    world.entity_mut(root).add_child(shell);
    world
        .entity_mut(shell)
        .add_children(&[shell_primitive, pin]);
    world.entity_mut(pin).add_child(pin_primitive);

    let connector_render = world
        .spawn((
            CanonicalConnectivityRenderOwner(connector.clone()),
            ConnectorMeshLink::exact(root, Some("shell"), None, ETH_RGBA),
            OwnerComp(comp),
        ))
        .id();
    let connector_mapping = world
        .spawn(CanonicalConnectivityPickMapping {
            object: connector.clone(),
            render_entity: connector_render,
            kind: ObjectKind::Connector,
        })
        .id();
    world.entity_mut(shell).add_child(connector_mapping);

    let pin_render = world
        .spawn((
            CanonicalConnectivityRenderOwner(position.clone()),
            ConnectorMeshLink::exact(root, Some("shell/pin"), None, ETH_RGBA),
            OwnerComp(comp),
        ))
        .id();
    let pin_mapping = world
        .spawn(CanonicalConnectivityPickMapping {
            object: position.clone(),
            render_entity: pin_render,
            kind: ObjectKind::Position,
        })
        .id();
    world.entity_mut(pin).add_child(pin_mapping);

    PickScene {
        comp,
        shell_primitive,
        pin_primitive,
        connector,
        position,
        pin_render,
    }
}

fn pointer_location() -> bevy::picking::pointer::Location {
    use bevy::camera::{ManualTextureViewHandle, NormalizedRenderTarget};
    bevy::picking::pointer::Location {
        target: NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0)),
        position: Vec2::ZERO,
    }
}

fn pointer_hit() -> bevy::picking::backend::HitData {
    bevy::picking::backend::HitData::new(Entity::PLACEHOLDER, 0.0, None, None)
}

fn trigger_primary_click(app: &mut App, target: Entity) {
    use bevy::picking::events::{Click, Pointer};
    use bevy::picking::pointer::{PointerButton, PointerId};
    use std::time::Duration;
    app.world_mut().trigger(Pointer::new(
        PointerId::Mouse,
        pointer_location(),
        Click {
            button: PointerButton::Primary,
            hit: pointer_hit(),
            duration: Duration::ZERO,
            count: 1,
        },
        target,
    ));
}

#[test]
fn primitive_click_selects_nearest_pin_and_owning_component_in_either_observer_order() {
    for connector_observer_first in [true, false] {
        let mut app = interaction_app(connector_observer_first);
        let scene = spawn_exact_pick_scene(&mut app);

        trigger_primary_click(&mut app, scene.pin_primitive);
        assert_eq!(
            app.world().resource::<SelectedConnectivityObject>().0,
            Some(scene.position.clone()),
            "the nearest pin mapping must beat its connector-shell ancestor"
        );
        assert_eq!(app.world().resource::<Selected>().0, Some(scene.comp));
        assert_eq!(
            app.world().resource::<ActiveConnector>().selected(),
            Some(scene.pin_render)
        );

        trigger_primary_click(&mut app, scene.shell_primitive);
        assert_eq!(
            app.world().resource::<SelectedConnectivityObject>().0,
            Some(scene.connector),
            "shell geometry must select the connector mapping"
        );
        assert_eq!(app.world().resource::<Selected>().0, Some(scene.comp));
    }
}

#[test]
fn legacy_mesh_link_remains_clickable_without_a_canonical_owner_marker() {
    for connector_observer_first in [true, false] {
        let mut app = interaction_app(connector_observer_first);
        let comp = app
            .world_mut()
            .spawn(CompEntity {
                comp_index: 0,
                name: "base".to_owned(),
            })
            .id();
        let link = app
            .world_mut()
            .spawn((
                ConnectorMeshLink::new(Some("main"), Some("pin"), ETH_RGBA)
                    .expect("legacy mesh link"),
                OwnerComp(comp),
            ))
            .id();
        app.world_mut().entity_mut(comp).add_child(link);

        trigger_primary_click(&mut app, link);
        assert_eq!(
            app.world().resource::<ActiveConnector>().selected(),
            Some(link)
        );
        assert_eq!(app.world().resource::<Selected>().0, Some(comp));
        assert!(app
            .world()
            .resource::<SelectedConnectivityObject>()
            .0
            .is_none());
    }
}

#[test]
fn standalone_canonical_pin_click_selects_its_component_and_hierarchy_switch_clears_it() {
    let CanonicalInteractionScene {
        mut app,
        first_component,
        second_component,
        first_pin,
        ..
    } = canonical_interaction_scene();
    let root = app.world_mut().spawn_empty().id();
    let render = app
        .world_mut()
        .spawn((
            CanonicalConnectivityRenderOwner(first_pin.clone()),
            ConnectorMeshLink::exact(root, None, None, ETH_RGBA),
            OwnerComp(root),
        ))
        .id();
    let mapping = app
        .world_mut()
        .spawn(CanonicalConnectivityPickMapping {
            object: first_pin.clone(),
            render_entity: render,
            kind: ObjectKind::Position,
        })
        .id();
    app.world_mut().entity_mut(root).add_child(mapping);

    trigger_primary_click(&mut app, root);
    app.update();
    assert_eq!(
        app.world().resource::<SelectedConnectivityObject>().0,
        Some(first_pin)
    );
    assert_eq!(app.world().resource::<Selected>().0, Some(first_component));
    assert_eq!(
        app.world().resource::<ActiveConnector>().selected(),
        Some(render)
    );
    let first_pulse = app.world().resource::<ConnectivitySelectionPulse>().0;
    trigger_primary_click(&mut app, root);
    app.update();
    assert_eq!(
        app.world().resource::<ConnectivitySelectionPulse>().0,
        first_pulse.wrapping_add(1),
        "re-clicking the same pin must emit a fresh reveal pulse"
    );

    app.world_mut().resource_mut::<Selected>().0 = Some(second_component);
    app.update();
    assert!(app
        .world()
        .resource::<SelectedConnectivityObject>()
        .0
        .is_none());
    assert!(app
        .world()
        .resource::<ActiveConnector>()
        .selected()
        .is_none());
}

#[test]
fn semantic_owner_wins_when_a_model_part_is_hosted_by_another_component() {
    let CanonicalInteractionScene {
        mut app,
        first_component,
        second_component,
        first_pin,
        ..
    } = canonical_interaction_scene();
    let root = app.world_mut().spawn_empty().id();
    app.world_mut().entity_mut(second_component).add_child(root);
    let render = app
        .world_mut()
        .spawn((
            CanonicalConnectivityRenderOwner(first_pin.clone()),
            ConnectorMeshLink::exact(root, None, None, ETH_RGBA),
            OwnerComp(second_component),
        ))
        .id();
    let mapping = app
        .world_mut()
        .spawn(CanonicalConnectivityPickMapping {
            object: first_pin.clone(),
            render_entity: render,
            kind: ObjectKind::Position,
        })
        .id();
    app.world_mut().entity_mut(root).add_child(mapping);

    trigger_primary_click(&mut app, root);
    app.update();
    assert_eq!(
        app.world().resource::<SelectedConnectivityObject>().0,
        Some(first_pin)
    );
    assert_eq!(
        app.world().resource::<Selected>().0,
        Some(first_component),
        "the connector's semantic owner must win over the visual host"
    );
    assert_eq!(
        app.world().resource::<ActiveConnector>().selected(),
        Some(render)
    );
}

#[test]
fn canonical_replacement_clears_hover_and_restores_a_live_mesh() {
    let CanonicalInteractionScene {
        mut app,
        first_component,
        generation,
        ..
    } = canonical_interaction_scene();
    let original = app
        .world_mut()
        .resource_mut::<Assets<StandardMaterial>>()
        .add(StandardMaterial::default());
    let holder = app
        .world_mut()
        .spawn((Name::new("visual:main"), VisualItem))
        .id();
    let node = app.world_mut().spawn(Name::new("pin")).id();
    let primitive = app.world_mut().spawn(MeshMaterial3d(original.clone())).id();
    app.world_mut()
        .entity_mut(first_component)
        .add_child(holder);
    app.world_mut().entity_mut(holder).add_child(node);
    app.world_mut().entity_mut(node).add_child(primitive);
    let link = app
        .world_mut()
        .spawn((
            ConnectorMeshLink::new(Some("main"), Some("pin"), ETH_RGBA).expect("hover link"),
            OwnerComp(first_component),
        ))
        .id();
    app.world_mut().resource_mut::<ActiveConnector>().hovered = Some(link);
    app.update();
    assert_ne!(material_of(&mut app, primitive), original);

    app.world_mut()
        .write_message(CanonicalConnectivityUpdate::clear(generation));
    app.update();
    assert!(app.world().resource::<ActiveConnector>().hovered.is_none());
    assert_eq!(material_of(&mut app, primitive), original);
}

#[test]
fn primitive_hover_previews_nearest_pin_without_changing_stable_selection() {
    use bevy::picking::events::{Out, Over, Pointer};
    use bevy::picking::pointer::PointerId;

    let mut app = interaction_app(true);
    let scene = spawn_exact_pick_scene(&mut app);
    app.world_mut().trigger(Pointer::new(
        PointerId::Mouse,
        pointer_location(),
        Over { hit: pointer_hit() },
        scene.pin_primitive,
    ));
    assert_eq!(
        app.world().resource::<ActiveConnector>().hovered,
        Some(scene.pin_render)
    );
    assert!(app
        .world()
        .resource::<SelectedConnectivityObject>()
        .0
        .is_none());

    app.world_mut().trigger(Pointer::new(
        PointerId::Mouse,
        pointer_location(),
        Out { hit: pointer_hit() },
        scene.pin_primitive,
    ));
    assert!(app.world().resource::<ActiveConnector>().hovered.is_none());
}
