//! Canonical physical-representation realization.
//!
//! This module consumes only the immutable canonical connectivity graph plus exact structural
//! anchors. It renders standalone primitive and model representations, resolves model-part node
//! paths against component visuals or assembly models, and publishes stable owner and
//! representation mappings for the rest of hcdviz.

use std::collections::BTreeMap;

use bevy::gltf::GltfAssetLabel;
use bevy::prelude::*;
use bevy::world_serialization::WorldAssetRoot;

use crate::connectivity::{
    CanonicalConnectivityAnchor, CanonicalConnectivityGeneration, CanonicalConnectivityPickMapping,
    CanonicalConnectivityRenderOwner, CanonicalConnectivityState, ConnectivitySet,
};
use crate::connector::{resolve_exact_model_node, ConnectorMeshLink};
use crate::geometry::{resolve_connectivity_primitive, GeometryCache};
use crate::scene::{CompEntity, OwnerComp, SceneItem, VisualItem};
use crate::schema::connectivity::{
    ConnectivityNodeData, EdgeKind, NormalizedConnectivityGraph, ObjectKind, StableObjectId,
};
use crate::schema::model::connectivity::{Placement, PlacementRotation, Representation};

const PHYSICAL_RGBA: [f32; 4] = [0.18, 0.62, 0.82, 0.92];
const HIGHLIGHT_RGBA: [f32; 4] = [1.0, 0.58, 0.12, 1.0];

/// Marks a scene entity created to realize a canonical physical representation.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct CanonicalPhysicalRepresentation(pub StableObjectId);

#[derive(Debug, Clone)]
struct RepresentationPlan {
    owner: StableObjectId,
    owner_kind: ObjectKind,
    representation: StableObjectId,
    value: Representation,
    frame: Option<StableObjectId>,
    model_root: Option<StableObjectId>,
}

#[derive(Debug, Clone, Copy)]
struct Realization {
    holder: Option<Entity>,
    target: Option<Entity>,
    scene_owner: Entity,
    mapped: bool,
}

#[derive(Resource, Default)]
struct PhysicalPresentationRegistry {
    initialized: bool,
    generation: Option<CanonicalConnectivityGeneration>,
    realizations: BTreeMap<StableObjectId, Realization>,
    owned_roots: Vec<Entity>,
}

#[derive(bevy::ecs::system::SystemParam)]
struct PhysicalScene<'w, 's> {
    anchors: Query<
        'w,
        's,
        (
            Entity,
            &'static CanonicalConnectivityAnchor,
            Option<&'static OwnerComp>,
            Has<CompEntity>,
        ),
    >,
    names: Query<'w, 's, &'static Name>,
    children: Query<'w, 's, &'static Children>,
    added_anchors: Query<'w, 's, (), Added<CanonicalConnectivityAnchor>>,
    added_names: Query<'w, 's, (), Added<Name>>,
    added_meshes: Query<'w, 's, (), Added<Mesh3d>>,
}

#[derive(bevy::ecs::system::SystemParam)]
struct PhysicalAssets<'w> {
    meshes: Option<ResMut<'w, Assets<Mesh>>>,
    materials: Option<ResMut<'w, Assets<StandardMaterial>>>,
    cache: Option<ResMut<'w, GeometryCache>>,
    asset_server: Option<Res<'w, AssetServer>>,
    mem_store: Option<Res<'w, crate::mem_assets::MemAssetStore>>,
}

/// Realizes canonical physical declarations and publishes exact stable-ID scene mappings.
pub struct CanonicalPhysicalPresentationPlugin;

impl Plugin for CanonicalPhysicalPresentationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GeometryCache>()
            .init_resource::<PhysicalPresentationRegistry>()
            .add_systems(
                Update,
                update_physical_presentations
                    .in_set(ConnectivitySet::Index)
                    .before(crate::connectivity::rebuild_canonical_scene_index),
            );
    }
}

fn representation_plans(graph: &NormalizedConnectivityGraph) -> Vec<RepresentationPlan> {
    let mut plans = graph
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Representation)
        .filter_map(|edge| {
            let owner = graph.node(edge.from())?;
            let representation = graph.node(edge.to())?;
            if !is_physical_owner(owner.kind()) {
                return None;
            }
            let ConnectivityNodeData::Representation {
                representation: value,
            } = representation.data()
            else {
                return None;
            };
            let frame = graph
                .edges()
                .iter()
                .find(|candidate| {
                    candidate.kind() == EdgeKind::RepresentationFrame
                        && candidate.to() == representation.id()
                })
                .map(|candidate| candidate.from().clone());
            let model_root = graph
                .edges()
                .iter()
                .find(|candidate| {
                    candidate.kind() == EdgeKind::RepresentationModelRoot
                        && candidate.to() == representation.id()
                })
                .map(|candidate| candidate.from().clone());
            Some(RepresentationPlan {
                owner: owner.id().clone(),
                owner_kind: owner.kind(),
                representation: representation.id().clone(),
                value: value.clone(),
                frame,
                model_root,
            })
        })
        .collect::<Vec<_>>();
    plans.sort_by(|first, second| first.representation.cmp(&second.representation));
    plans
}

fn is_physical_owner(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::PhysicalAssembly
            | ObjectKind::Connector
            | ObjectKind::Position
            | ObjectKind::Antenna
            | ObjectKind::PhysicalPath
            | ObjectKind::Junction
            | ObjectKind::Termination
    )
}

fn update_physical_presentations(
    mut commands: Commands,
    state: Res<CanonicalConnectivityState>,
    generation: Res<CanonicalConnectivityGeneration>,
    mut registry: ResMut<PhysicalPresentationRegistry>,
    scene: PhysicalScene,
    mut assets: PhysicalAssets,
) {
    let reset =
        !registry.initialized || registry.generation != Some(*generation) || state.is_changed();
    if reset {
        for entity in registry.owned_roots.drain(..) {
            commands.entity(entity).try_despawn();
        }
        registry.realizations.clear();
        registry.initialized = true;
        registry.generation = Some(*generation);
    }

    let grew = !scene.added_anchors.is_empty()
        || !scene.added_names.is_empty()
        || !scene.added_meshes.is_empty();
    if !reset && !grew {
        return;
    }
    let Some(graph) = state.graph() else {
        return;
    };

    let anchors = exact_anchor_map(&scene);
    let plans = representation_plans(graph);
    for plan in plans.iter().filter(|plan| {
        matches!(
            plan.value,
            Representation::Primitive { .. } | Representation::Model { .. }
        )
    }) {
        realize_standalone(
            &mut commands,
            plan,
            &anchors,
            &scene,
            &mut assets,
            &mut registry,
        );
    }
    for plan in plans
        .iter()
        .filter(|plan| matches!(plan.value, Representation::ModelPart(_)))
    {
        realize_model_part(&mut commands, plan, graph, &anchors, &scene, &mut registry);
    }
}

fn exact_anchor_map(scene: &PhysicalScene) -> BTreeMap<StableObjectId, (Entity, Entity)> {
    let mut candidates = scene
        .anchors
        .iter()
        .map(|(entity, anchor, owner, is_component)| {
            let scene_owner = if is_component {
                entity
            } else {
                owner.map_or(entity, |owner| owner.0)
            };
            (anchor.0.clone(), entity, scene_owner)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|first, second| {
        first
            .0
            .cmp(&second.0)
            .then_with(|| first.1.index().cmp(&second.1.index()))
    });
    let mut anchors = BTreeMap::new();
    for (id, entity, owner) in candidates {
        anchors.entry(id).or_insert((entity, owner));
    }
    anchors
}

fn realize_standalone(
    commands: &mut Commands,
    plan: &RepresentationPlan,
    anchors: &BTreeMap<StableObjectId, (Entity, Entity)>,
    scene: &PhysicalScene,
    assets: &mut PhysicalAssets,
    registry: &mut PhysicalPresentationRegistry,
) {
    if !registry.realizations.contains_key(&plan.representation) {
        let frame = match &plan.frame {
            Some(frame) => match anchors.get(frame).copied() {
                Some(frame) => Some(frame),
                None => return,
            },
            None => None,
        };
        let spawned = match &plan.value {
            Representation::Primitive {
                primitive,
                placement,
            } => spawn_primitive(commands, plan, primitive, placement, frame, assets),
            Representation::Model { model, placement } => {
                spawn_model(commands, plan, &model.uri, placement, frame, assets)
            }
            Representation::ModelPart(_) | Representation::DerivedRoute(_) => None,
        };
        let Some((holder, scene_owner)) = spawned else {
            return;
        };
        registry.owned_roots.push(holder);
        registry.realizations.insert(
            plan.representation.clone(),
            Realization {
                holder: Some(holder),
                target: None,
                scene_owner,
                mapped: false,
            },
        );
    }

    let Some(mut realization) = registry.realizations.get(&plan.representation).copied() else {
        return;
    };
    if realization.mapped {
        return;
    }
    let Some(holder) = realization.holder else {
        return;
    };
    let selector = match &plan.value {
        Representation::Model { model, .. } => model.node_path.as_deref(),
        Representation::Primitive { .. } => None,
        Representation::ModelPart(_) | Representation::DerivedRoute(_) => return,
    };
    let target = selector
        .and_then(split_node_path)
        .and_then(|path| exact_model_node(holder, Some(&path), None, scene))
        .or_else(|| selector.is_none().then_some(holder));
    let Some(target) = target else {
        return;
    };
    let link = matches!(
        plan.owner_kind,
        ObjectKind::Connector | ObjectKind::Position
    )
    .then(|| ConnectorMeshLink::exact(holder, selector, None, HIGHLIGHT_RGBA));
    spawn_mapping(commands, target, realization.scene_owner, plan, link);
    realization.target = Some(target);
    realization.mapped = true;
    registry
        .realizations
        .insert(plan.representation.clone(), realization);
}

fn realize_model_part(
    commands: &mut Commands,
    plan: &RepresentationPlan,
    graph: &NormalizedConnectivityGraph,
    anchors: &BTreeMap<StableObjectId, (Entity, Entity)>,
    scene: &PhysicalScene,
    registry: &mut PhysicalPresentationRegistry,
) {
    if registry
        .realizations
        .get(&plan.representation)
        .is_some_and(|realization| realization.mapped)
    {
        return;
    }
    let Representation::ModelPart(model_part) = &plan.value else {
        return;
    };
    let Some(root_id) = &plan.model_root else {
        return;
    };
    let Some(root_node) = graph.node(root_id) else {
        return;
    };
    let (root, scene_owner, external_root) = match root_node.kind() {
        ObjectKind::StructuralVisualRoot => {
            let Some((entity, owner)) = anchors.get(root_id).copied() else {
                return;
            };
            (entity, owner, true)
        }
        ObjectKind::Representation => {
            let Some(realization) = registry.realizations.get(root_id).copied() else {
                return;
            };
            let Some(target) = realization.target else {
                return;
            };
            (target, realization.scene_owner, false)
        }
        _ => return,
    };
    let Some(path) = split_node_path(&model_part.node_path) else {
        return;
    };
    let Some(target) = exact_model_node(
        root,
        Some(&path),
        model_part.submesh_fallback.as_deref(),
        scene,
    ) else {
        return;
    };
    let link = matches!(
        plan.owner_kind,
        ObjectKind::Connector | ObjectKind::Position
    )
    .then(|| {
        ConnectorMeshLink::exact(
            root,
            Some(&model_part.node_path),
            model_part.submesh_fallback.as_deref(),
            HIGHLIGHT_RGBA,
        )
    });
    let mapping = spawn_mapping(commands, target, scene_owner, plan, link);
    if external_root {
        registry.owned_roots.push(mapping);
    }
    registry.realizations.insert(
        plan.representation.clone(),
        Realization {
            holder: None,
            target: Some(target),
            scene_owner,
            mapped: true,
        },
    );
}

fn split_node_path(value: &str) -> Option<Vec<String>> {
    let parts = value
        .split('/')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    (!parts.is_empty()).then_some(parts)
}

fn exact_model_node(
    root: Entity,
    path: Option<&[String]>,
    fallback: Option<&str>,
    scene: &PhysicalScene,
) -> Option<Entity> {
    resolve_exact_model_node(
        root,
        path,
        fallback,
        &|entity| {
            scene
                .names
                .get(entity)
                .ok()
                .map(|name| name.as_str().to_owned())
        },
        &|entity| {
            scene
                .children
                .get(entity)
                .map(|children| children.iter().collect())
                .unwrap_or_default()
        },
    )
}

fn spawn_primitive(
    commands: &mut Commands,
    plan: &RepresentationPlan,
    primitive: &crate::schema::model::connectivity::PrimitiveRepresentation,
    placement: &Placement,
    frame: Option<(Entity, Entity)>,
    assets: &mut PhysicalAssets,
) -> Option<(Entity, Entity)> {
    let meshes = assets.meshes.as_mut()?;
    let materials = assets.materials.as_mut()?;
    let cache = assets.cache.as_mut()?;
    let resolved = resolve_connectivity_primitive(primitive, cache, meshes);
    let material = cache.material(PHYSICAL_RGBA, materials);
    let scene_owner = frame.map(|(_, owner)| owner);
    let root = commands
        .spawn((
            SceneItem,
            VisualItem,
            CanonicalPhysicalRepresentation(plan.representation.clone()),
            Mesh3d(resolved.mesh),
            MeshMaterial3d(material),
            placement_transform(placement).with_scale(resolved.scale),
            Visibility::default(),
            Name::new(format!(
                "connectivity-representation:{}",
                plan.representation.as_str()
            )),
        ))
        .id();
    let scene_owner = scene_owner.unwrap_or(root);
    commands.entity(root).insert(OwnerComp(scene_owner));
    if let Some((parent, _)) = frame {
        commands.entity(parent).add_child(root);
    }
    Some((root, scene_owner))
}

fn spawn_model(
    commands: &mut Commands,
    plan: &RepresentationPlan,
    uri: &str,
    placement: &Placement,
    frame: Option<(Entity, Entity)>,
    assets: &PhysicalAssets,
) -> Option<(Entity, Entity)> {
    let asset_server = assets.asset_server.as_deref()?;
    let path = assets
        .mem_store
        .as_deref()
        .map_or_else(|| uri.to_owned(), |store| store.asset_path(uri));
    let scene_owner = frame.map(|(_, owner)| owner);
    let root = commands
        .spawn((
            SceneItem,
            VisualItem,
            CanonicalPhysicalRepresentation(plan.representation.clone()),
            WorldAssetRoot(asset_server.load(GltfAssetLabel::Scene(0).from_asset(path))),
            placement_transform(placement),
            Visibility::default(),
            Name::new(format!(
                "connectivity-representation:{}",
                plan.representation.as_str()
            )),
        ))
        .id();
    let scene_owner = scene_owner.unwrap_or(root);
    commands.entity(root).insert(OwnerComp(scene_owner));
    if let Some((parent, _)) = frame {
        commands.entity(parent).add_child(root);
    }
    Some((root, scene_owner))
}

fn spawn_mapping(
    commands: &mut Commands,
    target: Entity,
    scene_owner: Entity,
    plan: &RepresentationPlan,
    link: Option<ConnectorMeshLink>,
) -> Entity {
    let mapping = commands
        .spawn((
            SceneItem,
            Transform::IDENTITY,
            Visibility::Inherited,
            Name::new(format!(
                "connectivity-mapping:{}",
                plan.representation.as_str()
            )),
        ))
        .id();
    let mut owner = commands.spawn((
        SceneItem,
        CanonicalConnectivityAnchor(plan.owner.clone()),
        CanonicalConnectivityRenderOwner(plan.owner.clone()),
        OwnerComp(scene_owner),
        Transform::IDENTITY,
        Visibility::Inherited,
        Name::new(format!("connectivity-owner:{}", plan.owner.as_str())),
    ));
    if let Some(link) = link {
        owner.insert(link);
    }
    let owner = owner.id();
    commands
        .entity(mapping)
        .insert(CanonicalConnectivityPickMapping {
            object: plan.owner.clone(),
            render_entity: owner,
            kind: plan.owner_kind,
        });
    let representation = commands
        .spawn((
            SceneItem,
            CanonicalConnectivityAnchor(plan.representation.clone()),
            CanonicalConnectivityRenderOwner(plan.representation.clone()),
            OwnerComp(scene_owner),
            Transform::IDENTITY,
            Visibility::Inherited,
            Name::new(format!(
                "connectivity-representation-anchor:{}",
                plan.representation.as_str()
            )),
        ))
        .id();
    commands
        .entity(mapping)
        .add_children(&[owner, representation]);
    commands.entity(target).add_child(mapping);
    mapping
}

fn placement_transform(placement: &Placement) -> Transform {
    let translation = Vec3::new(
        placement.xyz[0] as f32,
        placement.xyz[1] as f32,
        placement.xyz[2] as f32,
    );
    let rotation = match placement.rotation {
        PlacementRotation::Rpy([roll, pitch, yaw]) => {
            Quat::from_rotation_z(yaw as f32)
                * Quat::from_rotation_y(pitch as f32)
                * Quat::from_rotation_x(roll as f32)
        }
        PlacementRotation::Quaternion([x, y, z, w]) => {
            Quat::from_xyzw(x as f32, y as f32, z as f32, w as f32)
        }
    };
    Transform::from_translation(translation).with_rotation(rotation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::{
        CanonicalConnectivityPlugin, CanonicalConnectivitySceneIndex, CanonicalConnectivityUpdate,
        SelectedConnectivityObject,
    };
    use crate::connector::{ActiveConnector, ConnectorHighlightPlugin};
    use crate::pick::Selected;
    use crate::schema::connectivity::normalize_connectivity;
    use crate::schema::model::connectivity as c;
    use bevy::asset::AssetPlugin;

    fn model_part(root: c::ModelRootRef, path: &str, fallback: Option<&str>) -> Representation {
        Representation::ModelPart(c::ModelPartRepresentation {
            model_root: root,
            node_path: path.to_owned(),
            submesh_fallback: fallback.map(str::to_owned),
        })
    }

    fn empty_component(name: &str) -> c::ComponentConnectivity {
        c::ComponentConnectivity {
            component: name.to_owned(),
            ports: Vec::new(),
            connectors: Vec::new(),
            antennas: Vec::new(),
            functions: Vec::new(),
            paths: Vec::new(),
            junctions: Vec::new(),
            terminations: Vec::new(),
        }
    }

    fn component_scope(
        instance: c::IncludeInstanceId,
        connector_path: &str,
        fallback: Option<&str>,
    ) -> c::ConnectivityScope {
        let root = c::ModelRootRef::ComponentVisual {
            component: c::ComponentRef::local("base"),
            visual: "body".to_owned(),
        };
        let mut component = empty_component("base");
        component.connectors.push(c::Connector {
            name: "J1".to_owned(),
            family: None,
            positions: vec![c::Position {
                name: "1".to_owned(),
                kind: c::PositionKind::Pin,
                role: None,
                local_group: None,
                representation: Some(model_part(root.clone(), "shell/J1/pin1", None)),
            }],
            representation: Some(model_part(root, connector_path, fallback)),
        });
        let mut scope = c::ConnectivityScope::new(instance);
        scope.components.push(component);
        scope.structural_anchors.push(c::StructuralAnchors {
            component: "base".to_owned(),
            visuals: vec![c::StructuralVisual {
                name: "body".to_owned(),
                model_backed: true,
            }],
            frames: Vec::new(),
        });
        scope
    }

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(AssetPlugin::default())
            .init_asset::<Mesh>()
            .init_asset::<StandardMaterial>()
            .init_asset::<bevy::world_serialization::WorldAsset>()
            .add_plugins(CanonicalConnectivityPlugin)
            .add_plugins(CanonicalPhysicalPresentationPlugin)
            .add_plugins(ConnectorHighlightPlugin)
            .add_observer(crate::pick::on_click);
        app.update();
        app
    }

    fn publish(app: &mut App, graph: NormalizedConnectivityGraph) {
        let generation = *app.world().resource::<CanonicalConnectivityGeneration>();
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(generation, graph));
        app.update();
    }

    fn representation_id(
        graph: &NormalizedConnectivityGraph,
        owner: &StableObjectId,
    ) -> StableObjectId {
        graph
            .edges()
            .iter()
            .find(|edge| edge.kind() == EdgeKind::Representation && edge.from() == owner)
            .expect("owner representation edge")
            .to()
            .clone()
    }

    fn spawn_component_visual(app: &mut App, visual_id: StableObjectId) -> (Entity, Entity) {
        let component = app
            .world_mut()
            .spawn((
                CompEntity {
                    comp_index: 0,
                    name: "base".to_owned(),
                },
                Transform::IDENTITY,
            ))
            .id();
        let holder = app
            .world_mut()
            .spawn((
                VisualItem,
                OwnerComp(component),
                CanonicalConnectivityAnchor(visual_id),
                Transform::IDENTITY,
                Name::new("visual:body"),
            ))
            .id();
        app.world_mut().entity_mut(component).add_child(holder);
        (component, holder)
    }

    fn spawn_named_child(app: &mut App, parent: Entity, name: &str) -> Entity {
        let child = app
            .world_mut()
            .spawn((Transform::IDENTITY, Name::new(name.to_owned())))
            .id();
        app.world_mut().entity_mut(parent).add_child(child);
        child
    }

    fn trigger_primary_click(app: &mut App, target: Entity) {
        use bevy::camera::{ManualTextureViewHandle, NormalizedRenderTarget};
        use bevy::picking::backend::HitData;
        use bevy::picking::events::{Click, Pointer};
        use bevy::picking::pointer::{Location, PointerButton, PointerId};
        use std::time::Duration;

        app.world_mut().trigger(Pointer::new(
            PointerId::Mouse,
            Location {
                target: NormalizedRenderTarget::TextureView(ManualTextureViewHandle(0)),
                position: Vec2::ZERO,
            },
            Click {
                button: PointerButton::Primary,
                hit: HitData::new(Entity::PLACEHOLDER, 0.0, None, None),
                duration: Duration::ZERO,
                count: 1,
            },
            target,
        ));
    }

    #[test]
    fn component_model_parts_publish_exact_connector_position_and_representation_mappings() {
        let mut document = c::ConnectivityDocument::new(
            c::DocumentIdentity::new("memory://hcdviz/physical-parts").unwrap(),
        );
        document.scopes[0] = component_scope(c::IncludeInstanceId::root(), "shell/J1", None);
        let graph = normalize_connectivity(&document).unwrap();
        let resolver = graph.resolver(c::IncludeInstanceId::root());
        let visual = resolver
            .visual_root(&c::StructuralVisualRef::local("base", "body"))
            .unwrap()
            .id()
            .clone();
        let connector = resolver
            .connector(&c::ConnectorRef::local_component("base", "J1"))
            .unwrap()
            .id()
            .clone();
        let position = resolver
            .position(&c::PositionRef::local_component("base", "J1", "1"))
            .unwrap()
            .id()
            .clone();
        let connector_representation = representation_id(&graph, &connector);
        let position_representation = representation_id(&graph, &position);

        let mut app = app();
        let (_, holder) = spawn_component_visual(&mut app, visual);
        let shell = spawn_named_child(&mut app, holder, "shell");
        let j1 = spawn_named_child(&mut app, shell, "J1");
        spawn_named_child(&mut app, j1, "pin1");
        publish(&mut app, graph);

        let index = app.world().resource::<CanonicalConnectivitySceneIndex>();
        for id in [
            &connector,
            &position,
            &connector_representation,
            &position_representation,
        ] {
            assert!(
                index.anchor_entity(id).is_some(),
                "missing exact anchor for {id}"
            );
            assert!(!index.render_entities(id).is_empty());
        }
        let connector_entity = index.render_entities(&connector)[0];
        let link = app
            .world()
            .get::<ConnectorMeshLink>(connector_entity)
            .expect("connector mapping carries exact mesh link");
        assert_eq!(link.exact_root(), Some(holder));
        assert_eq!(link.node_path(), Some("shell/J1"));
        assert!(app
            .world()
            .get::<crate::scene::ConnectorMarker>(connector_entity)
            .is_none());
    }

    #[test]
    fn broken_path_is_omitted_then_late_node_registration_replaces_it_exactly() {
        let mut document = c::ConnectivityDocument::new(
            c::DocumentIdentity::new("memory://hcdviz/late-physical-part").unwrap(),
        );
        document.scopes[0] = component_scope(c::IncludeInstanceId::root(), "shell/J1", None);
        let graph = normalize_connectivity(&document).unwrap();
        let resolver = graph.resolver(c::IncludeInstanceId::root());
        let visual = resolver
            .visual_root(&c::StructuralVisualRef::local("base", "body"))
            .unwrap()
            .id()
            .clone();
        let connector = resolver
            .connector(&c::ConnectorRef::local_component("base", "J1"))
            .unwrap()
            .id()
            .clone();

        let mut app = app();
        let (_, holder) = spawn_component_visual(&mut app, visual);
        publish(&mut app, graph);
        assert!(app
            .world()
            .resource::<CanonicalConnectivitySceneIndex>()
            .anchor_entity(&connector)
            .is_none());

        let shell = spawn_named_child(&mut app, holder, "shell");
        spawn_named_child(&mut app, shell, "J1");
        app.update();
        let anchor = app
            .world()
            .resource::<CanonicalConnectivitySceneIndex>()
            .anchor_entity(&connector)
            .expect("late exact node publishes connector anchor");
        assert_ne!(
            anchor, holder,
            "exact mapping must not use the visual root fallback"
        );
    }

    #[test]
    fn same_generation_replacement_removes_the_old_exact_mapping() {
        let identity = c::DocumentIdentity::new("memory://hcdviz/replaced-physical-part").unwrap();
        let mut first_document = c::ConnectivityDocument::new(identity.clone());
        first_document.scopes[0] = component_scope(c::IncludeInstanceId::root(), "shell/J1", None);
        let first_graph = normalize_connectivity(&first_document).unwrap();
        let resolver = first_graph.resolver(c::IncludeInstanceId::root());
        let visual = resolver
            .visual_root(&c::StructuralVisualRef::local("base", "body"))
            .unwrap()
            .id()
            .clone();
        let connector = resolver
            .connector(&c::ConnectorRef::local_component("base", "J1"))
            .unwrap()
            .id()
            .clone();

        let mut app = app();
        let (_, holder) = spawn_component_visual(&mut app, visual);
        let shell = spawn_named_child(&mut app, holder, "shell");
        spawn_named_child(&mut app, shell, "J1");
        spawn_named_child(&mut app, shell, "J2");
        publish(&mut app, first_graph);
        let old = app
            .world()
            .resource::<CanonicalConnectivitySceneIndex>()
            .anchor_entity(&connector)
            .unwrap();

        let mut second_document = c::ConnectivityDocument::new(identity);
        second_document.scopes[0] = component_scope(c::IncludeInstanceId::root(), "shell/J2", None);
        publish(&mut app, normalize_connectivity(&second_document).unwrap());
        let index = app.world().resource::<CanonicalConnectivitySceneIndex>();
        let new = index.anchor_entity(&connector).unwrap();
        assert_ne!(new, old);
        assert!(app.world().get_entity(old).is_err());
        assert_eq!(
            app.world()
                .get::<ConnectorMeshLink>(index.render_entities(&connector)[0])
                .unwrap()
                .node_path(),
            Some("shell/J2")
        );
    }

    #[test]
    fn authored_submesh_fallback_is_used_only_after_the_node_path_misses() {
        let mut document = c::ConnectivityDocument::new(
            c::DocumentIdentity::new("memory://hcdviz/physical-fallback").unwrap(),
        );
        document.scopes[0] = component_scope(
            c::IncludeInstanceId::root(),
            "missing/connector",
            Some("J1_fallback"),
        );
        let graph = normalize_connectivity(&document).unwrap();
        let resolver = graph.resolver(c::IncludeInstanceId::root());
        let visual = resolver
            .visual_root(&c::StructuralVisualRef::local("base", "body"))
            .unwrap()
            .id()
            .clone();
        let connector = resolver
            .connector(&c::ConnectorRef::local_component("base", "J1"))
            .unwrap()
            .id()
            .clone();

        let mut app = app();
        let (_, holder) = spawn_component_visual(&mut app, visual);
        spawn_named_child(&mut app, holder, "J1_fallback");
        publish(&mut app, graph);
        let index = app.world().resource::<CanonicalConnectivitySceneIndex>();
        let entity = index.render_entities(&connector)[0];
        let link = app.world().get::<ConnectorMeshLink>(entity).unwrap();
        assert_eq!(link.node_path(), Some("missing/connector"));
        assert_eq!(link.submesh_fallback(), Some("J1_fallback"));
    }

    #[test]
    fn repeated_include_visuals_with_same_local_names_map_to_distinct_stable_objects() {
        let first = c::IncludeInstanceId::root().child("module", 0);
        let second = c::IncludeInstanceId::root().child("module", 1);
        let document = c::ConnectivityDocument {
            document: c::DocumentIdentity::new("memory://hcdviz/repeated-physical").unwrap(),
            scopes: vec![
                c::ConnectivityScope::root(),
                component_scope(first.clone(), "J1", None),
                component_scope(second.clone(), "J1", None),
            ],
        };
        let graph = normalize_connectivity(&document).unwrap();
        let mut ids = Vec::new();
        let mut visuals = Vec::new();
        for instance in [first, second] {
            let resolver = graph.resolver(instance);
            ids.push(
                resolver
                    .connector(&c::ConnectorRef::local_component("base", "J1"))
                    .unwrap()
                    .id()
                    .clone(),
            );
            visuals.push(
                resolver
                    .visual_root(&c::StructuralVisualRef::local("base", "body"))
                    .unwrap()
                    .id()
                    .clone(),
            );
        }
        assert_ne!(ids[0], ids[1]);

        let mut app = app();
        for visual in visuals {
            let (_, holder) = spawn_component_visual(&mut app, visual);
            spawn_named_child(&mut app, holder, "J1");
        }
        publish(&mut app, graph);
        let index = app.world().resource::<CanonicalConnectivitySceneIndex>();
        let first_anchor = index.anchor_entity(&ids[0]).unwrap();
        let second_anchor = index.anchor_entity(&ids[1]).unwrap();
        assert_ne!(first_anchor, second_anchor);
    }

    #[test]
    fn assembly_model_is_a_model_part_root_for_owned_connectors() {
        let mut document = c::ConnectivityDocument::new(
            c::DocumentIdentity::new("memory://hcdviz/assembly-model").unwrap(),
        );
        document.scopes[0].components.push(empty_component("base"));
        document.scopes[0]
            .structural_anchors
            .push(c::StructuralAnchors {
                component: "base".to_owned(),
                visuals: Vec::new(),
                frames: Vec::new(),
            });
        document.scopes[0].assemblies.push(c::PhysicalAssembly {
            name: "harness".to_owned(),
            connectors: vec![c::Connector {
                name: "plug".to_owned(),
                family: None,
                positions: Vec::new(),
                representation: Some(model_part(
                    c::ModelRootRef::AssemblyModel {
                        assembly: c::AssemblyRef::local("harness"),
                    },
                    "plug",
                    None,
                )),
            }],
            kind: c::AssemblyKind::Harness,
            representation: Some(Representation::Model {
                model: c::ModelRepresentation {
                    uri: "assets/harness.glb".to_owned(),
                    sha: None,
                    node_path: None,
                },
                placement: c::Placement {
                    frame: c::RouteFrameRef::ComponentOrigin {
                        component: c::ComponentRef::local("base"),
                    },
                    xyz: [0.0, 0.0, 0.0],
                    rotation: c::PlacementRotation::Rpy([0.0, 0.0, 0.0]),
                },
            }),
            paths: Vec::new(),
            junctions: Vec::new(),
            terminations: Vec::new(),
        });
        let graph = normalize_connectivity(&document).unwrap();
        let resolver = graph.resolver(c::IncludeInstanceId::root());
        let component = resolver
            .component(&c::ComponentRef::local("base"))
            .unwrap()
            .id()
            .clone();
        let assembly = resolver
            .assembly(&c::AssemblyRef::local("harness"))
            .unwrap()
            .id()
            .clone();
        let connector = resolver
            .connector(&c::ConnectorRef {
                owner: c::OwnerRef::local_assembly("harness"),
                connector: "plug".to_owned(),
            })
            .unwrap()
            .id()
            .clone();
        assert_eq!(
            crate::connector::connectivity_host_component_id(&graph, &connector),
            Some(component.clone()),
            "an assembly-owned connector is hosted where the assembly model is placed"
        );

        let mut app = app();
        let component_entity = app
            .world_mut()
            .spawn((
                CompEntity {
                    comp_index: 0,
                    name: "base".to_owned(),
                },
                CanonicalConnectivityAnchor(component),
                Transform::IDENTITY,
            ))
            .id();
        publish(&mut app, graph);
        let model_root = {
            let mut query = app
                .world_mut()
                .query::<(Entity, &CanonicalPhysicalRepresentation)>();
            query
                .iter(app.world())
                .next()
                .map(|(entity, _)| entity)
                .expect("assembly standalone model root")
        };
        let plug = spawn_named_child(&mut app, model_root, "plug");
        app.update();
        let index = app.world().resource::<CanonicalConnectivitySceneIndex>();
        assert!(index.anchor_entity(&assembly).is_some());
        assert!(index.anchor_entity(&connector).is_some());
        let link_entity = index.render_entities(&connector)[0];
        assert_eq!(
            app.world()
                .get::<ConnectorMeshLink>(link_entity)
                .unwrap()
                .exact_root(),
            Some(model_root)
        );

        trigger_primary_click(&mut app, plug);
        app.update();
        assert_eq!(
            app.world().resource::<SelectedConnectivityObject>().0,
            Some(connector),
            "the canonical assembly connector selection must survive interaction arbitration"
        );
        assert_eq!(
            app.world().resource::<Selected>().0,
            Some(component_entity),
            "the assembly placement component hosts the connector selection"
        );
        assert_eq!(
            app.world().resource::<ActiveConnector>().selected(),
            Some(link_entity)
        );
    }
}
