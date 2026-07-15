//! Connectivity presentation boundary.
//!
//! Structural scene construction remains owned by [`crate::scene::ScenePlugin`]. This module owns
//! the accepted canonical connectivity graph, its presentation-entity index, and renderer-local
//! connectivity overlays. Component, joint, visual, collision, sensor, and frame scene construction
//! remains independent of connectivity state.

use std::sync::Arc;

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::doc::HcdfDoc;
use crate::scene::{CompEntity, SceneSet};
use crate::schema::connectivity::{
    ConnectivityNode, NormalizationError, NormalizedConnectivityGraph, ObjectKind, StableObjectId,
};
use crate::schema::document_set::{ConnectivityProjection, ProjectedConnectivityIssue};

/// Ordered lifecycle for document-scoped connectivity presentation state.
///
/// `Clear` removes references to the previous accepted document. `Rebuild` accepts a complete
/// canonical replacement. `Index` maps that state to the structural scene, `Presentation` rebuilds
/// document-scoped overlays, and `Interaction` consumes the completed presentation.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConnectivitySet {
    Clear,
    Rebuild,
    Index,
    Presentation,
    Interaction,
}

/// Identifies the accepted HCDF document generation used to normalize connectivity.
///
/// The value advances in [`ConnectivitySet::Clear`]. Producers must read it after that phase for
/// the document they normalize, then attach the same value to the resulting update.
#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CanonicalConnectivityGeneration(u64);

impl CanonicalConnectivityGeneration {
    pub fn value(self) -> u64 {
        self.0
    }

    fn advance(&mut self) {
        self.0 = self
            .0
            .checked_add(1)
            .expect("canonical connectivity generation exhausted");
    }
}

/// Marks the deterministic anchor entity for one canonical connectivity object.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct CanonicalConnectivityAnchor(pub StableObjectId);

/// Marks one render or highlight entity owned by a canonical connectivity object.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct CanonicalConnectivityRenderOwner(pub StableObjectId);

/// Maps one exact scene node to the canonical connector or position it physically represents.
///
/// The marker lives on a document-scoped child of the represented node rather than on the glTF
/// node itself. Several canonical objects may therefore map to one scene node without overwriting
/// one another, and removing the presentation automatically removes its pick mapping.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct CanonicalConnectivityPickMapping {
    pub object: StableObjectId,
    pub render_entity: Entity,
    pub kind: ObjectKind,
}

/// The currently accepted canonical connectivity graph, or the issues that rejected its replacement.
#[derive(Resource, Default)]
pub struct CanonicalConnectivityState {
    graph: Option<Arc<NormalizedConnectivityGraph>>,
    issues: Vec<ProjectedConnectivityIssue>,
}

impl CanonicalConnectivityState {
    pub fn graph(&self) -> Option<&NormalizedConnectivityGraph> {
        self.graph.as_deref()
    }

    pub fn issues(&self) -> &[ProjectedConnectivityIssue] {
        &self.issues
    }

    pub fn is_ready(&self) -> bool {
        self.graph.is_some()
    }

    fn clear(&mut self) {
        self.graph = None;
        self.issues.clear();
    }
}

/// Transactional replacement of canonical connectivity for one accepted document generation.
#[derive(Message, Debug, Clone)]
pub enum CanonicalConnectivityUpdate {
    Ready {
        generation: CanonicalConnectivityGeneration,
        graph: Arc<NormalizedConnectivityGraph>,
    },
    Invalid {
        generation: CanonicalConnectivityGeneration,
        issues: Vec<ProjectedConnectivityIssue>,
    },
    Clear {
        generation: CanonicalConnectivityGeneration,
    },
}

impl CanonicalConnectivityUpdate {
    pub fn ready(
        generation: CanonicalConnectivityGeneration,
        graph: NormalizedConnectivityGraph,
    ) -> Self {
        Self::Ready {
            generation,
            graph: Arc::new(graph),
        }
    }

    pub fn from_normalization(
        generation: CanonicalConnectivityGeneration,
        result: Result<NormalizedConnectivityGraph, NormalizationError>,
    ) -> Self {
        match result {
            Ok(graph) => Self::ready(generation, graph),
            Err(error) => Self::Invalid {
                generation,
                issues: error
                    .into_issues()
                    .into_iter()
                    .map(ProjectedConnectivityIssue::Normalization)
                    .collect(),
            },
        }
    }

    pub fn from_projection(
        generation: CanonicalConnectivityGeneration,
        projection: ConnectivityProjection,
    ) -> Self {
        match projection {
            ConnectivityProjection::Valid { graph, .. } => Self::ready(generation, graph),
            ConnectivityProjection::Invalid { issues } => Self::Invalid { generation, issues },
        }
    }

    pub fn clear(generation: CanonicalConnectivityGeneration) -> Self {
        Self::Clear { generation }
    }

    fn generation(&self) -> CanonicalConnectivityGeneration {
        match self {
            Self::Ready { generation, .. }
            | Self::Invalid { generation, .. }
            | Self::Clear { generation } => *generation,
        }
    }
}

/// Connectivity selection uses stable IDs and remains independent of structural entity selection.
#[derive(Resource, Default)]
pub struct SelectedConnectivityObject(pub Option<StableObjectId>);

/// Accepted canonical objects mapped to their current presentation entities.
#[derive(Resource, Default, PartialEq, Eq)]
pub struct CanonicalConnectivitySceneIndex {
    anchors: HashMap<StableObjectId, Entity>,
    render_entities: HashMap<StableObjectId, Vec<Entity>>,
    components: HashMap<StableObjectId, Entity>,
}

impl CanonicalConnectivitySceneIndex {
    pub fn anchor_entity(&self, id: &StableObjectId) -> Option<Entity> {
        self.anchors.get(id).copied()
    }

    pub fn render_entities(&self, id: &StableObjectId) -> &[Entity] {
        self.render_entities
            .get(id)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    pub fn component_entity(&self, id: &StableObjectId) -> Option<Entity> {
        self.components.get(id).copied()
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    pub fn render_object_count(&self) -> usize {
        self.render_entities.len()
    }

    pub fn render_entity_count(&self) -> usize {
        self.render_entities.values().map(Vec::len).sum()
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty() && self.render_entities.is_empty() && self.components.is_empty()
    }

    fn clear(&mut self) {
        self.anchors.clear();
        self.render_entities.clear();
        self.components.clear();
    }
}

fn clear_canonical_connectivity_on_document_change(
    mut generation: ResMut<CanonicalConnectivityGeneration>,
    mut state: ResMut<CanonicalConnectivityState>,
    mut index: ResMut<CanonicalConnectivitySceneIndex>,
    mut selected: ResMut<SelectedConnectivityObject>,
) {
    generation.advance();
    state.clear();
    index.clear();
    selected.0 = None;
}

fn apply_canonical_connectivity_updates(
    mut updates: MessageReader<CanonicalConnectivityUpdate>,
    generation: Res<CanonicalConnectivityGeneration>,
    mut state: ResMut<CanonicalConnectivityState>,
    mut index: ResMut<CanonicalConnectivitySceneIndex>,
    mut selected: ResMut<SelectedConnectivityObject>,
) {
    let mut latest = None;
    for update in updates.read() {
        if update.generation() != *generation {
            continue;
        }
        latest = Some(update.clone());
    }
    let Some(latest) = latest else {
        return;
    };

    index.clear();
    selected.0 = None;
    match latest {
        CanonicalConnectivityUpdate::Ready { graph, .. } => {
            state.graph = Some(graph);
            state.issues.clear();
        }
        CanonicalConnectivityUpdate::Invalid { issues, .. } => {
            state.graph = None;
            state.issues = issues;
        }
        CanonicalConnectivityUpdate::Clear { .. } => state.clear(),
    }
}

/// Returns the structural name bridge only for a root-instance component.
///
/// Included instances require an explicit [`CanonicalConnectivityAnchor`] on their [`CompEntity`]
/// so include provenance is retained and repeated local component names cannot collide.
fn root_component_name(node: &ConnectivityNode) -> Option<&str> {
    if node.kind() != ObjectKind::Component || !node.identity().instance().is_root() {
        return None;
    }
    node.identity()
        .local()
        .iter()
        .find(|part| part.field == "component")
        .map(|part| part.value.as_str())
}

pub(crate) fn rebuild_canonical_scene_index(
    state: Res<CanonicalConnectivityState>,
    mut index: ResMut<CanonicalConnectivitySceneIndex>,
    components: Query<(Entity, &CompEntity)>,
    anchor_markers: Query<(Entity, &CanonicalConnectivityAnchor, Option<&CompEntity>)>,
    render_markers: Query<(Entity, &CanonicalConnectivityRenderOwner)>,
) {
    let mut next = CanonicalConnectivitySceneIndex::default();
    let Some(graph) = state.graph() else {
        if *index != next {
            *index = next;
        }
        return;
    };

    let mut anchors = anchor_markers
        .iter()
        .filter_map(|(entity, marker, component)| {
            graph
                .node(&marker.0)
                .map(|node| (marker.0.clone(), entity, component.is_some(), node.kind()))
        })
        .collect::<Vec<_>>();
    anchors.sort_unstable_by(|first, second| {
        first
            .0
            .cmp(&second.0)
            .then_with(|| first.1.index().cmp(&second.1.index()))
    });
    for (id, entity, is_component_entity, kind) in anchors {
        next.anchors.entry(id.clone()).or_insert(entity);
        if kind == ObjectKind::Component && is_component_entity {
            next.components.entry(id).or_insert(entity);
        }
    }

    let mut render_entities = render_markers
        .iter()
        .filter(|(_, marker)| graph.node(&marker.0).is_some())
        .map(|(entity, marker)| (marker.0.clone(), entity))
        .collect::<Vec<_>>();
    render_entities.sort_unstable_by(|first, second| {
        first
            .0
            .cmp(&second.0)
            .then_with(|| first.1.index().cmp(&second.1.index()))
    });
    for (id, entity) in render_entities {
        next.render_entities.entry(id).or_default().push(entity);
    }

    let mut scene_components = components
        .iter()
        .map(|(entity, component)| (component.name.as_str(), component.comp_index, entity))
        .collect::<Vec<_>>();
    scene_components.sort_unstable_by(|first, second| {
        first
            .0
            .cmp(second.0)
            .then_with(|| first.1.cmp(&second.1))
            .then_with(|| first.2.index().cmp(&second.2.index()))
    });
    let mut components_by_name = HashMap::new();
    for (name, _, entity) in scene_components {
        components_by_name.entry(name).or_insert(entity);
    }
    for node in graph.nodes() {
        let Some(name) = root_component_name(node) else {
            continue;
        };
        let Some(entity) = components_by_name.get(name).copied() else {
            continue;
        };
        next.anchors.entry(node.id().clone()).or_insert(entity);
        next.components.entry(node.id().clone()).or_insert(entity);
    }

    if *index != next {
        *index = next;
    }
}

/// Owns the accepted canonical graph and its mapping to current scene entities.
pub struct CanonicalConnectivityPlugin;

impl Plugin for CanonicalConnectivityPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            Update,
            (
                ConnectivitySet::Clear,
                ConnectivitySet::Rebuild,
                ConnectivitySet::Index,
                ConnectivitySet::Presentation,
                ConnectivitySet::Interaction,
            )
                .chain()
                .after(SceneSet::Rebuild),
        )
        .init_resource::<HcdfDoc>()
        .init_resource::<CanonicalConnectivityGeneration>()
        .init_resource::<CanonicalConnectivityState>()
        .init_resource::<CanonicalConnectivitySceneIndex>()
        .init_resource::<SelectedConnectivityObject>()
        .add_message::<CanonicalConnectivityUpdate>()
        .add_systems(
            Update,
            clear_canonical_connectivity_on_document_change
                .in_set(ConnectivitySet::Clear)
                .run_if(resource_changed::<HcdfDoc>),
        )
        .add_systems(
            Update,
            apply_canonical_connectivity_updates.in_set(ConnectivitySet::Rebuild),
        )
        .add_systems(
            Update,
            rebuild_canonical_scene_index.in_set(ConnectivitySet::Index),
        );
    }
}

/// Owns all document-scoped connectivity presentation and interaction systems.
///
/// This plugin intentionally does not own structural scene construction. It is safe to include in
/// both the full viewer and headless scene tests.
pub struct ConnectivityPlugin;

impl Plugin for ConnectivityPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CanonicalConnectivityPlugin>() {
            app.add_plugins(CanonicalConnectivityPlugin);
        }
        app.add_plugins(crate::physical::CanonicalPhysicalPresentationPlugin)
            .add_plugins(crate::connector::ConnectorHighlightPlugin)
            .add_plugins(crate::network::NetworkOverlayPlugin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pick::Selected;
    use crate::schema::connectivity::normalize_connectivity;
    use crate::schema::document_set::{
        load_projected_document_set_from_model, DocumentDependencyDiagnostic, DocumentResourceKey,
        DocumentSetError, DocumentSetLimitKind, DocumentSetOptions, MemoryDocumentResolver,
        ResolverFailure,
    };
    use crate::schema::model::connectivity::{
        ComponentConnectivity, ComponentRef, ConnectivityDocument, Connector, ConnectorRef,
        DocumentIdentity, IncludeInstanceId, StructuralAnchors,
    };
    use crate::schema::model::{Hcdf, StreamProfileResource};

    fn component(name: &str) -> ComponentConnectivity {
        ComponentConnectivity {
            component: name.to_owned(),
            ports: Vec::new(),
            connectors: vec![Connector {
                name: "J1".to_owned(),
                family: None,
                positions: Vec::new(),
                representation: None,
            }],
            antennas: Vec::new(),
            functions: Vec::new(),
            paths: Vec::new(),
            junctions: Vec::new(),
            terminations: Vec::new(),
        }
    }

    fn document() -> ConnectivityDocument {
        let mut document =
            ConnectivityDocument::new(DocumentIdentity::new("memory://hcdviz/canonical").unwrap());
        document.scopes[0].components.push(component("base"));
        document.scopes[0]
            .structural_anchors
            .push(StructuralAnchors {
                component: "base".to_owned(),
                visuals: Vec::new(),
                frames: Vec::new(),
            });
        document
    }

    fn invalid_document() -> ConnectivityDocument {
        let mut document = document();
        document.scopes[0].components.push(component("base"));
        document
    }

    fn test_app() -> App {
        let mut app = App::new();
        app.add_plugins(CanonicalConnectivityPlugin)
            .init_resource::<Selected>();
        app.update();
        app
    }

    fn generation(app: &App) -> CanonicalConnectivityGeneration {
        *app.world().resource::<CanonicalConnectivityGeneration>()
    }

    fn object_ids(graph: &NormalizedConnectivityGraph) -> (StableObjectId, StableObjectId) {
        let resolver = graph.resolver(IncludeInstanceId::root());
        let component = resolver
            .component(&ComponentRef::local("base"))
            .unwrap()
            .id()
            .clone();
        let connector = resolver
            .connector(&ConnectorRef::local_component("base", "J1"))
            .unwrap()
            .id()
            .clone();
        (component, connector)
    }

    fn projected_issue_samples() -> Vec<ProjectedConnectivityIssue> {
        let document_set =
            ProjectedConnectivityIssue::DocumentSet(DocumentSetError::LimitExceeded {
                kind: DocumentSetLimitKind::Depth,
                limit: 2,
                actual: 3,
                resource: Some("memory://nested.hcdf".to_owned()),
            });
        let dependency = ProjectedConnectivityIssue::Dependency {
            instance: IncludeInstanceId::root(),
            include_index: 2,
            diagnostic: DocumentDependencyDiagnostic::ResolutionFailure {
                failure: ResolverFailure::not_found("missing included HCDF"),
            },
        };

        let root = Hcdf {
            name: "root".to_owned(),
            version: "1.0".to_owned(),
            stream_profile: vec![StreamProfileResource {
                uri: "profiles/missing.streams.xml".to_owned(),
                sha: None,
                required: true,
                selection_role: None,
            }],
            ..Default::default()
        };
        let projected = load_projected_document_set_from_model(
            root,
            DocumentIdentity::new("memory://hcdviz/projected-issues").unwrap(),
            DocumentResourceKey::new("/mem/root.hcdf").unwrap(),
            &mut MemoryDocumentResolver::new(),
            DocumentSetOptions::default(),
        )
        .unwrap();
        let stream_profile_dependency = match projected.connectivity() {
            ConnectivityProjection::Invalid { issues } => issues
                .iter()
                .find(|issue| {
                    matches!(
                        issue,
                        ProjectedConnectivityIssue::StreamProfileDependency { .. }
                    )
                })
                .cloned()
                .expect("missing required stream profile must retain its typed diagnostic"),
            ConnectivityProjection::Valid { .. } => {
                panic!("missing required stream profile must invalidate connectivity")
            }
        };

        let unflattened = Hcdf::from_xml_str(
            r#"<hcdf name="root" version="1.0"><include uri="module.hcdf"/></hcdf>"#,
        )
        .unwrap();
        let conversion = ProjectedConnectivityIssue::Conversion {
            instance: IncludeInstanceId::root(),
            error: unflattened
                .to_connectivity_document(
                    DocumentIdentity::new("memory://hcdviz/conversion-issue").unwrap(),
                )
                .unwrap_err(),
        };

        let normalization = ProjectedConnectivityIssue::Normalization(
            normalize_connectivity(&invalid_document())
                .unwrap_err()
                .into_issues()
                .into_iter()
                .next()
                .expect("invalid document must report a normalization issue"),
        );

        vec![
            document_set,
            dependency,
            stream_profile_dependency,
            conversion,
            normalization,
        ]
    }

    #[test]
    fn accepted_graph_indexes_deterministic_anchors_and_all_render_entities() {
        let graph = normalize_connectivity(&document()).unwrap();
        let (component_id, connector_id) = object_ids(&graph);
        let mut app = test_app();
        let generation = generation(&app);
        let component_entity = app
            .world_mut()
            .spawn(CompEntity {
                comp_index: 0,
                name: "base".to_owned(),
            })
            .id();
        let connector_anchor = app
            .world_mut()
            .spawn(CanonicalConnectivityAnchor(connector_id.clone()))
            .id();
        let duplicate_anchor = app
            .world_mut()
            .spawn(CanonicalConnectivityAnchor(connector_id.clone()))
            .id();
        let first_render_entity = app
            .world_mut()
            .spawn(CanonicalConnectivityRenderOwner(connector_id.clone()))
            .id();
        let second_render_entity = app
            .world_mut()
            .spawn(CanonicalConnectivityRenderOwner(connector_id.clone()))
            .id();
        app.world_mut().resource_mut::<Selected>().0 = Some(component_entity);
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(generation, graph));

        app.update();

        let state = app.world().resource::<CanonicalConnectivityState>();
        assert!(state.is_ready());
        assert!(state.issues().is_empty());
        let index = app.world().resource::<CanonicalConnectivitySceneIndex>();
        assert_eq!(
            index.component_entity(&component_id),
            Some(component_entity)
        );
        assert_eq!(index.anchor_entity(&component_id), Some(component_entity));
        assert_eq!(index.anchor_entity(&connector_id), Some(connector_anchor));
        assert_eq!(
            index.render_entities(&connector_id),
            &[first_render_entity, second_render_entity]
        );
        assert_eq!(app.world().resource::<Selected>().0, Some(component_entity));

        let _ = app.world_mut().despawn(connector_anchor);
        let _ = app.world_mut().despawn(first_render_entity);
        app.update();
        let index = app.world().resource::<CanonicalConnectivitySceneIndex>();
        assert_eq!(index.anchor_entity(&connector_id), Some(duplicate_anchor));
        assert_eq!(
            index.render_entities(&connector_id),
            &[second_render_entity]
        );
        assert_eq!(index.anchor_entity(&component_id), Some(component_entity));
        assert_eq!(app.world().resource::<Selected>().0, Some(component_entity));
    }

    #[test]
    fn invalid_replacement_clears_graph_index_and_connectivity_selection_atomically() {
        let graph = normalize_connectivity(&document()).unwrap();
        let (_, connector_id) = object_ids(&graph);
        let mut app = test_app();
        let generation = generation(&app);
        let structural_entity = app
            .world_mut()
            .spawn(CompEntity {
                comp_index: 0,
                name: "base".to_owned(),
            })
            .id();
        app.world_mut()
            .spawn(CanonicalConnectivityAnchor(connector_id.clone()));
        app.world_mut().resource_mut::<Selected>().0 = Some(structural_entity);
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(generation, graph));
        app.update();
        app.world_mut()
            .resource_mut::<SelectedConnectivityObject>()
            .0 = Some(connector_id);

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::from_normalization(
                generation,
                normalize_connectivity(&invalid_document()),
            ));
        app.update();

        let state = app.world().resource::<CanonicalConnectivityState>();
        assert!(!state.is_ready());
        assert!(!state.issues().is_empty());
        assert!(app
            .world()
            .resource::<CanonicalConnectivitySceneIndex>()
            .is_empty());
        assert!(app
            .world()
            .resource::<SelectedConnectivityObject>()
            .0
            .is_none());
        assert_eq!(
            app.world().resource::<Selected>().0,
            Some(structural_entity)
        );
    }

    #[test]
    fn projected_invalid_preserves_every_typed_issue_variant() {
        let expected = projected_issue_samples();
        assert!(expected
            .iter()
            .any(|issue| matches!(issue, ProjectedConnectivityIssue::DocumentSet(_))));
        assert!(expected
            .iter()
            .any(|issue| matches!(issue, ProjectedConnectivityIssue::Dependency { .. })));
        assert!(expected.iter().any(|issue| matches!(
            issue,
            ProjectedConnectivityIssue::StreamProfileDependency { .. }
        )));
        assert!(expected
            .iter()
            .any(|issue| matches!(issue, ProjectedConnectivityIssue::Conversion { .. })));
        assert!(expected
            .iter()
            .any(|issue| matches!(issue, ProjectedConnectivityIssue::Normalization(_))));

        let mut app = test_app();
        let generation = generation(&app);
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::from_projection(
                generation,
                ConnectivityProjection::Invalid {
                    issues: expected.clone(),
                },
            ));
        app.update();

        let state = app.world().resource::<CanonicalConnectivityState>();
        assert!(!state.is_ready());
        assert_eq!(state.issues(), expected);
    }

    #[test]
    fn normalization_failures_are_wrapped_as_projected_issues() {
        let error = normalize_connectivity(&invalid_document()).unwrap_err();
        let expected = error
            .issues()
            .iter()
            .cloned()
            .map(ProjectedConnectivityIssue::Normalization)
            .collect::<Vec<_>>();
        let mut app = test_app();
        let generation = generation(&app);
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::from_normalization(
                generation,
                Err(error),
            ));
        app.update();

        assert_eq!(
            app.world()
                .resource::<CanonicalConnectivityState>()
                .issues(),
            expected
        );
    }

    #[test]
    fn same_generation_later_ready_replaces_invalid_state() {
        let authored = document();
        let graph = normalize_connectivity(&authored).unwrap();
        let mut app = test_app();
        let generation = generation(&app);
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::from_projection(
                generation,
                ConnectivityProjection::Invalid {
                    issues: projected_issue_samples(),
                },
            ));
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::from_projection(
                generation,
                ConnectivityProjection::Valid { authored, graph },
            ));
        app.update();

        let state = app.world().resource::<CanonicalConnectivityState>();
        assert!(state.is_ready());
        assert!(state.issues().is_empty());
    }

    #[test]
    fn stale_invalid_cannot_clear_a_newer_generation() {
        let graph = normalize_connectivity(&document()).unwrap();
        let mut app = test_app();
        let stale_generation = generation(&app);
        app.world_mut().resource_mut::<HcdfDoc>().set_changed();
        app.update();
        let current_generation = generation(&app);
        assert_ne!(current_generation, stale_generation);

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(
                current_generation,
                graph,
            ));
        app.update();
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::from_projection(
                stale_generation,
                ConnectivityProjection::Invalid {
                    issues: projected_issue_samples(),
                },
            ));
        app.update();

        let state = app.world().resource::<CanonicalConnectivityState>();
        assert!(state.is_ready());
        assert!(state.issues().is_empty());
    }

    #[test]
    fn clear_and_document_generation_reject_stale_normalization_results() {
        let graph = normalize_connectivity(&document()).unwrap();
        let (_, connector_id) = object_ids(&graph);
        let mut app = test_app();
        let original_generation = generation(&app);
        app.world_mut()
            .spawn(CanonicalConnectivityAnchor(connector_id));
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(
                original_generation,
                graph.clone(),
            ));
        app.update();
        assert!(app
            .world()
            .resource::<CanonicalConnectivityState>()
            .is_ready());

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::clear(original_generation));
        app.update();
        assert!(!app
            .world()
            .resource::<CanonicalConnectivityState>()
            .is_ready());
        assert!(app
            .world()
            .resource::<CanonicalConnectivitySceneIndex>()
            .is_empty());

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(
                original_generation,
                graph.clone(),
            ));
        app.update();
        assert!(app
            .world()
            .resource::<CanonicalConnectivityState>()
            .is_ready());

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(
                original_generation,
                graph.clone(),
            ));
        app.world_mut().resource_mut::<HcdfDoc>().set_changed();
        app.update();

        let replacement_generation = generation(&app);
        assert_ne!(replacement_generation, original_generation);
        assert!(!app
            .world()
            .resource::<CanonicalConnectivityState>()
            .is_ready());
        assert!(app
            .world()
            .resource::<CanonicalConnectivitySceneIndex>()
            .is_empty());

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(
                replacement_generation,
                graph,
            ));
        app.update();
        assert!(app
            .world()
            .resource::<CanonicalConnectivityState>()
            .is_ready());

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::from_normalization(
                original_generation,
                normalize_connectivity(&invalid_document()),
            ));
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::clear(original_generation));
        app.update();
        let state = app.world().resource::<CanonicalConnectivityState>();
        assert!(state.is_ready());
        assert!(state.issues().is_empty());
    }
}
