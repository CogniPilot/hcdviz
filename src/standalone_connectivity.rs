//! Standalone document-set projection and canonical connectivity publication.
//!
//! The visualization core remains a pure consumer. This opt-in plugin is installed only by the
//! standalone hcdviz binary. Embedders such as dendrite_build keep their own single projection
//! authority and compose [`crate::HcdvizCorePlugin`] without this plugin.

use std::collections::{HashMap, VecDeque};

use bevy::prelude::*;
use hcdformat::document_set::{
    load_projected_document_set_from_bytes, ConnectivityProjection, DocumentResourceKey,
    DocumentSetError, DocumentSetOptions, MemoryDocumentResolver, ProjectedConnectivityDocumentSet,
    StructuralSceneProjection,
};
#[cfg(not(target_arch = "wasm32"))]
use hcdformat::document_set::{
    AssetResourceKey, CallbackDocumentResolver, ResolvedAssetResource, ResolvedDocumentResource,
    ResolverFailure, ResolverFailureKind,
};
use hcdformat::model::connectivity::DocumentIdentity;

use crate::connectivity::{
    CanonicalConnectivityAnchor, CanonicalConnectivityGeneration, CanonicalConnectivityPlugin,
    CanonicalConnectivityUpdate, ConnectivitySet,
};
use crate::doc::LoadHcdf;
use crate::scene::{CompEntity, FrameMarker, OwnerComp, VisualItem};

/// One staged result for the matching [`LoadHcdf`] request.
#[doc(hidden)]
pub enum StagedStandaloneLoad {
    /// The source could not be read, so the normal loader owns the user-facing read error.
    Unavailable,
    /// The resolver-owned document-set projection, including an exact hard failure when one occurred.
    Projection(Box<Result<ProjectedConnectivityDocumentSet, DocumentSetError>>),
}

/// Projection results waiting for the normal document loader's transactional acceptance verdict.
#[derive(Resource, Default)]
#[doc(hidden)]
pub struct StandaloneLoadStaging(VecDeque<StagedStandaloneLoad>);

impl StandaloneLoadStaging {
    pub(crate) fn pop(&mut self) -> Option<StagedStandaloneLoad> {
        self.0.pop_front()
    }
}

/// The projection accepted in the same transaction as the current [`HcdfDoc`].
#[derive(Resource, Default)]
#[doc(hidden)]
pub struct AcceptedStandaloneProjection {
    revision: u64,
    result: Option<Result<ProjectedConnectivityDocumentSet, DocumentSetError>>,
}

impl AcceptedStandaloneProjection {
    pub(crate) fn accept(
        &mut self,
        result: Result<ProjectedConnectivityDocumentSet, DocumentSetError>,
    ) {
        self.revision = self
            .revision
            .checked_add(1)
            .expect("standalone projection revision exhausted");
        self.result = Some(result);
    }
}

/// The accepted standalone projection inputs for the current connectivity generation.
#[derive(Resource, Default)]
pub struct StandaloneConnectivityProjection {
    generation: Option<CanonicalConnectivityGeneration>,
    connectivity: Option<ConnectivityProjection>,
    structural: Option<StructuralSceneProjection>,
}

impl StandaloneConnectivityProjection {
    pub fn generation(&self) -> Option<CanonicalConnectivityGeneration> {
        self.generation
    }

    pub fn connectivity(&self) -> Option<&ConnectivityProjection> {
        self.connectivity.as_ref()
    }

    pub fn structural(&self) -> Option<&StructuralSceneProjection> {
        self.structural.as_ref()
    }
}

/// Projects standalone file-open inputs and publishes their canonical connectivity.
///
/// Install this beside [`crate::HcdvizAppPlugin`] only when hcdviz owns the open flow. The producer
/// is intentionally absent from [`crate::HcdvizCorePlugin`].
pub struct StandaloneConnectivityProducerPlugin;

impl Plugin for StandaloneConnectivityProducerPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CanonicalConnectivityPlugin>() {
            app.add_plugins(CanonicalConnectivityPlugin);
        }
        app.init_resource::<StandaloneLoadStaging>()
            .init_resource::<AcceptedStandaloneProjection>()
            .init_resource::<StandaloneConnectivityProjection>()
            .add_systems(
                Update,
                stage_standalone_loads
                    .after(crate::open::drain_open_channel)
                    .before(crate::doc::load_hcdf_system),
            )
            .add_systems(
                Update,
                publish_accepted_projection
                    .after(ConnectivitySet::Clear)
                    .before(ConnectivitySet::Rebuild),
            )
            .add_systems(
                Update,
                publish_structural_anchors
                    .run_if(structural_anchor_inputs_changed)
                    .after(ConnectivitySet::Rebuild)
                    .before(ConnectivitySet::Index),
            );
    }
}

fn stage_standalone_loads(
    mut loads: MessageReader<LoadHcdf>,
    mut staging: ResMut<StandaloneLoadStaging>,
) {
    for load in loads.read() {
        staging.0.push_back(stage_load(load));
    }
}

fn stage_load(load: &LoadHcdf) -> StagedStandaloneLoad {
    match load {
        LoadHcdf::Path(path) => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                match std::fs::read(path) {
                    Ok(bytes) => StagedStandaloneLoad::Projection(Box::new(project_filesystem(
                        bytes,
                        path.to_string_lossy().into_owned(),
                    ))),
                    Err(_) => StagedStandaloneLoad::Unavailable,
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = path;
                StagedStandaloneLoad::Unavailable
            }
        }
        LoadHcdf::Xml(xml) => StagedStandaloneLoad::Projection(Box::new(project_memory(
            xml.as_bytes().to_vec(),
            "memory://hcdviz/root.hcdf".to_owned(),
            &[],
            &[],
        ))),
        LoadHcdf::Open {
            xml,
            root_key,
            documents,
            assets,
            filesystem_fallback,
        } => {
            #[cfg(not(target_arch = "wasm32"))]
            if *filesystem_fallback {
                return StagedStandaloneLoad::Projection(Box::new(project_filesystem(
                    xml.as_bytes().to_vec(),
                    root_key.clone(),
                )));
            }
            let _ = filesystem_fallback;
            StagedStandaloneLoad::Projection(Box::new(project_memory(
                xml.as_bytes().to_vec(),
                root_key.clone(),
                documents,
                assets,
            )))
        }
    }
}

fn validated_root(
    root_key: String,
) -> Result<(DocumentResourceKey, DocumentIdentity), DocumentSetError> {
    let key = DocumentResourceKey::new(root_key).map_err(|error| {
        DocumentSetError::RootSerialization {
            message: format!("invalid standalone root resource key: {error}"),
        }
    })?;
    let identity = DocumentIdentity::new(key.as_str()).map_err(|error| {
        DocumentSetError::RootSerialization {
            message: format!("invalid standalone document identity: {error}"),
        }
    })?;
    Ok((key, identity))
}

fn project_memory(
    root_bytes: Vec<u8>,
    root_key: String,
    documents: &[(String, Vec<u8>)],
    assets: &[(String, Vec<u8>)],
) -> Result<ProjectedConnectivityDocumentSet, DocumentSetError> {
    let (root_key, identity) = validated_root(root_key)?;
    let mut resolver = MemoryDocumentResolver::new();
    for (key, bytes) in documents {
        resolver
            .insert_document(key.clone(), bytes.clone())
            .map_err(|error| DocumentSetError::RootSerialization {
                message: format!("invalid in-memory document resource key: {error}"),
            })?;
    }
    for (key, bytes) in assets {
        resolver
            .insert_asset(key.clone(), bytes.clone())
            .map_err(|error| DocumentSetError::RootSerialization {
                message: format!("invalid in-memory asset resource key: {error}"),
            })?;
    }
    load_projected_document_set_from_bytes(
        root_bytes,
        identity,
        root_key,
        &mut resolver,
        DocumentSetOptions::default(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn project_filesystem(
    root_bytes: Vec<u8>,
    root_key: String,
) -> Result<ProjectedConnectivityDocumentSet, DocumentSetError> {
    let (root_key, identity) = validated_root(root_key)?;
    let mut resolver = CallbackDocumentResolver::new(
        |parent: &DocumentResourceKey, uri: &str, _kind| {
            let key = resolve_document_key(parent, uri)?;
            let bytes = read_local_resource(&key)?;
            Ok(ResolvedDocumentResource { key, bytes })
        },
        |parent: &DocumentResourceKey, uri: &str| {
            let key = resolve_asset_key(parent, uri)?;
            let bytes = read_local_asset(&key)?;
            Ok(ResolvedAssetResource { key, bytes })
        },
    );
    load_projected_document_set_from_bytes(
        root_bytes,
        identity,
        root_key,
        &mut resolver,
        DocumentSetOptions::default(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn invalid_reference(message: impl Into<String>) -> ResolverFailure {
    ResolverFailure::new(ResolverFailureKind::InvalidReference, message)
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_document_key(
    parent: &DocumentResourceKey,
    uri: &str,
) -> Result<DocumentResourceKey, ResolverFailure> {
    let resolved = hcdformat::resolve_resource_reference(parent.as_str(), uri)
        .map_err(|error| invalid_reference(error.to_string()))?;
    DocumentResourceKey::new(resolved).map_err(|error| invalid_reference(error.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_asset_key(
    parent: &DocumentResourceKey,
    uri: &str,
) -> Result<AssetResourceKey, ResolverFailure> {
    let resolved = hcdformat::resolve_resource_reference(parent.as_str(), uri)
        .map_err(|error| invalid_reference(error.to_string()))?;
    AssetResourceKey::new(resolved).map_err(|error| invalid_reference(error.to_string()))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_local_resource(key: &DocumentResourceKey) -> Result<Vec<u8>, ResolverFailure> {
    reject_non_file_key(key.as_str())?;
    std::fs::read(key.as_str()).map_err(|error| io_failure(key.as_str(), error))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_local_asset(key: &AssetResourceKey) -> Result<Vec<u8>, ResolverFailure> {
    reject_non_file_key(key.as_str())?;
    std::fs::read(key.as_str()).map_err(|error| io_failure(key.as_str(), error))
}

#[cfg(not(target_arch = "wasm32"))]
fn reject_non_file_key(key: &str) -> Result<(), ResolverFailure> {
    if key.contains("://") {
        return Err(ResolverFailure::new(
            ResolverFailureKind::Unsupported,
            format!("standalone document-set resource {key:?} is not a local filesystem path"),
        ));
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn io_failure(key: &str, error: std::io::Error) -> ResolverFailure {
    let kind = match error.kind() {
        std::io::ErrorKind::NotFound => ResolverFailureKind::NotFound,
        std::io::ErrorKind::PermissionDenied => ResolverFailureKind::Denied,
        _ => ResolverFailureKind::Other,
    };
    ResolverFailure::new(kind, format!("could not read {key:?}: {error}"))
}

pub(crate) fn projection_status_suffix(projected: &ProjectedConnectivityDocumentSet) -> String {
    let included = projected.instances().len().saturating_sub(1);
    if included == 0 {
        String::new()
    } else {
        format!(" [+{included} included comp set(s)]")
    }
}

fn publish_accepted_projection(
    accepted: Res<AcceptedStandaloneProjection>,
    generation: Res<CanonicalConnectivityGeneration>,
    mut cache: ResMut<StandaloneConnectivityProjection>,
    mut updates: MessageWriter<CanonicalConnectivityUpdate>,
) {
    if !accepted.is_changed() {
        return;
    }
    let Some(result) = accepted.result.as_ref() else {
        return;
    };
    let (structural, connectivity) = match result {
        Ok(projected) => (
            Some(projected.structural_projection().clone()),
            projected.connectivity().clone(),
        ),
        Err(error) => (None, error.clone().into()),
    };
    updates.write(CanonicalConnectivityUpdate::clear(*generation));
    updates.write(CanonicalConnectivityUpdate::from_projection(
        *generation,
        connectivity.clone(),
    ));
    cache.generation = Some(*generation);
    cache.connectivity = Some(connectivity);
    cache.structural = structural;
}

fn structural_anchor_inputs_changed(
    cache: Res<StandaloneConnectivityProjection>,
    components: Query<(), Added<CompEntity>>,
    visuals: Query<(), Added<VisualItem>>,
    frames: Query<(), Added<FrameMarker>>,
) -> bool {
    cache.is_changed() || !components.is_empty() || !visuals.is_empty() || !frames.is_empty()
}

/// Attaches exact canonical identities to structural entities projected from the same document set.
fn publish_structural_anchors(
    mut commands: Commands,
    cache: Res<StandaloneConnectivityProjection>,
    components: Query<(Entity, &CompEntity)>,
    visuals: Query<(Entity, &OwnerComp, &Name), With<VisualItem>>,
    frames: Query<(Entity, &OwnerComp, &FrameMarker)>,
) {
    let Some(structural) = cache.structural.as_ref() else {
        return;
    };

    let component_entities = components
        .iter()
        .map(|(entity, component)| (component.name.as_str(), entity))
        .collect::<HashMap<_, _>>();
    for component in &structural.components {
        if let Some(entity) = component_entities.get(component.flattened_name.as_str()) {
            commands
                .entity(*entity)
                .insert(CanonicalConnectivityAnchor(component.id.clone()));
        }
    }

    let mut visual_entities = HashMap::<(Entity, String), Vec<Entity>>::new();
    for (entity, owner, name) in &visuals {
        let local_name = name
            .as_str()
            .strip_prefix("visual:")
            .unwrap_or(name.as_str());
        visual_entities
            .entry((owner.0, local_name.to_owned()))
            .or_default()
            .push(entity);
    }
    for entities in visual_entities.values_mut() {
        entities.sort_unstable_by_key(|entity| entity.index());
    }
    let mut visual_occurrences = HashMap::<(Entity, String), usize>::new();
    for visual in &structural.visuals {
        let Some(owner) = component_entities
            .get(visual.flattened_component.as_str())
            .copied()
        else {
            continue;
        };
        let key = (owner, visual.local_name.clone());
        let occurrence = visual_occurrences.entry(key.clone()).or_default();
        if let Some(entity) = visual_entities
            .get(&key)
            .and_then(|entities| entities.get(*occurrence))
        {
            commands
                .entity(*entity)
                .insert(CanonicalConnectivityAnchor(visual.id.clone()));
        }
        *occurrence += 1;
    }

    let mut frame_entities = HashMap::<(Entity, String), Vec<Entity>>::new();
    for (entity, owner, frame) in &frames {
        frame_entities
            .entry((owner.0, frame.label.clone()))
            .or_default()
            .push(entity);
    }
    for entities in frame_entities.values_mut() {
        entities.sort_unstable_by_key(|entity| entity.index());
    }
    let mut frame_occurrences = HashMap::<(Entity, String), usize>::new();
    for frame in &structural.frames {
        let Some(owner) = component_entities
            .get(frame.flattened_component.as_str())
            .copied()
        else {
            continue;
        };
        let key = (owner, frame.local_name.clone());
        let occurrence = frame_occurrences.entry(key.clone()).or_default();
        if let Some(entity) = frame_entities
            .get(&key)
            .and_then(|entities| entities.get(*occurrence))
        {
            commands
                .entity(*entity)
                .insert(CanonicalConnectivityAnchor(frame.id.clone()));
        }
        *occurrence += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::{CanonicalConnectivityState, CanonicalConnectivityUpdate};
    use crate::doc::{load_hcdf_system, HcdfDoc, SchemaStatus};
    use crate::scene::SceneSet;
    use hcdformat::connectivity::ObjectKind;
    use hcdformat::document_set::{DocumentDependencyDiagnostic, ProjectedConnectivityIssue};
    #[cfg(not(target_arch = "wasm32"))]
    use std::sync::atomic::{AtomicU64, Ordering};

    fn producer_app() -> App {
        let mut app = App::new();
        app.init_resource::<HcdfDoc>()
            .init_resource::<SchemaStatus>()
            .add_message::<LoadHcdf>()
            .add_plugins(CanonicalConnectivityPlugin)
            .add_systems(Update, load_hcdf_system.before(SceneSet::Rebuild))
            .add_plugins(StandaloneConnectivityProducerPlugin);
        app
    }

    fn xml_with_component(name: &str) -> String {
        format!(r#"<hcdf name="root" version="1.0"><comp name="{name}"/></hcdf>"#)
    }

    fn graph_has_component(state: &CanonicalConnectivityState, name: &str) -> bool {
        state.graph().is_some_and(|graph| {
            graph.nodes().iter().any(|node| {
                node.kind() == ObjectKind::Component
                    && node
                        .identity()
                        .local()
                        .iter()
                        .any(|part| part.field == "component" && part.value == name)
            })
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn temp_dir(label: &str) -> std::path::PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "hcdviz_standalone_connectivity_{label}_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("create temporary directory");
        path
    }

    #[test]
    fn memory_projection_uses_resolver_owned_dependencies() {
        let root =
            br#"<hcdf name="root" version="1.0"><include name="child" uri="child.hcdf"/></hcdf>"#;
        let child = br#"<hcdf name="child" version="1.0"><comp name="part"/></hcdf>"#;
        let projected = project_memory(
            root.to_vec(),
            "root.hcdf".to_owned(),
            &[("child.hcdf".to_owned(), child.to_vec())],
            &[],
        )
        .expect("project in-memory document set");
        assert_eq!(projected.instances().len(), 2);
        assert!(projected
            .flattened()
            .comp
            .iter()
            .any(|component| component.name == "child/part"));
    }

    #[test]
    fn hard_document_set_error_remains_typed() {
        let root = br#"<hcdf name="root" version="1.0"><include name="same" uri="a.hcdf"/><include name="same" uri="b.hcdf"/></hcdf>"#;
        let error = project_memory(root.to_vec(), "root.hcdf".to_owned(), &[], &[])
            .expect_err("duplicate include name is a hard document-set error");
        assert!(matches!(
            error,
            DocumentSetError::DuplicateIncludeName { .. }
        ));
    }

    #[test]
    fn standalone_load_publishes_a_ready_graph_for_the_exact_generation() {
        let mut app = producer_app();
        app.world_mut()
            .write_message(LoadHcdf::Xml(xml_with_component("base")));

        app.update();

        let generation = *app.world().resource::<CanonicalConnectivityGeneration>();
        let projection = app.world().resource::<StandaloneConnectivityProjection>();
        assert_eq!(projection.generation(), Some(generation));
        assert!(matches!(
            projection.connectivity(),
            Some(ConnectivityProjection::Valid { .. })
        ));
        assert!(graph_has_component(
            app.world().resource::<CanonicalConnectivityState>(),
            "base"
        ));
    }

    #[test]
    fn in_memory_open_resolves_included_documents_without_filesystem_access() {
        let mut app = producer_app();
        app.world_mut().write_message(LoadHcdf::Open {
            xml: r#"<hcdf name="root" version="1.0"><include name="child" uri="modules/child.hcdf"/></hcdf>"#
                .to_owned(),
            root_key: "root.hcdf".to_owned(),
            documents: vec![(
                "modules/child.hcdf".to_owned(),
                xml_with_component("part")
                    .replace("name=\"root\"", "name=\"child\"")
                    .into_bytes(),
            )],
            assets: Vec::new(),
            filesystem_fallback: false,
        });

        app.update();

        assert!(graph_has_component(
            app.world().resource::<CanonicalConnectivityState>(),
            "part"
        ));
        assert!(app
            .world()
            .resource::<HcdfDoc>()
            .0
            .as_ref()
            .unwrap()
            .comp
            .iter()
            .any(|component| component.name == "child/part"));
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_include_uses_projected_flattening_and_exact_structural_anchor() {
        let base = temp_dir("include");
        let root_path = base.join("root.hcdf");
        std::fs::write(
            base.join("child.hcdf"),
            xml_with_component("part").replace("name=\"root\"", "name=\"child\""),
        )
        .expect("write child document");
        std::fs::write(
            &root_path,
            r#"<hcdf name="root" version="1.0"><include name="child" uri="child.hcdf"/></hcdf>"#,
        )
        .expect("write root document");

        let mut app = producer_app();
        let scene_component = app
            .world_mut()
            .spawn(CompEntity {
                comp_index: 0,
                name: "child/part".to_owned(),
            })
            .id();
        app.world_mut().write_message(LoadHcdf::Path(root_path));

        app.update();

        let loaded = app.world().resource::<HcdfDoc>().0.as_ref().unwrap();
        assert!(loaded
            .comp
            .iter()
            .any(|component| component.name == "child/part"));
        let expected = app
            .world()
            .resource::<StandaloneConnectivityProjection>()
            .structural()
            .unwrap()
            .components
            .iter()
            .find(|component| component.flattened_name == "child/part")
            .unwrap()
            .id
            .clone();
        assert_eq!(
            app.world()
                .entity(scene_component)
                .get::<CanonicalConnectivityAnchor>()
                .map(|anchor| anchor.0.clone()),
            Some(expected)
        );
        std::fs::remove_dir_all(base).expect("remove temporary directory");
    }

    #[test]
    fn missing_dependency_publishes_typed_invalid_connectivity() {
        let mut app = producer_app();
        app.world_mut().write_message(LoadHcdf::Xml(
            r#"<hcdf name="root" version="1.0"><include name="missing" uri="missing.hcdf"/></hcdf>"#
                .to_owned(),
        ));

        app.update();

        let state = app.world().resource::<CanonicalConnectivityState>();
        assert!(!state.is_ready());
        assert!(state.issues().iter().any(|issue| {
            matches!(
                issue,
                ProjectedConnectivityIssue::Dependency {
                    diagnostic: DocumentDependencyDiagnostic::ResolutionFailure { failure },
                    ..
                } if failure.kind == hcdformat::document_set::ResolverFailureKind::NotFound
            )
        }));
    }

    #[test]
    fn hard_loader_error_is_published_without_erasing_its_variant() {
        let mut app = producer_app();
        app.world_mut().write_message(LoadHcdf::Xml(
            r#"<hcdf name="root" version="1.0"><include name="same" uri="a.hcdf"/><include name="same" uri="b.hcdf"/></hcdf>"#
                .to_owned(),
        ));

        app.update();

        assert!(matches!(
            app.world()
                .resource::<CanonicalConnectivityState>()
                .issues(),
            [ProjectedConnectivityIssue::DocumentSet(
                DocumentSetError::DuplicateIncludeName { .. }
            )]
        ));
    }

    #[test]
    fn reload_replaces_the_graph_and_rejects_a_stale_generation_update() {
        let mut app = producer_app();
        app.world_mut()
            .write_message(LoadHcdf::Xml(xml_with_component("first")));
        app.update();
        let old_generation = *app.world().resource::<CanonicalConnectivityGeneration>();
        let old_graph = app
            .world()
            .resource::<CanonicalConnectivityState>()
            .graph()
            .unwrap()
            .clone();

        app.world_mut()
            .write_message(LoadHcdf::Xml(xml_with_component("second")));
        app.update();
        let new_generation = *app.world().resource::<CanonicalConnectivityGeneration>();
        assert_ne!(new_generation, old_generation);
        assert!(graph_has_component(
            app.world().resource::<CanonicalConnectivityState>(),
            "second"
        ));
        assert!(!graph_has_component(
            app.world().resource::<CanonicalConnectivityState>(),
            "first"
        ));

        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(
                old_generation,
                old_graph,
            ));
        app.update();
        assert!(graph_has_component(
            app.world().resource::<CanonicalConnectivityState>(),
            "second"
        ));
    }

    #[test]
    fn core_only_embedder_has_no_standalone_projection_writer() {
        let mut app = App::new();
        app.init_resource::<HcdfDoc>()
            .init_resource::<SchemaStatus>()
            .add_message::<LoadHcdf>()
            .add_plugins(CanonicalConnectivityPlugin)
            .add_systems(Update, load_hcdf_system.before(SceneSet::Rebuild));
        app.world_mut()
            .write_message(LoadHcdf::Xml(xml_with_component("embedded")));

        app.update();

        assert!(!app
            .world()
            .contains_resource::<StandaloneConnectivityProjection>());
        assert!(!app
            .world()
            .resource::<CanonicalConnectivityState>()
            .is_ready());
    }
}
