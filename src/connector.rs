//! Mesh-linked connectivity endpoint highlighting.
//!
//! A connectivity endpoint render entity can identify a physical connector inside a visual's GLB.
//! Pointer interaction with that entity selects or previews the exact linked mesh, applying an
//! emissive/base-color tint that is restored on deselect or hover-out.
//!
//! Lookup relies on Bevy's glTF loader naming: every glTF node entity carries `Name` = its node name
//! (`bevy_gltf` `node_name`), with mesh primitives as material-bearing child entities. The search is
//! scoped to the owning comp's spawned visual holders (`Name` = `visual:{name}`), narrowed to
//! the named visual when given, so same-named nodes in another comp/visual can never match. The
//! highlight is a per-entity material swap (clone plus tint, original handle restored), never a
//! mutation of the possibly-shared source material.
use bevy::color::LinearRgba;
use bevy::picking::pointer::PointerButton;
use bevy::prelude::*;

use crate::connectivity::{
    CanonicalConnectivityPickMapping, CanonicalConnectivityRenderOwner,
    CanonicalConnectivitySceneIndex, CanonicalConnectivityState, ConnectivitySet,
    SelectedConnectivityObject,
};
use crate::display::DisplayRegistry;
use crate::doc::HcdfDoc;
use crate::pick::{Selected, SelectionOverrides};
use crate::scene::{OwnerComp, VisualItem, ID_CONNECTIVITY};
use crate::schema::connectivity::{
    structural_component_identity, EdgeKind, NormalizedConnectivityGraph, ObjectKind,
    StableObjectId,
};

/// Link data stamped on a connectivity endpoint render entity: the glTF node to highlight, the
/// visual whose GLB scopes the search, and the endpoint color for the tint.
#[derive(Component, Debug, Clone, PartialEq)]
pub struct ConnectorMeshLink {
    /// The owning component's visual whose subtree is searched (`None` means every visual).
    pub visual: Option<String>,
    /// The glTF node name to find.
    pub mesh: String,
    /// The endpoint's highlight tint.
    pub rgba: [f32; 4],
    /// An exact model root supplied by the canonical physical-presentation producer. `None` keeps
    /// the component-visual lookup used by older endpoint glyph publishers.
    exact_root: Option<Entity>,
    /// The canonical model-part node path, relative to `exact_root`.
    node_path: Option<String>,
    /// The authored fallback node name, considered only when `node_path` does not resolve.
    submesh_fallback: Option<String>,
}

impl ConnectorMeshLink {
    /// A mesh link, or `None` when the mesh name is absent or empty. A visual name alone cannot
    /// identify a node.
    pub fn new(visual: Option<&str>, mesh: Option<&str>, rgba: [f32; 4]) -> Option<Self> {
        let mesh = mesh?.to_string();
        (!mesh.is_empty()).then(|| Self {
            visual: visual.filter(|v| !v.is_empty()).map(str::to_string),
            mesh,
            rgba,
            exact_root: None,
            node_path: None,
            submesh_fallback: None,
        })
    }

    /// Link a canonical physical object to an exact model root and optional model-part selector.
    /// An absent selector targets the complete root. A fallback is used only after a non-empty node
    /// path fails, matching the authored representation contract.
    pub fn exact(
        root: Entity,
        node_path: Option<&str>,
        submesh_fallback: Option<&str>,
        rgba: [f32; 4],
    ) -> Self {
        let node_path = node_path
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let submesh_fallback = submesh_fallback
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        Self {
            visual: None,
            mesh: node_path
                .clone()
                .or_else(|| submesh_fallback.clone())
                .unwrap_or_else(|| "<model-root>".to_owned()),
            rgba,
            exact_root: Some(root),
            node_path,
            submesh_fallback,
        }
    }

    pub fn exact_root(&self) -> Option<Entity> {
        self.exact_root
    }

    pub fn node_path(&self) -> Option<&str> {
        self.node_path.as_deref()
    }

    pub fn submesh_fallback(&self) -> Option<&str> {
        self.submesh_fallback.as_deref()
    }
}

/// The connectivity endpoint render entities the pointer currently interacts with. `hovered`
/// previews over `selected`, which persists until another mesh is clicked, Esc is pressed, or the
/// document reloads. `sync_connector_highlight` tints the hovered object, or every physical render
/// entity associated with the selected logical object when there is no hover.
#[derive(Resource, Default)]
pub struct ActiveConnector {
    pub hovered: Option<Entity>,
    /// Primary selected render entity retained for network lookup.
    selected: Option<Entity>,
    /// Additional physical render entities selected by one logical object, such as every exact pin
    /// bound beneath a multi-channel port.
    selected_related: Vec<Entity>,
}

/// Monotonic notification that exact connector or position geometry was clicked. Unlike the stable
/// selection resource, this changes when an already-selected object is clicked again, allowing an
/// embedder to reveal a collapsed detail section on every direct pick.
#[derive(Resource, Default)]
pub struct ConnectivitySelectionPulse(pub u64);

#[derive(Resource, Default)]
enum PendingConnectivityComponentSelection {
    #[default]
    None,
    DirectPick(Option<Entity>),
}

impl ActiveConnector {
    /// Primary selected render entity, used by integrations that need one representative target.
    pub fn selected(&self) -> Option<Entity> {
        self.selected
    }

    /// Every physical render entity selected by the current logical object.
    pub fn selected_targets(&self) -> impl Iterator<Item = Entity> + '_ {
        self.selected
            .into_iter()
            .chain(self.selected_related.iter().copied())
    }

    /// Select one physical render entity and discard any previous multi-target expansion.
    pub fn set_selected(&mut self, target: Option<Entity>) {
        self.set_selected_targets(target.into_iter().collect());
    }

    /// Replace the complete physical target set. The first target is the primary representative.
    pub fn set_selected_targets(&mut self, targets: Vec<Entity>) {
        self.selected = targets.first().copied();
        self.selected_related = targets.into_iter().skip(1).collect();
    }

    /// Clear the primary and all related physical selections together.
    pub fn clear_selected_targets(&mut self) {
        self.selected = None;
        self.selected_related.clear();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConnectorHighlightKey {
    root: Entity,
    visual: Option<String>,
    mesh: String,
    node_path: Option<String>,
    submesh_fallback: Option<String>,
    rgba_bits: [u32; 4],
}

/// What [`sync_connector_highlight`] last applied, so it can restore swapped materials and skip
/// quiet frames. An empty `originals` collection with non-empty keys means at least one target has
/// no resolvable mesh yet (the glTF may still be loading); the system retries when new meshes spawn.
/// The key includes every field that affects target resolution or tinting, so links scoped to
/// different visuals and links with different colors cannot be mistaken for an unchanged selection.
#[derive(Resource, Default)]
struct ConnectorHighlightState {
    applied_for: Vec<ConnectorHighlightKey>,
    originals: Vec<(Entity, Handle<StandardMaterial>)>,
}

/// Selection and hover of connectivity endpoint entities, plus the material-swap highlight on their
/// linked glTF meshes. Registered by `ConnectivityPlugin`; endpoint annotations shown by the
/// Connectivity display are the interaction surface.
pub struct ConnectorHighlightPlugin;

impl Plugin for ConnectorHighlightPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ActiveConnector>()
            .init_resource::<ConnectivitySelectionPulse>()
            .init_resource::<PendingConnectivityComponentSelection>()
            .init_resource::<ConnectorHighlightState>()
            .init_resource::<CanonicalConnectivitySceneIndex>()
            .init_resource::<CanonicalConnectivityState>()
            .init_resource::<SelectedConnectivityObject>()
            .init_resource::<DisplayRegistry>()
            .init_resource::<SelectionOverrides>()
            .init_resource::<Selected>()
            .init_resource::<HcdfDoc>()
            .init_resource::<ButtonInput<KeyCode>>()
            .add_observer(on_connector_click)
            .add_observer(on_connector_over)
            .add_observer(on_connector_out)
            .add_systems(
                Update,
                reset_connector_state_on_doc_change.in_set(ConnectivitySet::Clear),
            )
            .add_systems(
                Update,
                (
                    clear_connector_selection_on_escape,
                    reset_connector_interaction_on_canonical_change,
                    arbitrate_connectivity_and_structural_selection,
                    sync_stable_connector_selection,
                    // The applied highlight stays correct until an interaction/display input changes
                    // or new meshes spawn (`meshes_added`: the async glTF finishing is what lets an
                    // unresolved link retry), so quiet frames skip entirely.
                    sync_connector_highlight.run_if(
                        resource_changed::<ActiveConnector>
                            .or_else(resource_changed::<DisplayRegistry>)
                            .or_else(resource_changed::<SelectionOverrides>)
                            .or_else(meshes_added),
                    ),
                )
                    .chain()
                    .in_set(ConnectivitySet::Interaction),
            );
    }
}

/// Run-condition input for the retry path: any mesh spawned since the last check (an async glTF
/// subtree materializing).
fn meshes_added(q: Query<(), Added<Mesh3d>>) -> bool {
    !q.is_empty()
}

/// Primary click selects the hit connectivity endpoint entity. A click on any other mesh clears the
/// selection. This resolves from the original hit and writes only on a real
/// change so the sync system's `resource_changed` gate stays meaningful.
#[derive(bevy::ecs::system::SystemParam)]
struct ConnectorPickContext<'w, 's> {
    links: Query<
        'w,
        's,
        (
            &'static ConnectorMeshLink,
            Option<&'static CanonicalConnectivityRenderOwner>,
        ),
    >,
    mappings: Query<'w, 's, &'static CanonicalConnectivityPickMapping>,
    children: Query<'w, 's, &'static Children>,
    parents: Query<'w, 's, &'static ChildOf>,
    state: Res<'w, CanonicalConnectivityState>,
    index: Res<'w, CanonicalConnectivitySceneIndex>,
}

fn on_connector_click(
    click: On<Pointer<Click>>,
    scene: ConnectorPickContext,
    mut active: ResMut<ActiveConnector>,
    mut selected: ResMut<SelectedConnectivityObject>,
    mut pending_component: ResMut<PendingConnectivityComponentSelection>,
    mut pulse: ResMut<ConnectivitySelectionPulse>,
) {
    if click.button != PointerButton::Primary {
        return;
    }
    let hit = click.original_event_target();
    let target = resolve_pointer_target(
        hit,
        &scene.links,
        &scene.mappings,
        &scene.children,
        &scene.parents,
    );
    let render_entity = target.as_ref().map(|target| target.render_entity);
    if active.selected != render_entity || !active.selected_related.is_empty() {
        active.set_selected_targets(render_entity.into_iter().collect());
    }
    let object = target.and_then(|target| target.object);
    let component = object
        .as_ref()
        .and_then(|id| {
            scene
                .state
                .graph()
                .and_then(|graph| connectivity_host_component_id(graph, id))
        })
        .and_then(|component_id| scene.index.component_entity(&component_id));
    if object.is_some() {
        *pending_component = PendingConnectivityComponentSelection::DirectPick(component);
        pulse.0 = pulse.0.wrapping_add(1);
    } else {
        *pending_component = PendingConnectivityComponentSelection::None;
    }
    if selected.0 != object {
        selected.0 = object;
    }
}

/// Hovering a connectivity endpoint entity previews the tint on its linked mesh.
fn on_connector_over(
    over: On<Pointer<Over>>,
    links: Query<(
        &ConnectorMeshLink,
        Option<&CanonicalConnectivityRenderOwner>,
    )>,
    mappings: Query<&CanonicalConnectivityPickMapping>,
    children: Query<&Children>,
    parents: Query<&ChildOf>,
    mut active: ResMut<ActiveConnector>,
) {
    let hit = over.original_event_target();
    let target = resolve_pointer_target(hit, &links, &mappings, &children, &parents)
        .map(|target| target.render_entity);
    if target.is_some() && active.hovered != target {
        active.hovered = target;
    }
}

/// Hover end. Only clears when the leaving entity IS the recorded hover, so a late `Out` after the
/// pointer already moved onto another glyph cannot cancel the new hover.
fn on_connector_out(
    out: On<Pointer<Out>>,
    links: Query<(
        &ConnectorMeshLink,
        Option<&CanonicalConnectivityRenderOwner>,
    )>,
    mappings: Query<&CanonicalConnectivityPickMapping>,
    children: Query<&Children>,
    parents: Query<&ChildOf>,
    mut active: ResMut<ActiveConnector>,
) {
    let hit = out.original_event_target();
    let target = resolve_pointer_target(hit, &links, &mappings, &children, &parents)
        .map(|target| target.render_entity);
    if target.is_some() && active.hovered == target {
        active.hovered = None;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ConnectivityPickTarget {
    object: Option<crate::schema::connectivity::StableObjectId>,
    render_entity: Entity,
}

/// Resolve a pointer hit to the nearest exact semantic model part. glTF ray hits land on primitive
/// descendants, while each mapping is a child of its represented named node. Walking upward and
/// checking each ancestor's direct mapping children therefore chooses a pin before its containing
/// connector shell. Multiple mappings on one node are deterministic, with positions preferred.
fn resolve_pointer_target(
    hit: Entity,
    links: &Query<(
        &ConnectorMeshLink,
        Option<&CanonicalConnectivityRenderOwner>,
    )>,
    mappings: &Query<&CanonicalConnectivityPickMapping>,
    children: &Query<&Children>,
    parents: &Query<&ChildOf>,
) -> Option<ConnectivityPickTarget> {
    let mut current = hit;
    for _ in 0..10_000 {
        if let Ok((_, owner)) = links.get(current) {
            return Some(ConnectivityPickTarget {
                object: owner.map(|owner| owner.0.clone()),
                render_entity: current,
            });
        }

        let mut candidates = children
            .get(current)
            .into_iter()
            .flat_map(|children| children.iter())
            .filter_map(|child| mappings.get(child).ok())
            .collect::<Vec<_>>();
        candidates.sort_by(|first, second| {
            pick_priority(second.kind)
                .cmp(&pick_priority(first.kind))
                .then_with(|| first.object.cmp(&second.object))
                .then_with(|| {
                    first
                        .render_entity
                        .index()
                        .cmp(&second.render_entity.index())
                })
        });
        if let Some(mapping) = candidates.first() {
            return Some(ConnectivityPickTarget {
                object: Some(mapping.object.clone()),
                render_entity: mapping.render_entity,
            });
        }

        let Ok(parent) = parents.get(current) else {
            return None;
        };
        current = parent.parent();
    }
    None
}

fn pick_priority(kind: ObjectKind) -> u8 {
    match kind {
        ObjectKind::Position => 2,
        ObjectKind::Connector => 1,
        _ => 0,
    }
}

/// Esc clears the sticky connector selection (matching `pick::clear_selection_on_escape`); a live
/// hover is left alone; it ends by itself on pointer-out.
fn clear_connector_selection_on_escape(
    keys: Res<ButtonInput<KeyCode>>,
    mut active: ResMut<ActiveConnector>,
    mut selected: ResMut<SelectedConnectivityObject>,
) {
    if keys.just_pressed(KeyCode::Escape) {
        if active.selected.is_some() || !active.selected_related.is_empty() {
            active.clear_selected_targets();
        }
        if selected.0.is_some() {
            selected.0 = None;
        }
    }
}

/// A canonical replacement invalidates every transient render entity, including a hover that has no
/// stable selection counterpart. Clearing only the interaction resource lets the normal highlight
/// sync restore any still-live structural mesh material before presentation entities are replaced.
fn reset_connector_interaction_on_canonical_change(
    state: Res<CanonicalConnectivityState>,
    mut active: ResMut<ActiveConnector>,
) {
    if !state.is_changed() {
        return;
    }
    if active.hovered.is_some() || active.selected.is_some() || !active.selected_related.is_empty()
    {
        *active = ActiveConnector::default();
    }
}

/// Resolve the structural component that should host a canonical connectivity selection.
///
/// Component-owned objects resolve directly. Assembly-owned objects resolve through the owning
/// assembly's primary representation to the component origin or named component frame at which the
/// assembly is placed. World-placed assemblies intentionally have no structural component host.
pub fn connectivity_host_component_id(
    graph: &NormalizedConnectivityGraph,
    id: &StableObjectId,
) -> Option<StableObjectId> {
    let node = graph.node(id)?;
    if let Some(component) = node
        .identity()
        .local()
        .iter()
        .find(|part| part.field == "component")
        .map(|part| part.value.as_str())
    {
        return Some(
            structural_component_identity(
                node.identity().document(),
                node.identity().instance(),
                component,
            )
            .stable_id(),
        );
    }

    let assembly = physical_assembly_owner_id(graph, id)?;
    let representation = graph
        .edges()
        .iter()
        .find(|edge| edge.kind() == EdgeKind::Representation && edge.from() == assembly)?
        .to();
    let frame = graph
        .edges()
        .iter()
        .find(|edge| edge.kind() == EdgeKind::RepresentationFrame && edge.to() == representation)?
        .from();
    let frame_node = graph.node(frame)?;
    match frame_node.kind() {
        ObjectKind::Component => Some(frame.clone()),
        ObjectKind::StructuralFrame => {
            let component = frame_node
                .identity()
                .local()
                .iter()
                .find(|part| part.field == "component")?
                .value
                .as_str();
            Some(
                structural_component_identity(
                    frame_node.identity().document(),
                    frame_node.identity().instance(),
                    component,
                )
                .stable_id(),
            )
        }
        _ => None,
    }
}

fn physical_assembly_owner_id<'a>(
    graph: &'a NormalizedConnectivityGraph,
    id: &StableObjectId,
) -> Option<&'a StableObjectId> {
    let mut current = graph.node(id)?.id();
    for _ in 0..graph.nodes().len() {
        if graph.node(current)?.kind() == ObjectKind::PhysicalAssembly {
            return Some(current);
        }
        current = graph
            .edges()
            .iter()
            .find(|edge| {
                edge.to() == current && matches!(edge.kind(), EdgeKind::Owns | EdgeKind::Contains)
            })?
            .from();
    }
    None
}

fn take_pending_connectivity_pick(
    pending: &mut PendingConnectivityComponentSelection,
) -> Option<Option<Entity>> {
    match std::mem::take(pending) {
        PendingConnectivityComponentSelection::None => None,
        PendingConnectivityComponentSelection::DirectPick(owner) => Some(owner),
    }
}

/// Structural and connectivity selections may change together on an exact model-part click. Keep
/// that paired selection only when the canonical owner maps to the selected component entity. A
/// hierarchy switch changes just the structural selection, so an old pin cannot remain highlighted
/// while another component is being inspected. Included instances remain disjoint through the
/// canonical component ID and scene index.
fn arbitrate_connectivity_and_structural_selection(
    mut structural: ResMut<Selected>,
    state: Res<CanonicalConnectivityState>,
    index: Res<CanonicalConnectivitySceneIndex>,
    mut selected: ResMut<SelectedConnectivityObject>,
    mut pending_component: ResMut<PendingConnectivityComponentSelection>,
) {
    let direct_pick = take_pending_connectivity_pick(&mut pending_component);
    if direct_pick.is_none() && !structural.is_changed() && !selected.is_changed() {
        return;
    }
    let Some(graph) = state.graph() else {
        return;
    };
    let owner = selected
        .0
        .as_ref()
        .and_then(|id| connectivity_host_component_id(graph, id))
        .and_then(|component_id| index.component_entity(&component_id));
    let selected_owner = selected.is_changed().then_some(owner).flatten();
    if let Some(owner) = direct_pick.flatten().or(selected_owner) {
        if structural.0 != Some(owner) {
            structural.0 = Some(owner);
        }
    } else if direct_pick.is_none() && structural.is_changed() && owner != structural.0 {
        selected.0 = None;
    }
}

/// Resolve stable canonical selection to its exact physical render entity. This keeps selection
/// independent of transient Bevy entity IDs while reusing the existing mesh-highlight path.
fn sync_stable_connector_selection(
    selected: Res<SelectedConnectivityObject>,
    state: Res<CanonicalConnectivityState>,
    index: Res<CanonicalConnectivitySceneIndex>,
    links: Query<(), With<ConnectorMeshLink>>,
    mut active: ResMut<ActiveConnector>,
) {
    if !selected.is_changed() && !state.is_changed() && !index.is_changed() {
        return;
    }
    let next = selected.0.as_ref().map_or_else(Vec::new, |id| {
        let physical = state
            .graph()
            .map(|graph| physical_selection_ids(graph, id))
            .unwrap_or_else(|| vec![id.clone()]);
        physical
            .into_iter()
            .flat_map(|physical_id| {
                index
                    .render_entities(&physical_id)
                    .iter()
                    .copied()
                    .filter(|entity| links.contains(*entity))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });
    let primary = next.first().copied();
    let related = next.get(1..).unwrap_or_default();
    if active.selected != primary || active.selected_related.as_slice() != related {
        active.set_selected_targets(next);
    }
}

/// Resolve a logical or physical canonical selection to represented physical objects. Channels
/// follow exact binding edges to positions. Ports prefer all of their channel-position bindings,
/// falling back to a presented whole-port connector binding when no channel has a physical target.
/// Physical objects retain their own identity.
pub fn physical_selection_ids(
    graph: &NormalizedConnectivityGraph,
    selected: &StableObjectId,
) -> Vec<StableObjectId> {
    let Some(node) = graph.node(selected) else {
        return vec![selected.clone()];
    };
    let binding_targets = |functional: &StableObjectId| {
        graph
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Binding && edge.from() == functional)
            .map(|edge| edge.to().clone())
            .collect::<Vec<_>>()
    };
    let mut targets = match node.kind() {
        ObjectKind::Channel => binding_targets(selected),
        ObjectKind::Port => {
            let mut positions = graph
                .edges()
                .iter()
                .filter(|edge| edge.kind() == EdgeKind::Contains && edge.from() == selected)
                .flat_map(|edge| binding_targets(edge.to()))
                .collect::<Vec<_>>();
            if positions.is_empty() {
                positions = binding_targets(selected);
            }
            positions
        }
        ObjectKind::Binding => graph
            .edges()
            .iter()
            .filter(|edge| edge.kind() == EdgeKind::Binding && edge.subject() == selected)
            .map(|edge| edge.to().clone())
            .collect(),
        _ => vec![selected.clone()],
    };
    targets.sort();
    targets.dedup();
    targets
}

/// A doc change despawns the whole scene (glyphs, glTF meshes, and the swapped-out originals with
/// it), so drop every stale entity reference without attempting a restore. Guarded writes keep the
/// change ticks quiet when there is nothing to clear.
fn reset_connector_state_on_doc_change(
    doc: Res<HcdfDoc>,
    mut active: ResMut<ActiveConnector>,
    mut state: ResMut<ConnectorHighlightState>,
) {
    if !doc.is_changed() {
        return;
    }
    if active.hovered.is_some() || active.selected.is_some() || !active.selected_related.is_empty()
    {
        *active = ActiveConnector::default();
    }
    if !state.applied_for.is_empty() || !state.originals.is_empty() {
        *state = ConnectorHighlightState::default();
    }
}

/// Pure glTF-node lookup for one connector link (unit-testable headless; no World/GPU).
///
/// From the owning comp's direct children, take the visual holders (`is_visual_holder`), narrowed to
/// the one named `visual:{visual_scope}` when a scope is given (the spawn-time holder naming). The
/// FIRST descendant named exactly `node_name` (Bevy's glTF loader stamps `Name` = node name, and an
/// ancestor node is always visited before its primitive children) is the connector node; returned
/// are its material-bearing entities: the node itself if it draws, else every drawing descendant
/// (glTF puts primitives on child entities). Empty when the node isn't (yet) in the tree.
pub fn resolve_connector_meshes(
    comp_children: &[Entity],
    visual_scope: Option<&str>,
    node_name: &str,
    is_visual_holder: impl Fn(Entity) -> bool,
    name_of: impl Fn(Entity) -> Option<String>,
    children_of: impl Fn(Entity) -> Vec<Entity>,
    has_material: impl Fn(Entity) -> bool,
) -> Vec<Entity> {
    let scope_name = visual_scope.map(|s| format!("visual:{s}"));
    let mut node = None;
    'holders: for &holder in comp_children {
        if !is_visual_holder(holder)
            || scope_name
                .as_deref()
                .is_some_and(|s| name_of(holder).as_deref() != Some(s))
        {
            continue;
        }
        let mut stack = children_of(holder);
        while let Some(cur) = stack.pop() {
            if name_of(cur).as_deref() == Some(node_name) {
                node = Some(cur);
                break 'holders;
            }
            stack.extend(children_of(cur));
        }
    }
    let Some(node) = node else {
        return Vec::new();
    };
    if has_material(node) {
        return vec![node];
    }
    let mut out = Vec::new();
    let mut stack = children_of(node);
    while let Some(cur) = stack.pop() {
        if has_material(cur) {
            out.push(cur);
        }
        stack.extend(children_of(cur));
    }
    out
}

/// Resolve a canonical model-part selector beneath one exact model root. The path is a sequence of
/// glTF node names separated by `/`; unnamed loader wrapper entities are ignored. If that exact
/// path is absent, `submesh_fallback` selects the first exact node name. No root or component-origin
/// fallback is made for a broken selector.
pub fn resolve_exact_model_meshes(
    root: Entity,
    node_path: Option<&str>,
    submesh_fallback: Option<&str>,
    name_of: impl Fn(Entity) -> Option<String>,
    children_of: impl Fn(Entity) -> Vec<Entity>,
    has_material: impl Fn(Entity) -> bool,
) -> Vec<Entity> {
    let path = node_path
        .map(|value| {
            value
                .split('/')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|parts| !parts.is_empty());

    let target = resolve_exact_model_node(
        root,
        path.as_deref(),
        submesh_fallback,
        &name_of,
        &children_of,
    );
    let Some(target) = target else {
        return Vec::new();
    };
    drawing_entities(target, &children_of, &has_material)
}

/// Resolve only the exact transform entity for a canonical model selector. This is shared by the
/// physical-presentation producer and the material resolver so anchors and highlighting cannot
/// disagree about a node path or fallback.
pub fn resolve_exact_model_node(
    root: Entity,
    node_path: Option<&[String]>,
    submesh_fallback: Option<&str>,
    name_of: &impl Fn(Entity) -> Option<String>,
    children_of: &impl Fn(Entity) -> Vec<Entity>,
) -> Option<Entity> {
    node_path
        .and_then(|parts| resolve_named_path(root, parts, name_of, children_of))
        .or_else(|| {
            submesh_fallback
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .and_then(|name| resolve_named_node(root, name, name_of, children_of))
        })
        .or_else(|| node_path.is_none().then_some(root))
}

fn resolve_named_path(
    root: Entity,
    expected: &[String],
    name_of: &impl Fn(Entity) -> Option<String>,
    children_of: &impl Fn(Entity) -> Vec<Entity>,
) -> Option<Entity> {
    let mut stack = vec![(root, Vec::<String>::new())];
    while let Some((entity, mut named_path)) = stack.pop() {
        if let Some(name) = name_of(entity) {
            named_path.push(name);
            if named_path.ends_with(expected) {
                return Some(entity);
            }
        }
        let children = children_of(entity);
        for child in children.into_iter().rev() {
            stack.push((child, named_path.clone()));
        }
    }
    None
}

fn resolve_named_node(
    root: Entity,
    expected: &str,
    name_of: &impl Fn(Entity) -> Option<String>,
    children_of: &impl Fn(Entity) -> Vec<Entity>,
) -> Option<Entity> {
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        if name_of(entity).as_deref() == Some(expected) {
            return Some(entity);
        }
        let children = children_of(entity);
        stack.extend(children.into_iter().rev());
    }
    None
}

fn drawing_entities(
    root: Entity,
    children_of: &impl Fn(Entity) -> Vec<Entity>,
    has_material: &impl Fn(Entity) -> bool,
) -> Vec<Entity> {
    if has_material(root) {
        return vec![root];
    }
    let mut out = Vec::new();
    let mut stack = children_of(root);
    stack.reverse();
    while let Some(entity) = stack.pop() {
        if has_material(entity) {
            out.push(entity);
        }
        let children = children_of(entity);
        stack.extend(children.into_iter().rev());
    }
    out
}

/// The glTF node names present in a component's spawned visual subtrees. Walks the same holder
/// subtree as [`resolve_connector_meshes`] (the component's visual holders,
/// narrowed to `visual:{visual_scope}` when a scope is given) and collects every DESCENDANT node's
/// `Name`, minus the holder's own synthetic `visual:{...}` name.
///
/// Returned sorted + de-duplicated so the dropdown is stable and free of the repeated primitive-name
/// noise a GLB carries. Empty when the component has no matching holder or its GLB has not finished
/// loading. This is pure and headless-testable, taking the scene edges as closures.
pub fn visual_subtree_node_names(
    comp_children: &[Entity],
    visual_scope: Option<&str>,
    is_visual_holder: impl Fn(Entity) -> bool,
    name_of: impl Fn(Entity) -> Option<String>,
    children_of: impl Fn(Entity) -> Vec<Entity>,
) -> Vec<String> {
    let scope_name = visual_scope.map(|s| format!("visual:{s}"));
    let mut names: Vec<String> = Vec::new();
    for &holder in comp_children {
        if !is_visual_holder(holder)
            || scope_name
                .as_deref()
                .is_some_and(|s| name_of(holder).as_deref() != Some(s))
        {
            continue;
        }
        let mut stack = children_of(holder);
        while let Some(cur) = stack.pop() {
            if let Some(n) = name_of(cur) {
                names.push(n);
            }
            stack.extend(children_of(cur));
        }
    }
    names.sort();
    names.dedup();
    names
}

/// The scene-side queries the highlight needs, grouped so the system stays within the argument
/// budget (mirrors `scene::SelectionBounds`; no lint suppression).
#[derive(bevy::ecs::system::SystemParam)]
struct ConnectorScene<'w, 's> {
    links: Query<'w, 's, (&'static ConnectorMeshLink, &'static OwnerComp)>,
    children: Query<'w, 's, &'static Children>,
    names: Query<'w, 's, &'static Name>,
    holders: Query<'w, 's, (), With<VisualItem>>,
    mats: Query<'w, 's, &'static MeshMaterial3d<StandardMaterial>>,
    /// Meshes spawned since this system last ran: the ONLY signal that lets an unresolved link retry
    /// its lookup (an async glTF subtree finishing). Bundled here (not a separate system arg) to stay
    /// within the clippy argument budget without a lint suppression.
    added: Query<'w, 's, (), Added<Mesh3d>>,
}

/// Keep the linked-mesh tint in sync with the active highlight SOURCE: restore the previous target's
/// original materials, then swap the new target's linked meshes to tinted clones. The Connectivity
/// display toggle controls passive endpoint glyph interaction. An explicit exact model-part selection
/// remains highlighted even when that overlay is off, so Inspector selection always has visible 3D
/// feedback. All entity writes are `try_insert`: on a rebuild frame the restore targets may already be
/// queued for despawn.
fn sync_connector_highlight(
    active: Res<ActiveConnector>,
    registry: Res<DisplayRegistry>,
    overrides: Res<SelectionOverrides>,
    scene: ConnectorScene,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut state: ResMut<ConnectorHighlightState>,
    mut commands: Commands,
) {
    let drawable = registry.enabled(ID_CONNECTIVITY)
        || overrides
            .kinds
            .get(ID_CONNECTIVITY)
            .copied()
            .unwrap_or(false);
    // Hover previews one object. Without a hover, a logical selection may fan out to several exact
    // physical render entities. With the passive overlay hidden, retain only exact model-part links.
    let selected_entities = active
        .selected
        .into_iter()
        .chain(active.selected_related.iter().copied());
    let source_entities = if drawable {
        active
            .hovered
            .map_or_else(|| selected_entities.collect(), |hovered| vec![hovered])
    } else {
        selected_entities.collect()
    };
    let sources = source_entities
        .into_iter()
        .filter_map(|entity| {
            scene
                .links
                .get(entity)
                .ok()
                .map(|(link, owner)| (owner.0, link.clone()))
        })
        .filter(|(_, link)| drawable || link.exact_root.is_some())
        .collect::<Vec<_>>();

    let mut target_ids = sources
        .iter()
        .map(|(owner, link)| ConnectorHighlightKey {
            root: link.exact_root.unwrap_or(*owner),
            visual: link.visual.clone(),
            mesh: link.mesh.clone(),
            node_path: link.node_path.clone(),
            submesh_fallback: link.submesh_fallback.clone(),
            rgba_bits: link.rgba.map(f32::to_bits),
        })
        .collect::<Vec<_>>();
    target_ids.sort_by(|first, second| {
        first
            .root
            .index()
            .cmp(&second.root.index())
            .then_with(|| first.visual.cmp(&second.visual))
            .then_with(|| first.mesh.cmp(&second.mesh))
            .then_with(|| first.node_path.cmp(&second.node_path))
            .then_with(|| first.submesh_fallback.cmp(&second.submesh_fallback))
            .then_with(|| first.rgba_bits.cmp(&second.rgba_bits))
    });
    target_ids.dedup();
    // An unchanged selection only needs another subtree walk when new glTF meshes arrived. This also
    // retries a still-unresolved exact link without spinning on every quiet frame.
    if state.applied_for == target_ids && scene.added.is_empty() {
        return;
    }
    state.applied_for = target_ids;

    let mut desired = Vec::<(Entity, [f32; 4])>::new();
    let mut desired_entities = std::collections::HashSet::new();
    for (owner, link) in &sources {
        let name_of = |entity| {
            scene
                .names
                .get(entity)
                .ok()
                .map(|name| name.as_str().to_owned())
        };
        let children_of = |entity| {
            scene
                .children
                .get(entity)
                .map(|children| children.iter().collect())
                .unwrap_or_default()
        };
        let meshes = if let Some(root) = link.exact_root {
            resolve_exact_model_meshes(
                root,
                link.node_path.as_deref(),
                link.submesh_fallback.as_deref(),
                name_of,
                children_of,
                |entity| scene.mats.contains(entity),
            )
        } else {
            let comp_children = scene
                .children
                .get(*owner)
                .map(|children| children.iter().collect::<Vec<_>>())
                .unwrap_or_default();
            resolve_connector_meshes(
                &comp_children,
                link.visual.as_deref(),
                &link.mesh,
                |entity| scene.holders.contains(entity),
                name_of,
                children_of,
                |entity| scene.mats.contains(entity),
            )
        };
        for mesh in meshes {
            if desired_entities.insert(mesh) {
                desired.push((mesh, link.rgba));
            }
        }
    }

    // Preserve the true source handle for meshes that remain selected while the selection expands,
    // contracts, or retries after async loading. Reading a handle after queuing a restore would still
    // see the tinted clone until command application and would lose the real original.
    let mut originals = std::mem::take(&mut state.originals)
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    for (&entity, original) in &originals {
        if !desired_entities.contains(&entity) {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.try_insert(MeshMaterial3d(original.clone()));
            }
        }
    }
    for (mesh, rgba) in desired {
        let original = originals
            .remove(&mesh)
            .or_else(|| scene.mats.get(mesh).ok().map(|material| material.0.clone()));
        let Some(original) = original else {
            continue;
        };
        let tinted = highlight_material(&mut materials, &original, rgba);
        state.originals.push((mesh, original));
        if let Ok(mut ec) = commands.get_entity(mesh) {
            ec.try_insert(MeshMaterial3d(tinted));
        }
    }
}

/// A tinted CLONE of the mesh's material: base color forced to the connector type color plus a
/// matching emissive glow (legacy dendrite's exact treatment), leaving the original asset (often
/// shared across the whole GLB) untouched. Falls back to a default-based material while the
/// original asset hasn't finished loading.
fn highlight_material(
    materials: &mut Assets<StandardMaterial>,
    original: &Handle<StandardMaterial>,
    rgba: [f32; 4],
) -> Handle<StandardMaterial> {
    let [r, g, b, _] = rgba;
    let mut m = materials.get(original).cloned().unwrap_or_default();
    m.base_color = Color::srgb(r, g, b);
    m.emissive = LinearRgba::new(r * 0.3, g * 0.3, b * 0.3, 1.0);
    materials.add(m)
}

#[cfg(test)]
mod resolve_tests {
    use super::{
        physical_selection_ids, resolve_connector_meshes, take_pending_connectivity_pick,
        PendingConnectivityComponentSelection,
    };
    use crate::schema::connectivity::{normalize_connectivity, ObjectKind};
    use crate::schema::model::connectivity as c;
    use bevy::platform::collections::{HashMap, HashSet};
    use bevy::prelude::Entity;

    #[test]
    fn unresolved_direct_pick_is_distinct_from_no_pending_pick() {
        let mut pending = PendingConnectivityComponentSelection::DirectPick(None);
        assert_eq!(take_pending_connectivity_pick(&mut pending), Some(None));
        assert_eq!(take_pending_connectivity_pick(&mut pending), None);
    }

    /// A synthetic scene tree: names, parent→children edges, and which entities "draw".
    #[derive(Default)]
    struct Tree {
        names: HashMap<Entity, String>,
        children: HashMap<Entity, Vec<Entity>>,
        materials: HashSet<Entity>,
        holders: HashSet<Entity>,
        next: u32,
    }

    impl Tree {
        fn node(&mut self, name: &str) -> Entity {
            let e = Entity::from_raw_u32(self.next).expect("valid test entity index");
            self.next += 1;
            if !name.is_empty() {
                self.names.insert(e, name.to_string());
            }
            e
        }
        fn holder(&mut self, visual_name: &str) -> Entity {
            let e = self.node(&format!("visual:{visual_name}"));
            self.holders.insert(e);
            e
        }
        fn mesh(&mut self, name: &str) -> Entity {
            let e = self.node(name);
            self.materials.insert(e);
            e
        }
        fn parent(&mut self, parent: Entity, child: Entity) {
            self.children.entry(parent).or_default().push(child);
        }
        fn resolve(
            &self,
            comp_children: &[Entity],
            scope: Option<&str>,
            node: &str,
        ) -> Vec<Entity> {
            resolve_connector_meshes(
                comp_children,
                scope,
                node,
                |e| self.holders.contains(&e),
                |e| self.names.get(&e).cloned(),
                |e| self.children.get(&e).cloned().unwrap_or_default(),
                |e| self.materials.contains(&e),
            )
        }
        fn node_names(&self, comp_children: &[Entity], scope: Option<&str>) -> Vec<String> {
            super::visual_subtree_node_names(
                comp_children,
                scope,
                |e| self.holders.contains(&e),
                |e| self.names.get(&e).cloned(),
                |e| self.children.get(&e).cloned().unwrap_or_default(),
            )
        }
    }

    /// Every descendant node name in the scoped holder subtree, sorted and de-duplicated, without
    /// the holder's own synthetic visual name. A wrong scope yields nothing.
    #[test]
    fn visual_subtree_node_names_lists_scoped_gltf_nodes() {
        let mut t = Tree::default();
        let (main, aux) = (t.holder("main"), t.holder("aux"));
        let eth0 = t.node("eth0");
        let prim = t.mesh("eth0_prim");
        let usb0 = t.node("usb0");
        let in_aux = t.node("ant0");
        t.parent(main, eth0);
        t.parent(eth0, prim);
        t.parent(main, usb0);
        t.parent(aux, in_aux);
        let comp_children = [main, aux];

        assert_eq!(
            t.node_names(&comp_children, Some("main")),
            vec![
                "eth0".to_string(),
                "eth0_prim".to_string(),
                "usb0".to_string()
            ],
            "scoped to visual:main, sorted+deduped, holder name excluded"
        );
        assert_eq!(
            t.node_names(&comp_children, Some("aux")),
            vec!["ant0".to_string()]
        );
        assert!(t.node_names(&comp_children, Some("missing")).is_empty());
    }

    /// The b3rb shape: holder `visual:main` → node `eth0` → unnamed-ish primitive children with
    /// materials. The node doesn't draw itself, so its primitives are what get tinted.
    #[test]
    fn finds_node_and_collects_its_primitive_meshes() {
        let mut t = Tree::default();
        let holder = t.holder("main");
        let eth0 = t.node("eth0");
        let prim_a = t.mesh("eth0_mesh.metal");
        let prim_b = t.mesh("eth0_mesh.plastic");
        let other = t.node("usb0");
        let other_prim = t.mesh("usb0_mesh");
        t.parent(holder, eth0);
        t.parent(eth0, prim_a);
        t.parent(eth0, prim_b);
        t.parent(holder, other);
        t.parent(other, other_prim);

        let mut found = t.resolve(&[holder], Some("main"), "eth0");
        found.sort();
        let mut expected = vec![prim_a, prim_b];
        expected.sort();
        assert_eq!(found, expected, "exactly eth0's primitives, never usb0's");
    }

    #[test]
    fn node_that_draws_itself_is_returned_directly() {
        let mut t = Tree::default();
        let holder = t.holder("main");
        let node = t.mesh("ant0"); // node entity carries the material itself
        let child = t.mesh("ant0_sub"); // a drawing child must NOT be double-collected
        t.parent(holder, node);
        t.parent(node, child);
        assert_eq!(t.resolve(&[holder], None, "ant0"), vec![node]);
    }

    /// Two visuals both contain a node named `eth0`; the named visual scope must disambiguate them,
    /// and a wrong scope resolves nothing.
    #[test]
    fn visual_scope_disambiguates_same_named_nodes() {
        let mut t = Tree::default();
        let (main, aux) = (t.holder("main"), t.holder("aux"));
        let (in_main, in_aux) = (t.node("eth0"), t.node("eth0"));
        let (mesh_main, mesh_aux) = (t.mesh("m"), t.mesh("m"));
        t.parent(main, in_main);
        t.parent(in_main, mesh_main);
        t.parent(aux, in_aux);
        t.parent(in_aux, mesh_aux);
        let comp_children = [main, aux];

        assert_eq!(
            t.resolve(&comp_children, Some("aux"), "eth0"),
            vec![mesh_aux]
        );
        assert_eq!(
            t.resolve(&comp_children, Some("main"), "eth0"),
            vec![mesh_main]
        );
        assert!(t
            .resolve(&comp_children, Some("missing"), "eth0")
            .is_empty());
        // No scope → first holder in child order wins (deterministic).
        assert_eq!(t.resolve(&comp_children, None, "eth0"), vec![mesh_main]);
    }

    /// A component model may nest visible terminal nodes beneath connector nodes. Lookup by exact
    /// node name still finds one terminal, while lookup of the connector finds all nested meshes.
    #[test]
    fn nested_pin_nodes_resolve_by_name() {
        let mut t = Tree::default();
        let main = t.holder("main");
        // XT30 power connector → "+"/"−" pins, each a node with its own primitive mesh.
        let xt30 = t.node("xt30");
        let pin_plus = t.node("pin_+");
        let mesh_plus = t.mesh("pin_+_prim");
        let pin_minus = t.node("pin_-");
        let mesh_minus = t.mesh("pin_-_prim");
        // JST-GH-2 CAN connector → canh/canl pins.
        let jst = t.node("jst_gh_2");
        let pin_canh = t.node("pin_canh");
        let mesh_canh = t.mesh("pin_canh_prim");
        t.parent(main, xt30);
        t.parent(xt30, pin_plus);
        t.parent(pin_plus, mesh_plus);
        t.parent(xt30, pin_minus);
        t.parent(pin_minus, mesh_minus);
        t.parent(main, jst);
        t.parent(jst, pin_canh);
        t.parent(pin_canh, mesh_canh);
        let comp_children = [main];

        // Each pin resolves to exactly its own nested mesh.
        assert_eq!(
            t.resolve(&comp_children, Some("main"), "pin_+"),
            vec![mesh_plus]
        );
        assert_eq!(
            t.resolve(&comp_children, Some("main"), "pin_-"),
            vec![mesh_minus]
        );
        assert_eq!(
            t.resolve(&comp_children, Some("main"), "pin_canh"),
            vec![mesh_canh]
        );
        // The connector node itself gathers its nested pin meshes (it draws nothing directly).
        let mut xt30_meshes = t.resolve(&comp_children, Some("main"), "xt30");
        xt30_meshes.sort();
        let mut expected = vec![mesh_plus, mesh_minus];
        expected.sort();
        assert_eq!(
            xt30_meshes, expected,
            "the XT30 node = its two nested pin meshes"
        );
    }

    #[test]
    fn absent_node_and_non_holder_children_resolve_empty() {
        let mut t = Tree::default();
        let holder = t.holder("main");
        let node = t.node("eth0"); // no material anywhere below
        t.parent(holder, node);
        assert!(
            t.resolve(&[holder], None, "nope").is_empty(),
            "unknown node name"
        );
        assert!(
            t.resolve(&[holder], None, "eth0").is_empty(),
            "node with no drawing meshes"
        );
        // A comp child that is NOT a visual holder (e.g. a frame marker) is never searched.
        let marker = t.node("eth0");
        assert!(t.resolve(&[marker], None, "eth0").is_empty());
    }

    #[test]
    fn logical_channel_and_port_selections_resolve_to_bound_positions() {
        let mut document = c::ConnectivityDocument::new(
            c::DocumentIdentity::new("memory://hcdviz/binding-selection").unwrap(),
        );
        document.scopes[0]
            .components
            .push(c::ComponentConnectivity {
                component: "base".to_owned(),
                ports: vec![c::Port {
                    name: "can".to_owned(),
                    capabilities: c::Capabilities::default(),
                    channels: vec![
                        c::Channel {
                            name: "can_h".to_owned(),
                            role: None,
                            local_group: None,
                            capabilities: c::Capabilities::default(),
                        },
                        c::Channel {
                            name: "can_l".to_owned(),
                            role: None,
                            local_group: None,
                            capabilities: c::Capabilities::default(),
                        },
                    ],
                }],
                connectors: vec![c::Connector {
                    name: "J1".to_owned(),
                    family: None,
                    positions: vec![
                        c::Position {
                            name: "1".to_owned(),
                            kind: c::PositionKind::Pin,
                            role: None,
                            local_group: None,
                            representation: None,
                        },
                        c::Position {
                            name: "2".to_owned(),
                            kind: c::PositionKind::Pin,
                            role: None,
                            local_group: None,
                            representation: None,
                        },
                    ],
                    representation: None,
                }],
                antennas: Vec::new(),
                functions: Vec::new(),
                paths: Vec::new(),
                junctions: Vec::new(),
                terminations: Vec::new(),
            });
        document.scopes[0]
            .structural_anchors
            .push(c::StructuralAnchors {
                component: "base".to_owned(),
                visuals: Vec::new(),
                frames: Vec::new(),
            });
        for (channel, position, name) in [("can_h", "1", "high"), ("can_l", "2", "low")] {
            document.scopes[0].bindings.push(c::Binding {
                name: name.to_owned(),
                functional: c::FunctionalEndpointRef::Channel(c::ChannelRef::local(
                    "base", "can", channel,
                )),
                physical: c::PhysicalEndpointRef::Position(c::PositionRef::local_component(
                    "base", "J1", position,
                )),
                fidelity: c::Fidelity::Exact,
            });
        }
        let graph = normalize_connectivity(&document).unwrap();
        let resolver = graph.resolver(c::IncludeInstanceId::root());
        let port = resolver
            .port(&c::PortRef::local("base", "can"))
            .unwrap()
            .id()
            .clone();
        let can_h = resolver
            .channel(&c::ChannelRef::local("base", "can", "can_h"))
            .unwrap()
            .id()
            .clone();
        let pin_1 = resolver
            .position(&c::PositionRef::local_component("base", "J1", "1"))
            .unwrap()
            .id()
            .clone();
        let pin_2 = resolver
            .position(&c::PositionRef::local_component("base", "J1", "2"))
            .unwrap()
            .id()
            .clone();

        assert_eq!(physical_selection_ids(&graph, &can_h), vec![pin_1.clone()]);
        let mut expected = vec![pin_1, pin_2];
        expected.sort();
        assert_eq!(physical_selection_ids(&graph, &port), expected);
        assert_eq!(
            graph.node(&can_h).unwrap().kind(),
            ObjectKind::Channel,
            "fixture must exercise the logical-channel branch"
        );
    }
}
