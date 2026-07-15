//! Canonical logical-network overlay.
//!
//! The renderer consumes only hcdformat's immutable [`NormalizedConnectivityGraph`]. It never parses
//! authored reference strings and never substitutes a component origin for a missing endpoint. An
//! endpoint is drawable only through an exact canonical anchor, an exact canonical render entity, or
//! a synthetic functional-endpoint annotation keyed by [`StableObjectId`]. Physical representations
//! remain separate from these logical topology lines.
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;

use crate::connectivity::{
    CanonicalConnectivitySceneIndex, CanonicalConnectivityState, ConnectivitySet,
};
use crate::connector::ActiveConnector;
use crate::display::Display;
use crate::doc::HcdfDoc;
use crate::pick::{HighlightSet, IsolateSet, SelectionOverrides};
use crate::scene::{kind_drawable, ConnectorMarker, OwnerComp, SceneItem, ID_NETWORK};
use crate::schema::connectivity::{
    ConnectivityNode, ConnectivityNodeData, EdgeKind, NormalizedConnectivityGraph, ObjectKind,
    StableObjectId,
};
use crate::schema::model::connectivity::{Carrier, Topology};

// ── glyph identity ───────────────────────────────────────────────────────────

/// Exact canonical identity of a functional endpoint annotation.
#[derive(Component, Debug, Clone, PartialEq, Eq)]
pub struct ConnectivityEndpointGlyph(pub StableObjectId);

/// Marks an endpoint annotation synthesized because no explicit canonical presentation entity exists.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntheticConnectivityGlyph;

// ── resolved graph model ─────────────────────────────────────────────────────

/// One exact presentation entity used by a logical network segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetEndpoint {
    /// An entity supplied by an exact canonical anchor or render-owner marker.
    Explicit(Entity),
    /// A renderer annotation keyed by the endpoint's exact stable identity.
    Synthetic(Entity),
}

impl NetEndpoint {
    fn entity(self) -> Entity {
        match self {
            Self::Explicit(entity) | Self::Synthetic(entity) => entity,
        }
    }
}

/// How a segment's endpoints connect, driving the edge geometry the draw emits.
#[derive(Debug, Clone, PartialEq)]
pub enum NetTopo {
    Bus,
    Link {
        radiated: bool,
    },
    Chain {
        ring: bool,
        legs: Vec<(usize, usize)>,
    },
    Star {
        radiated: bool,
    },
    Mesh {
        radiated: bool,
    },
    Tree {
        legs: Vec<(usize, usize)>,
    },
}

/// One resolved network segment ready to draw.
#[derive(Debug, Clone)]
pub struct NetSegment {
    /// Stable canonical network identity used for selection, isolation, and visibility.
    pub network: StableObjectId,
    /// Human-readable network name. It is never used as an identity key.
    pub network_label: String,
    pub name: String,
    pub topo: NetTopo,
    pub endpoints: Vec<NetEndpoint>,
    pub color: [f32; 4],
    pub label: String,
    pub members: Vec<Entity>,
}

/// One exact network identity and its display label for the UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkListEntry {
    pub id: StableObjectId,
    pub label: String,
}

/// Renderer-local comms presentation, rebuilt from the current structural scene by
/// [`NetworkOverlayPlugin`]. Empty when the document has no `<network>` or none resolve.
#[derive(Resource, Default)]
pub struct NetworkOverlayScene {
    pub segments: Vec<NetSegment>,
}

impl NetworkOverlayScene {
    /// Distinct networks in canonical graph order.
    pub fn networks(&self) -> Vec<NetworkListEntry> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for s in &self.segments {
            if seen.insert(s.network.clone()) {
                out.push(NetworkListEntry {
                    id: s.network.clone(),
                    label: s.network_label.clone(),
                });
            }
        }
        out
    }

    /// Distinct structural members of one exact network.
    pub fn members_of(&self, net: &StableObjectId) -> Vec<Entity> {
        let mut out: Vec<Entity> = Vec::new();
        for s in self.segments.iter().filter(|s| &s.network == net) {
            for &m in &s.members {
                if !out.contains(&m) {
                    out.push(m);
                }
            }
        }
        out
    }

    /// The first exact network containing an endpoint entity.
    pub fn network_of_glyph(&self, glyph: Entity) -> Option<&StableObjectId> {
        self.segments
            .iter()
            .find(|segment| {
                segment
                    .endpoints
                    .iter()
                    .any(|endpoint| endpoint.entity() == glyph)
            })
            .map(|segment| &segment.network)
    }
}

// ── interaction resources (hooks; standalone UI and authoring panels drive them) ──

/// The network the panel/inspector has selected (like `pick::SelectedJoint`). When `Some`, its member
/// comps join [`HighlightSet`] (green) and its edges brighten; drives the optional network-isolate.
#[derive(Resource, Default)]
pub struct SelectedNetwork(pub Option<StableObjectId>);

/// Isolate-to-network toggle: when ON *and* a network is [`SelectedNetwork`], the scene isolates to that
/// network's member comps (writes [`IsolateSet`]). Reverts to nothing-isolated when turned off.
#[derive(Resource, Default)]
pub struct IsolateNetwork(pub bool);

/// Per-network visibility overrides (the [`crate::pick::SensorVizOverrides`] pattern, keyed by network
/// name). A MISSING entry ⇒ shown (follows the global Networks toggle alone); an entry `false`
/// force-hides exactly that network's edges even while the Networks display is on.
#[derive(Resource, Default)]
pub struct NetworkVizOverrides(pub HashMap<StableObjectId, bool>);

impl NetworkVizOverrides {
    /// Whether `net` is shown per its override, BEFORE the global toggle. Absent ⇒ `true`.
    pub fn visible(&self, net: &StableObjectId) -> bool {
        self.0.get(net).copied().unwrap_or(true)
    }
}

/// The `HighlightSet` entries this module added for [`SelectedNetwork`], so `sync_network_highlight`
/// removes exactly its own entries (never clobbering an embedder's joint-highlight entries).
#[derive(Resource, Default)]
struct NetworkHighlightState(Vec<Entity>);

/// Whether this module currently owns the `IsolateSet` (so it clears it back to `None` only when it was
/// the writer, never stomping an embedder's joint-isolate).
#[derive(Resource, Default)]
struct NetworkIsolateActive(bool);

/// Mutable resources reset together at the accepted-document boundary.
#[derive(bevy::ecs::system::SystemParam)]
struct NetworkLifecycleState<'w> {
    overlay: ResMut<'w, NetworkOverlayScene>,
    selected: ResMut<'w, SelectedNetwork>,
    overrides: ResMut<'w, NetworkVizOverrides>,
    isolate_request: ResMut<'w, IsolateNetwork>,
    highlight: ResMut<'w, HighlightSet>,
    highlight_state: ResMut<'w, NetworkHighlightState>,
    isolate_set: ResMut<'w, IsolateSet>,
    isolate_active: ResMut<'w, NetworkIsolateActive>,
}

/// Clear every document-scoped network presentation reference before rebuilding from a replacement
/// document. Global display preferences remain intact, while name-keyed overrides cannot leak into a
/// different document.
fn clear_network_presentation(mut state: NetworkLifecycleState) {
    state.overlay.segments.clear();
    state.selected.0 = None;
    state.overrides.0.clear();
    state.isolate_request.0 = false;

    if !state.highlight_state.0.is_empty() {
        let owned: HashSet<Entity> = state.highlight_state.0.iter().copied().collect();
        state
            .highlight
            .0
            .retain(|(entity, _)| !owned.contains(entity));
        state.highlight_state.0.clear();
    }

    if state.isolate_active.0 {
        state.isolate_set.0 = None;
        state.isolate_active.0 = false;
    }
}

// ── canonical graph construction ─────────────────────────────────────────────

fn identity_value<'a>(node: &'a ConnectivityNode, field: &str) -> Option<&'a str> {
    node.identity()
        .local()
        .iter()
        .find(|part| part.field == field)
        .map(|part| part.value.as_str())
}

fn participant_endpoint(
    graph: &NormalizedConnectivityGraph,
    participant: &StableObjectId,
) -> Option<StableObjectId> {
    graph
        .edges()
        .iter()
        .find(|edge| edge.kind() == EdgeKind::ParticipantEndpoint && edge.to() == participant)
        .map(|edge| edge.from().clone())
}

fn endpoint_component(
    graph: &NormalizedConnectivityGraph,
    endpoint: &StableObjectId,
) -> Option<StableObjectId> {
    let endpoint_node = graph.node(endpoint)?;
    let port = match endpoint_node.kind() {
        ObjectKind::Port => endpoint.clone(),
        ObjectKind::Channel => graph
            .edges()
            .iter()
            .find(|edge| edge.kind() == EdgeKind::Contains && edge.to() == endpoint)
            .map(|edge| edge.from().clone())?,
        _ => return None,
    };
    graph
        .edges()
        .iter()
        .find(|edge| {
            edge.kind() == EdgeKind::Owns
                && edge.to() == &port
                && graph
                    .node(edge.from())
                    .is_some_and(|node| node.kind() == ObjectKind::Component)
        })
        .map(|edge| edge.from().clone())
}

fn network_members(
    graph: &NormalizedConnectivityGraph,
    network: &ConnectivityNode,
    order: &[String],
) -> Vec<StableObjectId> {
    let mut by_name = graph
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::NetworkMembership && edge.to() == network.id())
        .filter_map(|edge| {
            let participant = graph.node(edge.from())?;
            if participant.kind() != ObjectKind::Participant {
                return None;
            }
            Some((
                identity_value(participant, "participant")?.to_owned(),
                participant.id().clone(),
            ))
        })
        .collect::<HashMap<_, _>>();
    order
        .iter()
        .filter_map(|name| by_name.remove(name))
        .collect()
}

fn topology_nodes(
    graph: &NormalizedConnectivityGraph,
    network: &StableObjectId,
    kind: ObjectKind,
    field: &str,
) -> HashMap<String, StableObjectId> {
    graph
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::TopologyMembership && edge.to() == network)
        .filter_map(|edge| {
            let node = graph.node(edge.from())?;
            if node.kind() != kind {
                return None;
            }
            Some((identity_value(node, field)?.to_owned(), node.id().clone()))
        })
        .collect()
}

fn leg_relation(
    graph: &NormalizedConnectivityGraph,
    leg: &StableObjectId,
    kind: EdgeKind,
) -> Option<StableObjectId> {
    graph.edges().iter().find_map(|edge| match kind {
        EdgeKind::LegFromHop | EdgeKind::LegFromParticipant
            if edge.kind() == kind && edge.to() == leg =>
        {
            Some(edge.from().clone())
        }
        EdgeKind::LegToHop | EdgeKind::LegToParticipant
            if edge.kind() == kind && edge.from() == leg =>
        {
            Some(edge.to().clone())
        }
        _ => None,
    })
}

fn path_legs(
    graph: &NormalizedConnectivityGraph,
    network: &StableObjectId,
    hop_order: &[String],
    closed: bool,
) -> Option<Vec<StableObjectId>> {
    let hops = topology_nodes(graph, network, ObjectKind::Hop, "hop");
    let legs = topology_nodes(graph, network, ObjectKind::Leg, "leg");
    let mut pairs = legs
        .into_values()
        .filter_map(|leg| {
            Some((
                leg_relation(graph, &leg, EdgeKind::LegFromHop)?,
                leg_relation(graph, &leg, EdgeKind::LegToHop)?,
                leg,
            ))
        })
        .collect::<Vec<_>>();
    let mut adjacency = hop_order
        .windows(2)
        .map(|window| Some((hops.get(&window[0])?, hops.get(&window[1])?)))
        .collect::<Option<Vec<_>>>()?;
    if closed {
        adjacency.push((hops.get(hop_order.last()?)?, hops.get(hop_order.first()?)?));
    }
    adjacency
        .into_iter()
        .map(|(from, to)| {
            let index = pairs
                .iter()
                .position(|(leg_from, leg_to, _)| leg_from == from && leg_to == to)?;
            Some(pairs.swap_remove(index).2)
        })
        .collect()
}

fn ordered_participant_endpoints(
    graph: &NormalizedConnectivityGraph,
    participants: &[StableObjectId],
    endpoint_entity: &impl Fn(&StableObjectId) -> Option<NetEndpoint>,
    component_entity: &impl Fn(&StableObjectId) -> Option<Entity>,
) -> Option<(Vec<NetEndpoint>, Vec<Entity>)> {
    let mut endpoints = Vec::with_capacity(participants.len());
    let mut members = Vec::new();
    for participant in participants {
        let endpoint_id = participant_endpoint(graph, participant)?;
        endpoints.push(endpoint_entity(&endpoint_id)?);
        if let Some(member) = endpoint_component(graph, &endpoint_id)
            .as_ref()
            .and_then(component_entity)
        {
            if !members.contains(&member) {
                members.push(member);
            }
        }
    }
    (endpoints.len() >= 2).then_some((endpoints, members))
}

fn profile_label(data: &ConnectivityNodeData) -> String {
    let ConnectivityNodeData::Network { selected, .. } = data else {
        return String::new();
    };
    selected
        .profiles
        .iter()
        .map(|profile| profile.as_str())
        .collect::<Vec<_>>()
        .join(" + ")
}

fn carrier_color(carrier: Carrier) -> [f32; 4] {
    match carrier {
        Carrier::Electrical => [0.95, 0.72, 0.20, 1.0],
        Carrier::GuidedOptical => [0.35, 0.90, 0.80, 1.0],
        Carrier::ConductedRf => [0.90, 0.45, 0.90, 1.0],
        Carrier::RadiatedRf => [0.55, 0.70, 1.0, 1.0],
        Carrier::Liquid => [0.20, 0.55, 0.95, 1.0],
        Carrier::Gas => [0.70, 0.75, 0.82, 1.0],
    }
}

fn graph_network_segment(
    graph: &NormalizedConnectivityGraph,
    network: &ConnectivityNode,
    endpoint_entity: &impl Fn(&StableObjectId) -> Option<NetEndpoint>,
    component_entity: &impl Fn(&StableObjectId) -> Option<Entity>,
) -> Option<NetSegment> {
    let ConnectivityNodeData::Network {
        topology,
        participant_order,
        hop_order,
        leg_order,
        selected,
        ..
    } = network.data()
    else {
        return None;
    };
    let mut participants = network_members(graph, network, participant_order);
    let (endpoints, members, topo) = match topology {
        Topology::Link | Topology::Bus | Topology::Star | Topology::Mesh => {
            if *topology == Topology::Star {
                let coordinator = graph
                    .edges()
                    .iter()
                    .find(|edge| {
                        edge.kind() == EdgeKind::StarCoordinator && edge.to() == network.id()
                    })
                    .map(|edge| edge.from().clone())?;
                let position = participants.iter().position(|id| id == &coordinator)?;
                participants.swap(0, position);
            }
            let (endpoints, members) = ordered_participant_endpoints(
                graph,
                &participants,
                endpoint_entity,
                component_entity,
            )?;
            let topo = match topology {
                Topology::Link => NetTopo::Link {
                    radiated: selected.carrier == Carrier::RadiatedRf,
                },
                Topology::Bus => NetTopo::Bus,
                Topology::Star => NetTopo::Star {
                    radiated: selected.carrier == Carrier::RadiatedRf,
                },
                Topology::Mesh => NetTopo::Mesh {
                    radiated: selected.carrier == Carrier::RadiatedRf,
                },
                _ => unreachable!(),
            };
            (endpoints, members, topo)
        }
        Topology::Chain | Topology::Ring => {
            let closed = *topology == Topology::Ring;
            let legs = path_legs(graph, network.id(), hop_order, closed)?;
            let mut participant_indices = HashMap::<StableObjectId, usize>::new();
            let mut path_participants = Vec::new();
            let mut path_legs = Vec::new();
            for leg in &legs {
                let from = leg_relation(graph, leg, EdgeKind::LegFromParticipant)?;
                let to = leg_relation(graph, leg, EdgeKind::LegToParticipant)?;
                let from_index = *participant_indices.entry(from.clone()).or_insert_with(|| {
                    let index = path_participants.len();
                    path_participants.push(from);
                    index
                });
                let to_index = *participant_indices.entry(to.clone()).or_insert_with(|| {
                    let index = path_participants.len();
                    path_participants.push(to);
                    index
                });
                path_legs.push((from_index, to_index));
            }
            let (endpoints, members) = ordered_participant_endpoints(
                graph,
                &path_participants,
                endpoint_entity,
                component_entity,
            )?;
            (
                endpoints,
                members,
                NetTopo::Chain {
                    ring: closed,
                    legs: path_legs,
                },
            )
        }
        Topology::Tree => {
            let legs_by_name = topology_nodes(graph, network.id(), ObjectKind::Leg, "leg");
            let mut participant_indices = HashMap::<StableObjectId, usize>::new();
            let mut tree_participants = Vec::new();
            let mut tree_legs = Vec::new();
            for name in leg_order {
                let leg = legs_by_name.get(name)?;
                leg_relation(graph, leg, EdgeKind::LegFromHop)?;
                leg_relation(graph, leg, EdgeKind::LegToHop)?;
                let from = leg_relation(graph, leg, EdgeKind::LegFromParticipant)?;
                let to = leg_relation(graph, leg, EdgeKind::LegToParticipant)?;
                let from_index = *participant_indices.entry(from.clone()).or_insert_with(|| {
                    let index = tree_participants.len();
                    tree_participants.push(from);
                    index
                });
                let to_index = *participant_indices.entry(to.clone()).or_insert_with(|| {
                    let index = tree_participants.len();
                    tree_participants.push(to);
                    index
                });
                tree_legs.push((from_index, to_index));
            }
            let (endpoints, members) = ordered_participant_endpoints(
                graph,
                &tree_participants,
                endpoint_entity,
                component_entity,
            )?;
            (endpoints, members, NetTopo::Tree { legs: tree_legs })
        }
    };
    let name = identity_value(network, "network")?.to_owned();
    let network_label = if network.identity().instance().is_root() {
        name.clone()
    } else {
        format!("{name} ({})", network.identity().instance().display_path())
    };
    Some(NetSegment {
        network: network.id().clone(),
        network_label,
        name,
        topo,
        endpoints,
        color: carrier_color(selected.carrier),
        label: profile_label(network.data()),
        members,
    })
}

/// Build logical topology segments from exact canonical identities and typed edges.
pub fn build_network_overlay(
    graph: &NormalizedConnectivityGraph,
    endpoint_entity: impl Fn(&StableObjectId) -> Option<NetEndpoint>,
    component_entity: impl Fn(&StableObjectId) -> Option<Entity>,
) -> Vec<NetSegment> {
    graph
        .nodes()
        .iter()
        .filter(|node| node.kind() == ObjectKind::Network)
        .filter_map(|network| {
            graph_network_segment(graph, network, &endpoint_entity, &component_entity)
        })
        .collect()
}

fn participant_endpoints(graph: &NormalizedConnectivityGraph) -> Vec<StableObjectId> {
    let mut endpoints = graph
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::ParticipantEndpoint)
        .map(|edge| edge.from().clone())
        .collect::<Vec<_>>();
    endpoints.sort();
    endpoints.dedup();
    endpoints
}

fn synthetic_endpoint_offset(index: usize, count: usize) -> Vec3 {
    let radius = (0.012 * count.max(2) as f32 / std::f32::consts::TAU).max(0.018);
    let angle = std::f32::consts::TAU * index as f32 / count.max(1) as f32;
    Vec3::new(
        radius * angle.cos(),
        0.004 + index as f32 * 0.0015,
        radius * angle.sin(),
    )
}

fn rebuild_network_presentation(
    mut commands: Commands,
    state: Res<CanonicalConnectivityState>,
    index: Res<CanonicalConnectivitySceneIndex>,
    stale_synthetic: Query<Entity, With<SyntheticConnectivityGlyph>>,
    mut overlay: ResMut<NetworkOverlayScene>,
) {
    let stale = stale_synthetic.iter().collect::<HashSet<_>>();
    for entity in &stale {
        commands.entity(*entity).despawn();
    }
    let Some(graph) = state.graph() else {
        overlay.segments.clear();
        return;
    };

    let endpoint_ids = participant_endpoints(graph);
    let mut endpoint_entities = HashMap::<StableObjectId, NetEndpoint>::new();
    let mut pending = HashMap::<StableObjectId, Vec<StableObjectId>>::new();
    for endpoint in endpoint_ids {
        let exact = index
            .anchor_entity(&endpoint)
            .filter(|entity| !stale.contains(entity))
            .or_else(|| {
                index
                    .render_entities(&endpoint)
                    .iter()
                    .copied()
                    .find(|entity| !stale.contains(entity))
            });
        if let Some(entity) = exact {
            endpoint_entities.insert(endpoint, NetEndpoint::Explicit(entity));
        } else if let Some(component) = endpoint_component(graph, &endpoint) {
            pending.entry(component).or_default().push(endpoint);
        }
    }
    for (component, mut endpoints) in pending {
        let Some(owner) = index.component_entity(&component) else {
            continue;
        };
        endpoints.sort();
        let count = endpoints.len();
        for (position, endpoint) in endpoints.into_iter().enumerate() {
            let entity = commands
                .spawn((
                    SceneItem,
                    ConnectorMarker,
                    ConnectivityEndpointGlyph(endpoint.clone()),
                    SyntheticConnectivityGlyph,
                    OwnerComp(owner),
                    Transform::from_translation(synthetic_endpoint_offset(position, count)),
                    Visibility::Hidden,
                    Name::new(format!(
                        "synthetic-connectivity-endpoint:{}",
                        endpoint.as_str()
                    )),
                ))
                .id();
            commands.entity(owner).add_child(entity);
            endpoint_entities.insert(endpoint, NetEndpoint::Synthetic(entity));
        }
    }
    overlay.segments = build_network_overlay(
        graph,
        |endpoint| endpoint_entities.get(endpoint).copied(),
        |component| index.component_entity(component),
    );
}

// ── drawing ──────────────────────────────────────────────────────────────────

/// World-space height of a network label, metres (matches the frame/sensor label size).
const NET_LABEL_SIZE: f32 = 0.02;
/// Base arc lift (metres) added per network index so overlapping networks don't z-fight the kinematics
/// lines or each other.
const NET_LIFT_BASE: f32 = 0.012;
const NET_LIFT_STEP: f32 = 0.006;

/// Exact canonical endpoint transforms used by the logical overlay.
#[derive(bevy::ecs::system::SystemParam)]
struct NetScene<'w, 's> {
    transforms: Query<'w, 's, &'static GlobalTransform>,
}

impl NetScene<'_, '_> {
    /// World position of the exact explicit or synthetic endpoint entity.
    fn endpoint_pos(&self, ep: NetEndpoint) -> Option<Vec3> {
        self.transforms
            .get(ep.entity())
            .ok()
            .map(|transform| transform.translation())
    }
}

/// Draw the network overlay: protocol-colored edges + labels per resolved segment. Immediate-mode, so
/// gated by `kind_drawable(ID_NETWORK)` (runs only when the Networks display can show at all); per-network
/// visibility overrides and the isolate/selection state are applied inside.
fn draw_network_edges(
    graph: Res<NetworkOverlayScene>,
    overrides: Res<NetworkVizOverrides>,
    selected: Res<SelectedNetwork>,
    net_isolate: Res<IsolateNetwork>,
    scene: NetScene,
    mut gizmos: Gizmos,
) {
    // Precompute the per-network lift so overlapping networks separate.
    let networks = graph.networks();
    for seg in &graph.segments {
        if !overrides.visible(&seg.network) {
            continue;
        }
        let net_idx = networks
            .iter()
            .position(|network| network.id == seg.network)
            .unwrap_or(0);
        let lift = NET_LIFT_BASE + net_idx as f32 * NET_LIFT_STEP;
        let is_selected = selected.0.as_ref() == Some(&seg.network);
        // Isolate-to-network dims non-selected networks to near-nothing (skip drawing them).
        if net_isolate.0 && selected.0.is_some() && !is_selected {
            continue;
        }
        let color = to_color(seg.color, is_selected);

        // Resolve every endpoint world position once; a None endpoint drops just that edge.
        let pts: Vec<Option<Vec3>> = seg
            .endpoints
            .iter()
            .map(|&e| scene.endpoint_pos(e))
            .collect();
        for (endpoint, point) in seg.endpoints.iter().zip(&pts) {
            if matches!(endpoint, NetEndpoint::Synthetic(_)) {
                if let Some(point) = point {
                    draw_synthetic_endpoint(&mut gizmos, *point, color);
                }
            }
        }
        draw_topo(&mut gizmos, &seg.topo, &pts, color, lift);

        // Label at the centroid of the resolved points.
        draw_label(&mut gizmos, seg, &pts, color);
    }
}

/// sRGBA(+brighten) → Bevy `Color`. Selected segments lerp toward white for emphasis.
fn to_color(rgba: [f32; 4], selected: bool) -> Color {
    let f = if selected { 0.5 } else { 0.0 };
    let l = |c: f32| c + (1.0 - c) * f;
    Color::srgb(l(rgba[0]), l(rgba[1]), l(rgba[2]))
}

/// Emit the edge geometry for one topology from its resolved endpoint positions.
fn draw_topo(gizmos: &mut Gizmos, topo: &NetTopo, pts: &[Option<Vec3>], color: Color, lift: f32) {
    match topo {
        NetTopo::Bus => {
            for w in pts.windows(2) {
                if let (Some(a), Some(b)) = (w[0], w[1]) {
                    draw_arc(gizmos, a, b, lift, color, false);
                }
            }
        }
        NetTopo::Chain { legs, .. } => {
            for &(from, to) in legs {
                if let (Some(Some(a)), Some(Some(b))) = (pts.get(from), pts.get(to)) {
                    draw_arc(gizmos, *a, *b, lift, color, false);
                }
            }
        }
        NetTopo::Mesh { radiated } => {
            for (from, to) in complete_mesh_pairs(pts.len()) {
                if let (Some(a), Some(b)) = (pts[from], pts[to]) {
                    draw_arc(gizmos, a, b, lift, color, *radiated);
                }
            }
        }
        NetTopo::Star { radiated } => {
            if let Some(Some(hub)) = pts.first() {
                for spoke in pts.iter().skip(1).flatten() {
                    draw_arc(gizmos, *hub, *spoke, lift, color, *radiated);
                }
            }
        }
        NetTopo::Tree { legs } => {
            for &(from, to) in legs {
                if let (Some(Some(a)), Some(Some(b))) = (pts.get(from), pts.get(to)) {
                    draw_arc(gizmos, *a, *b, lift, color, false);
                }
            }
        }
        NetTopo::Link { radiated } => {
            if let (Some(Some(a)), Some(Some(b))) = (pts.first(), pts.get(1)) {
                draw_arc(gizmos, *a, *b, lift, color, *radiated);
            }
        }
    }
}

fn complete_mesh_pairs(count: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..count).flat_map(move |from| ((from + 1)..count).map(move |to| (from, to)))
}

/// Draw a lifted quadratic-bezier arc from `a` to `b` (control point raised by `lift` along +Y). When
/// `dashed`, only every other sub-segment is drawn.
fn draw_arc(gizmos: &mut Gizmos, a: Vec3, b: Vec3, lift: f32, color: Color, dashed: bool) {
    const SEG: usize = 14;
    let ctrl = (a + b) * 0.5 + Vec3::Y * lift;
    let mut prev = a;
    for i in 1..=SEG {
        let t = i as f32 / SEG as f32;
        let u = 1.0 - t;
        let p = a * (u * u) + ctrl * (2.0 * u * t) + b * (t * t);
        if !dashed || i % 2 == 1 {
            gizmos.line(prev, p, color);
        }
        prev = p;
    }
}

/// A small three-axis cross distinguishes a synthetic endpoint annotation from physical geometry.
fn draw_synthetic_endpoint(gizmos: &mut Gizmos, point: Vec3, color: Color) {
    const HALF: f32 = 0.004;
    gizmos.line(point - Vec3::X * HALF, point + Vec3::X * HALF, color);
    gizmos.line(point - Vec3::Y * HALF, point + Vec3::Y * HALF, color);
    gizmos.line(point - Vec3::Z * HALF, point + Vec3::Z * HALF, color);
}

/// Draw the segment's name + terse param line at the centroid of its resolved endpoints.
fn draw_label(gizmos: &mut Gizmos, seg: &NetSegment, pts: &[Option<Vec3>], color: Color) {
    let mut sum = Vec3::ZERO;
    let mut n = 0.0f32;
    for p in pts.iter().flatten() {
        sum += *p;
        n += 1.0;
    }
    if n == 0.0 {
        return;
    }
    let mid = sum / n + Vec3::Y * (NET_LABEL_SIZE * 0.5);
    gizmos.text(
        Isometry3d::from_translation(mid),
        &seg.name,
        NET_LABEL_SIZE,
        Vec2::ZERO,
        color,
    );
    if !seg.label.is_empty() {
        gizmos.text(
            Isometry3d::from_translation(mid - Vec3::Y * (NET_LABEL_SIZE * 1.4)),
            &seg.label,
            NET_LABEL_SIZE * 0.9,
            Vec2::ZERO,
            Color::srgb(0.82, 0.82, 0.88),
        );
    }
}

// ── interaction systems ───────────────────────────────────────────────────────

/// Green highlight color for a selected network's member comps (distinct from joint orange/cyan).
const NETWORK_HIGHLIGHT: Color = Color::srgb(0.2, 0.9, 0.35);

/// Keep [`HighlightSet`] in sync with [`SelectedNetwork`]: remove this module's previously-added entries,
/// then add the selected network's member comps (green). Only ever touches its own entries, so an
/// embedder's joint-highlight entries survive.
fn sync_network_highlight(
    selected: Res<SelectedNetwork>,
    graph: Res<NetworkOverlayScene>,
    mut highlight: ResMut<HighlightSet>,
    mut state: ResMut<NetworkHighlightState>,
) {
    // Drop our old entries.
    if !state.0.is_empty() {
        let owned: HashSet<Entity> = state.0.iter().copied().collect();
        highlight.0.retain(|(e, _)| !owned.contains(e));
        state.0.clear();
    }
    if let Some(net) = &selected.0 {
        for e in graph.members_of(net) {
            highlight.0.push((e, NETWORK_HIGHLIGHT));
            state.0.push(e);
        }
    }
}

/// Isolate the scene to the selected network's member comps when [`IsolateNetwork`] is ON. Writes
/// [`IsolateSet`] only while this module owns it (tracked in [`NetworkIsolateActive`]), so turning the
/// toggle off restores `None` without stomping an embedder's own isolate.
fn sync_network_isolate(
    net_isolate: Res<IsolateNetwork>,
    selected: Res<SelectedNetwork>,
    graph: Res<NetworkOverlayScene>,
    mut iso: ResMut<IsolateSet>,
    mut active: ResMut<NetworkIsolateActive>,
) {
    let want = net_isolate.0 && selected.0.is_some();
    if want {
        if let Some(net) = &selected.0 {
            iso.0 = Some(graph.members_of(net).into_iter().collect());
            active.0 = true;
        }
    } else if active.0 {
        iso.0 = None;
        active.0 = false;
    }
}

/// Clicking a connectivity endpoint entity selects its first containing network, reusing the
/// `ActiveConnector` selection the Connectivity display already drives. A click that resolves to no
/// network leaves the selection untouched.
fn select_network_on_glyph_click(
    active: Res<ActiveConnector>,
    graph: Res<NetworkOverlayScene>,
    mut selected: ResMut<SelectedNetwork>,
) {
    let Some(glyph) = active.selected() else {
        return;
    };
    if let Some(net) = graph.network_of_glyph(glyph) {
        if selected.0.as_ref() != Some(net) {
            selected.0 = Some(net.clone());
        }
    }
}

// ── lifecycle + display plugins ──────────────────────────────────────────────

/// Core renderer-local network presentation lifecycle.
pub(crate) struct NetworkOverlayPlugin;

impl Plugin for NetworkOverlayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<NetworkOverlayScene>()
            .init_resource::<NetworkVizOverrides>()
            .init_resource::<SelectedNetwork>()
            .init_resource::<IsolateNetwork>()
            .init_resource::<NetworkHighlightState>()
            .init_resource::<NetworkIsolateActive>()
            .init_resource::<HighlightSet>()
            .init_resource::<IsolateSet>()
            .init_resource::<ActiveConnector>()
            .init_resource::<HcdfDoc>()
            .init_resource::<SelectionOverrides>()
            .add_systems(
                Update,
                (clear_network_presentation, rebuild_network_presentation)
                    .chain()
                    .in_set(ConnectivitySet::Presentation)
                    .run_if(
                        resource_changed::<CanonicalConnectivityState>
                            .or_else(resource_changed::<CanonicalConnectivitySceneIndex>),
                    ),
            );
    }
}

/// NetworksDisplay: the spatial comms overlay. Default OFF, following the Connectivity display (this
/// overlay is the same annotation family and would clutter the default view otherwise).
pub struct NetworksDisplay;
impl Display for NetworksDisplay {
    fn id(&self) -> &'static str {
        ID_NETWORK
    }
    fn label(&self) -> &str {
        "Networks (comms overlay)"
    }
    fn default_enabled(&self) -> bool {
        false
    }
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<crate::connectivity::ConnectivityPlugin>() {
            app.add_plugins(crate::connectivity::ConnectivityPlugin);
        }
        app.add_systems(
            Update,
            draw_network_edges
                .run_if(kind_drawable(self.id()))
                .in_set(ConnectivitySet::Interaction),
        )
        .add_systems(
            Update,
            (
                select_network_on_glyph_click.run_if(resource_changed::<ActiveConnector>),
                sync_network_highlight.run_if(
                    resource_changed::<SelectedNetwork>
                        .or_else(resource_changed::<NetworkOverlayScene>),
                ),
                sync_network_isolate.run_if(
                    resource_changed::<IsolateNetwork>
                        .or_else(resource_changed::<SelectedNetwork>)
                        .or_else(resource_changed::<NetworkOverlayScene>),
                ),
            )
                .chain()
                .in_set(ConnectivitySet::Interaction),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::connectivity::{
        CanonicalConnectivityAnchor, CanonicalConnectivityGeneration, CanonicalConnectivityPlugin,
        CanonicalConnectivityUpdate,
    };
    use crate::scene::CompEntity;
    use crate::schema::connectivity::normalize_connectivity;
    use crate::schema::model::connectivity::{
        Capabilities, ComponentConnectivity, ComponentRef, ConnectivityDocument, DocumentIdentity,
        FunctionalEndpointRef, IncludeInstanceId, Network, NetworkConfiguration, NetworkSelection,
        NetworkStructure, Participant, Port, PortRef, Purpose, StructuralAnchors,
    };

    fn canonical_link() -> NormalizedConnectivityGraph {
        let capabilities = Capabilities {
            purposes: BTreeSet::from([Purpose::Communication]),
            carriers: BTreeSet::from([Carrier::Electrical]),
            profiles: BTreeSet::new(),
            limits: Default::default(),
        };
        let component = |name: &str| ComponentConnectivity {
            component: name.to_owned(),
            ports: vec![Port {
                name: "data".to_owned(),
                capabilities: capabilities.clone(),
                channels: Vec::new(),
            }],
            connectors: Vec::new(),
            antennas: Vec::new(),
            functions: Vec::new(),
            paths: Vec::new(),
            junctions: Vec::new(),
            terminations: Vec::new(),
        };
        let mut document = ConnectivityDocument::new(
            DocumentIdentity::new("memory://hcdviz/synthetic-endpoints").unwrap(),
        );
        document.scopes[0].components = vec![component("a"), component("b")];
        document.scopes[0].structural_anchors = ["a", "b"]
            .into_iter()
            .map(|name| StructuralAnchors {
                component: name.to_owned(),
                visuals: Vec::new(),
                frames: Vec::new(),
            })
            .collect();
        document.scopes[0].networks.push(Network {
            name: "link".to_owned(),
            structure: NetworkStructure::Link,
            description: None,
            selected: NetworkSelection {
                purpose: Purpose::Communication,
                carrier: Carrier::Electrical,
                profiles: BTreeSet::new(),
                rate: None,
                voltage: None,
                current: None,
                power: None,
                impedance: None,
                frequency: None,
                rf: None,
                pressure: None,
                flow: None,
                temperature: None,
            },
            participants: ["a", "b"]
                .into_iter()
                .map(|name| Participant {
                    name: name.to_owned(),
                    endpoint: FunctionalEndpointRef::Port(PortRef::local(name, "data")),
                    role: None,
                })
                .collect(),
            configuration: NetworkConfiguration::default(),
        });
        normalize_connectivity(&document).unwrap()
    }

    #[test]
    fn presentation_uses_exact_anchor_then_stable_synthetic_endpoint() {
        let graph = canonical_link();
        let resolver = graph.resolver(IncludeInstanceId::root());
        let a_port = resolver
            .port(&PortRef::local("a", "data"))
            .unwrap()
            .id()
            .clone();
        let b_port = resolver
            .port(&PortRef::local("b", "data"))
            .unwrap()
            .id()
            .clone();

        let mut app = App::new();
        app.add_plugins((CanonicalConnectivityPlugin, NetworkOverlayPlugin));
        app.update();
        let generation = *app.world().resource::<CanonicalConnectivityGeneration>();
        let a_component = app
            .world_mut()
            .spawn(CompEntity {
                comp_index: 0,
                name: "a".to_owned(),
            })
            .id();
        let b_component = app
            .world_mut()
            .spawn(CompEntity {
                comp_index: 1,
                name: "b".to_owned(),
            })
            .id();
        let exact_anchor = app
            .world_mut()
            .spawn((
                CanonicalConnectivityAnchor(a_port),
                Transform::from_xyz(0.1, 0.2, 0.3),
            ))
            .id();
        app.world_mut()
            .write_message(CanonicalConnectivityUpdate::ready(generation, graph));
        app.update();

        let overlay = app.world().resource::<NetworkOverlayScene>();
        assert_eq!(overlay.segments.len(), 1);
        assert_eq!(
            overlay.segments[0].endpoints,
            vec![
                NetEndpoint::Explicit(exact_anchor),
                overlay.segments[0].endpoints[1],
            ]
        );
        let NetEndpoint::Synthetic(synthetic) = overlay.segments[0].endpoints[1] else {
            panic!("the endpoint without an exact presentation entity must be synthetic");
        };
        assert_ne!(synthetic, a_component);
        assert_ne!(synthetic, b_component);
        assert_eq!(
            app.world().get::<ConnectivityEndpointGlyph>(synthetic),
            Some(&ConnectivityEndpointGlyph(b_port.clone()))
        );
        assert!(app
            .world()
            .get::<Transform>(synthetic)
            .is_some_and(|transform| transform.translation != Vec3::ZERO));

        let late_exact_anchor = app
            .world_mut()
            .spawn((
                CanonicalConnectivityAnchor(b_port),
                Transform::from_xyz(0.4, 0.5, 0.6),
            ))
            .id();
        app.update();

        let overlay = app.world().resource::<NetworkOverlayScene>();
        assert_eq!(
            overlay.segments[0].endpoints,
            vec![
                NetEndpoint::Explicit(exact_anchor),
                NetEndpoint::Explicit(late_exact_anchor),
            ]
        );
        assert!(app.world().get_entity(synthetic).is_err());
    }

    #[test]
    fn synthetic_offsets_are_nonzero_and_distinct() {
        let offsets = (0..4)
            .map(|index| synthetic_endpoint_offset(index, 4))
            .collect::<Vec<_>>();
        assert!(offsets.iter().all(|offset| *offset != Vec3::ZERO));
        for (index, offset) in offsets.iter().enumerate() {
            assert!(!offsets[..index].contains(offset));
        }
    }

    #[test]
    fn mesh_topology_uses_every_unique_participant_pair() {
        assert_eq!(complete_mesh_pairs(0).collect::<Vec<_>>(), vec![]);
        assert_eq!(complete_mesh_pairs(1).collect::<Vec<_>>(), vec![]);
        assert_eq!(
            complete_mesh_pairs(4).collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        );
    }

    #[test]
    fn root_component_identity_lookup_is_exact() {
        let graph = canonical_link();
        let resolver = graph.resolver(IncludeInstanceId::root());
        assert_ne!(
            resolver.component(&ComponentRef::local("a")).unwrap().id(),
            resolver.component(&ComponentRef::local("b")).unwrap().id()
        );
    }
}
