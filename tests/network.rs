//! Canonical graph-driven logical-network overlay tests.

use std::collections::{BTreeSet, HashMap};

use bevy::prelude::Entity;
use hcdviz::network::{build_network_overlay, NetEndpoint, NetTopo};
use hcdviz::schema::connectivity::{normalize_connectivity, StableObjectId};
use hcdviz::schema::model::connectivity::{
    Capabilities, Carrier, ComponentConnectivity, ComponentRef, ConnectivityDocument,
    ConnectivityScope, DocumentIdentity, FunctionalEndpointRef, Hop, HopOwnerRef, HopRef,
    IncludeInstanceId, Leg, LegEnd, Network, NetworkConfiguration, NetworkRef, NetworkSelection,
    NetworkStructure, Participant, ParticipantRef, Port, PortRef, Purpose, StructuralAnchors,
};

fn entity(index: u32) -> Entity {
    Entity::from_raw_u32(index).expect("valid test entity")
}

fn capabilities(carrier: Carrier) -> Capabilities {
    Capabilities {
        purposes: BTreeSet::from([Purpose::Communication]),
        carriers: BTreeSet::from([carrier]),
        profiles: BTreeSet::new(),
        limits: Default::default(),
    }
}

fn component(name: &str, port: &str, carrier: Carrier) -> ComponentConnectivity {
    ComponentConnectivity {
        component: name.to_owned(),
        ports: vec![Port {
            name: port.to_owned(),
            capabilities: capabilities(carrier),
            channels: Vec::new(),
        }],
        connectors: Vec::new(),
        antennas: Vec::new(),
        functions: Vec::new(),
        paths: Vec::new(),
        junctions: Vec::new(),
        terminations: Vec::new(),
    }
}

fn selection(carrier: Carrier) -> NetworkSelection {
    NetworkSelection {
        purpose: Purpose::Communication,
        carrier,
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
    }
}

fn participant(name: &str, component: &str, port: &str) -> Participant {
    Participant {
        name: name.to_owned(),
        endpoint: FunctionalEndpointRef::Port(PortRef::local(component, port)),
        role: None,
    }
}

fn link_scope(instance: IncludeInstanceId, network_name: &str) -> ConnectivityScope {
    let carrier = Carrier::Electrical;
    let mut scope = ConnectivityScope::new(instance);
    scope.components = vec![
        component("controller", "link", carrier),
        component("actuator", "link", carrier),
    ];
    scope.structural_anchors = ["controller", "actuator"]
        .into_iter()
        .map(|component| StructuralAnchors {
            component: component.to_owned(),
            visuals: Vec::new(),
            frames: Vec::new(),
        })
        .collect();
    scope.networks.push(Network {
        name: network_name.to_owned(),
        structure: NetworkStructure::Link,
        description: None,
        selected: selection(carrier),
        participants: vec![
            participant("controller", "controller", "link"),
            participant("actuator", "actuator", "link"),
        ],
        configuration: NetworkConfiguration::default(),
    });
    scope
}

fn graph_maps(
    graph: &hcdviz::schema::connectivity::NormalizedConnectivityGraph,
    instances: &[IncludeInstanceId],
) -> (
    HashMap<StableObjectId, NetEndpoint>,
    HashMap<StableObjectId, Entity>,
) {
    let mut endpoints = HashMap::new();
    let mut components = HashMap::new();
    let mut next = 1u32;
    for instance in instances {
        let resolver = graph.resolver(instance.clone());
        for component_name in ["controller", "actuator"] {
            let component = resolver
                .component(&ComponentRef::local(component_name))
                .unwrap();
            let component_entity = entity(next);
            next += 1;
            components.insert(component.id().clone(), component_entity);
            let port = resolver
                .port(&PortRef::local(component_name, "link"))
                .unwrap();
            endpoints.insert(port.id().clone(), NetEndpoint::Explicit(entity(next)));
            next += 1;
        }
    }
    (endpoints, components)
}

#[test]
fn exact_graph_identities_drive_link_and_missing_anchor_never_falls_back() {
    let root = IncludeInstanceId::root();
    let mut document =
        ConnectivityDocument::new(DocumentIdentity::new("memory://hcdviz/exact-link").unwrap());
    document.scopes[0] = link_scope(root.clone(), "control-link");
    let graph = normalize_connectivity(&document).unwrap();
    let resolver = graph.resolver(root.clone());
    let network = resolver
        .network(&NetworkRef::local("control-link"))
        .unwrap();
    let (endpoints, components) = graph_maps(&graph, &[root]);

    let segments = build_network_overlay(
        &graph,
        |id| endpoints.get(id).copied(),
        |id| components.get(id).copied(),
    );
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].network, *network.id());
    assert_eq!(segments[0].network_label, "control-link");
    assert_eq!(segments[0].topo, NetTopo::Link { radiated: false });
    assert!(segments[0]
        .endpoints
        .iter()
        .all(|endpoint| matches!(endpoint, NetEndpoint::Explicit(_))));

    let missing = endpoints.keys().next().unwrap().clone();
    let segments = build_network_overlay(
        &graph,
        |id| {
            (id != &missing)
                .then(|| endpoints.get(id).copied())
                .flatten()
        },
        |id| components.get(id).copied(),
    );
    assert!(
        segments.is_empty(),
        "a missing exact endpoint anchor must omit the link, never use a component origin"
    );
}

#[test]
fn repeated_include_local_names_remain_distinct_stable_networks() {
    let first = IncludeInstanceId::root().child("drive", 0);
    let second = IncludeInstanceId::root().child("drive", 1);
    let document = ConnectivityDocument {
        document: DocumentIdentity::new("memory://hcdviz/repeated").unwrap(),
        scopes: vec![
            ConnectivityScope::root(),
            link_scope(first.clone(), "bus"),
            link_scope(second.clone(), "bus"),
        ],
    };
    let graph = normalize_connectivity(&document).unwrap();
    let (endpoints, components) = graph_maps(&graph, &[first.clone(), second.clone()]);
    let segments = build_network_overlay(
        &graph,
        |id| endpoints.get(id).copied(),
        |id| components.get(id).copied(),
    );

    assert_eq!(segments.len(), 2);
    assert_ne!(segments[0].network, segments[1].network);
    assert_ne!(segments[0].network_label, segments[1].network_label);
    assert_eq!(
        segments[0].network_label,
        format!("bus ({})", first.display_path())
    );
    assert_eq!(
        segments[1].network_label,
        format!("bus ({})", second.display_path())
    );
    let first_id = graph
        .resolver(first)
        .network(&NetworkRef::local("bus"))
        .unwrap()
        .id();
    let second_id = graph
        .resolver(second)
        .network(&NetworkRef::local("bus"))
        .unwrap()
        .id();
    assert_ne!(first_id, second_id);
}

#[test]
fn chain_order_comes_from_typed_hop_and_leg_edges() {
    let carrier = Carrier::Electrical;
    let mut document =
        ConnectivityDocument::new(DocumentIdentity::new("memory://hcdviz/chain").unwrap());
    let mut middle = component("b", "in", carrier);
    middle.ports.push(Port {
        name: "out".to_owned(),
        capabilities: capabilities(carrier),
        channels: Vec::new(),
    });
    document.scopes[0].components = vec![
        component("a", "data", carrier),
        middle,
        component("c", "data", carrier),
    ];
    document.scopes[0].structural_anchors = ["a", "b", "c"]
        .into_iter()
        .map(|component| StructuralAnchors {
            component: component.to_owned(),
            visuals: Vec::new(),
            frames: Vec::new(),
        })
        .collect();
    let network = "chain";
    let participant_ref = |name: &str| ParticipantRef::local(network, name);
    let hop_ref = |name: &str| HopRef {
        network: NetworkRef::local(network),
        hop: name.to_owned(),
    };
    document.scopes[0].networks.push(Network {
        name: network.to_owned(),
        structure: NetworkStructure::Chain(hcdviz::schema::model::connectivity::PathTopology {
            hops: ["a", "b", "c"]
                .into_iter()
                .map(|name| Hop {
                    name: format!("hop-{name}"),
                    owner: HopOwnerRef::Component(ComponentRef::local(name)),
                    role: None,
                    description: None,
                    processing_delay_ns: None,
                })
                .collect(),
            legs: vec![
                Leg {
                    name: "a-to-b".to_owned(),
                    from: LegEnd {
                        hop: hop_ref("hop-a"),
                        participant: participant_ref("a-out"),
                    },
                    to: LegEnd {
                        hop: hop_ref("hop-b"),
                        participant: participant_ref("b-in"),
                    },
                },
                Leg {
                    name: "b-to-c".to_owned(),
                    from: LegEnd {
                        hop: hop_ref("hop-b"),
                        participant: participant_ref("b-out"),
                    },
                    to: LegEnd {
                        hop: hop_ref("hop-c"),
                        participant: participant_ref("c-in"),
                    },
                },
            ],
        }),
        description: None,
        selected: selection(carrier),
        participants: vec![
            participant("a-out", "a", "data"),
            participant("b-in", "b", "in"),
            participant("b-out", "b", "out"),
            participant("c-in", "c", "data"),
        ],
        configuration: NetworkConfiguration::default(),
    });
    let graph = normalize_connectivity(&document).unwrap();
    let resolver = graph.resolver(IncludeInstanceId::root());
    let mut endpoints = HashMap::new();
    let mut components = HashMap::new();
    for (index, (name, port)) in [("a", "data"), ("b", "in"), ("b", "out"), ("c", "data")]
        .into_iter()
        .enumerate()
    {
        endpoints.insert(
            resolver
                .port(&PortRef::local(name, port))
                .unwrap()
                .id()
                .clone(),
            NetEndpoint::Explicit(entity(index as u32 + 10)),
        );
        components
            .entry(
                resolver
                    .component(&ComponentRef::local(name))
                    .unwrap()
                    .id()
                    .clone(),
            )
            .or_insert_with(|| entity(index as u32 + 20));
    }
    let segments = build_network_overlay(
        &graph,
        |id| endpoints.get(id).copied(),
        |id| components.get(id).copied(),
    );
    assert_eq!(segments.len(), 1);
    assert_eq!(
        segments[0].topo,
        NetTopo::Chain {
            ring: false,
            legs: vec![(0, 1), (2, 3)],
        }
    );
    assert_eq!(segments[0].endpoints.len(), 4);
}
