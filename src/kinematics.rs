//! Kinematic-tree extraction from `<joint>` elements.
//!
//! The scene graph is defined by joints, not nesting: each joint's `parent` comp is the Bevy parent
//! and its `child` comp the Bevy child, with the child entity's local `Transform` = `joint.origin`.
//! Comps with no incoming joint are roots (children of `WorldRoot`). A comp targeted by several joints
//! (parallel mechanisms / closed loops) gets ONE primary tree edge (the first non-`<loop>` joint),
//! and the remaining joints are surfaced as constraint links for the kinematics display to draw.
//!
//! This module is pure (no Bevy entities) so the spanning-tree logic unit-tests headless.
use crate::schema::{Hcdf, Joint};
use std::collections::{HashMap, HashSet};

/// One parent→child tree edge with the joint that defines it (index into `hcdf.joint`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeEdge {
    pub parent: usize,
    pub child: usize,
    pub joint: usize,
}

/// A joint that targets an already-parented child (a loop/parallel constraint), drawn as a link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintLink {
    pub parent: usize,
    pub child: usize,
    pub joint: usize,
}

/// The extracted tree: comp indices, primary edges, root comps, and extra constraint links.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KinematicTree {
    /// Comp indices that have no incoming primary edge (attach to `WorldRoot`).
    pub roots: Vec<usize>,
    /// Primary parent→child edges forming the spanning tree.
    pub edges: Vec<TreeEdge>,
    /// Secondary joints onto an already-claimed child (loops / parallel mechanisms).
    pub constraints: Vec<ConstraintLink>,
    /// child comp index → its primary edge index in `edges` (for fast lookup).
    pub primary_edge_of: HashMap<usize, usize>,
}

fn is_loop(j: &Joint) -> bool {
    // `@type` is a typed `JointType` enum now and has no `loop` variant (a document with `type="loop"`
    // fails to parse), so a loop closure is identified solely by the presence of the `<loop>` child.
    j.loop_.is_some()
}

/// Extract the kinematic spanning tree from an [`Hcdf`].
///
/// Robust against: missing parent/child names, joints referencing unknown comps (skipped), multiple
/// joints onto one child (first non-loop wins the edge, rest become constraints), and cycles (a joint
/// whose child is an ancestor of its parent is demoted to a constraint so the tree stays acyclic).
pub fn build_kinematic_tree(h: &Hcdf) -> KinematicTree {
    let by_name: HashMap<&str, usize> = h
        .comp
        .iter()
        .enumerate()
        .map(|(i, c)| (c.name.as_str(), i))
        .collect();

    let mut tree = KinematicTree::default();
    // Order joints so non-loop joints get first claim on a child.
    let mut order: Vec<usize> = (0..h.joint.len()).collect();
    order.sort_by_key(|&i| is_loop(&h.joint[i]));

    // child comp index -> parent comp index, to detect cycles cheaply.
    let mut parent_of: HashMap<usize, usize> = HashMap::new();

    for ji in order {
        let j = &h.joint[ji];
        let (Some(pn), Some(cn)) = (
            j.parent.as_ref().and_then(|p| p.comp.as_deref()),
            j.child.as_ref().and_then(|c| c.comp.as_deref()),
        ) else {
            continue;
        };
        let (Some(&pi), Some(&ci)) = (by_name.get(pn), by_name.get(cn)) else {
            continue; // references an unknown comp, so skip silently (lenient).
        };

        let child_taken = tree.primary_edge_of.contains_key(&ci);
        let creates_cycle = !child_taken && would_cycle(&parent_of, pi, ci);

        if child_taken || creates_cycle || is_loop(j) {
            tree.constraints.push(ConstraintLink {
                parent: pi,
                child: ci,
                joint: ji,
            });
        } else {
            let edge_idx = tree.edges.len();
            tree.edges.push(TreeEdge {
                parent: pi,
                child: ci,
                joint: ji,
            });
            tree.primary_edge_of.insert(ci, edge_idx);
            parent_of.insert(ci, pi);
        }
    }

    // Roots: every comp that never became someone's primary child.
    let claimed: HashSet<usize> = tree.primary_edge_of.keys().copied().collect();
    tree.roots = (0..h.comp.len()).filter(|i| !claimed.contains(i)).collect();
    tree
}

/// Would adding edge parent→child introduce a cycle? True if `child` is already an ancestor of
/// `parent` in the current `parent_of` map.
fn would_cycle(parent_of: &HashMap<usize, usize>, parent: usize, child: usize) -> bool {
    let mut cur = parent;
    // Walk up from parent; if we reach child, the new edge closes a loop. Bounded by map size.
    for _ in 0..=parent_of.len() {
        if cur == child {
            return true;
        }
        match parent_of.get(&cur) {
            Some(&p) => cur = p,
            None => return false,
        }
    }
    false
}
