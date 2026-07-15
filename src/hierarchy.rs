//! The shared "Hierarchy" comp-tree panel: a flat, indented, clickable list of EVERY comp in the
//! loaded doc, including geometry-less reference links / frames (e.g. `base_footprint`, `lidar_link`,
//! sensor mounts) that 3D mesh picking can never reach (picking only hits meshes).
//!
//! The layout is a PURE function ([`hierarchy_rows`]) over the kinematic spanning tree from
//! [`crate::kinematics::build_kinematic_tree`], so the depth-first ordering + depth assignment unit-test
//! headless. The panel itself ([`hierarchy_panel`]) is read-only: it only writes the [`Selected`]
//! resource (selection), never the doc; editing stays in dendrite_build's Inspector. Selecting a row
//! drives exactly what mesh-picking would (the same `Selected` resource), so a geometry-less comp lights
//! up the fallback highlight (box + triad at its origin) that [`crate::scene`] already draws.
use crate::doc::HcdfDoc;
use crate::kinematics::build_kinematic_tree;
use crate::pick::Selected;
use crate::scene::CompEntity;
use crate::schema::Hcdf;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};
use std::sync::Arc;

/// One row of the hierarchy list: which comp, its display name, its depth in the kinematic tree (roots
/// are depth 0), and whether it carries any renderable geometry (visual OR collision). A geometry-less
/// row is a reference link / frame that 3D picking can never select, the whole reason this panel exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HierarchyRow {
    pub comp_index: usize,
    pub name: String,
    pub depth: usize,
    pub has_geometry: bool,
}

/// Flatten the kinematic tree into a depth-first, indented row list with EVERY comp present exactly once.
///
/// Built from [`build_kinematic_tree`]: each tree root is emitted (depth 0) followed by its descendants
/// depth-first via the primary edges (a child's depth is its parent's depth + 1). The tree's `roots`
/// already include every comp with no incoming primary edge: geometry-less frames, floating comps, and
/// comps unreachable through any joint all land there, so iterating roots then walking edges visits all
/// `h.comp` exactly once. Ordering is deterministic: roots in their `tree.roots` order (ascending comp
/// index, the order `build_kinematic_tree` produces), and each node's children in primary-edge order.
pub fn hierarchy_rows(h: &Hcdf) -> Vec<HierarchyRow> {
    let tree = build_kinematic_tree(h);

    // child comp index -> the children it parents, in primary-edge declaration order. We index by
    // parent so the DFS can fan out without re-scanning every edge per node.
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); h.comp.len()];
    for e in &tree.edges {
        // edges only reference comps in 0..h.comp.len() (build_kinematic_tree resolves by index), so
        // these indices are always in bounds.
        children_of[e.parent].push(e.child);
    }

    let row_for = |comp_index: usize, depth: usize| -> HierarchyRow {
        let comp = &h.comp[comp_index];
        HierarchyRow {
            comp_index,
            name: comp.name.clone(),
            depth,
            has_geometry: !comp.visual.is_empty() || !comp.collision.is_empty(),
        }
    };

    let mut rows = Vec::with_capacity(h.comp.len());
    // Explicit stack DFS (no recursion: a pathological loop-free tree could be deep, and an iterative
    // walk keeps the helper allocation-bounded and panic-free). Each stack entry is (comp_index, depth);
    // children are pushed in reverse so they pop in declaration order.
    let mut stack: Vec<(usize, usize)> = Vec::new();
    for &root in tree.roots.iter().rev() {
        stack.push((root, 0));
    }
    while let Some((ci, depth)) = stack.pop() {
        rows.push(row_for(ci, depth));
        for &child in children_of[ci].iter().rev() {
            stack.push((child, depth + 1));
        }
    }
    rows
}

/// Horizontal indent (egui points) applied per tree depth in the [`hierarchy_panel`].
const INDENT_PER_DEPTH: f32 = 14.0;

/// Per-system cache of the computed row list, keyed by the identity of the `Arc<Hcdf>` it was built
/// from ([`HcdfDoc`] publishes a fresh `Arc` per document, so the pointer is a cheap generation tag).
/// Rebuilding the kinematic tree + rows every frame was pure waste for a static doc; with this cache
/// the rows recompute only when the doc actually changes, and only while the window body draws.
#[derive(Default)]
pub struct HierarchyRowCache {
    /// `Arc::as_ptr` of the doc `rows` was computed from; `None` = invalidated. Tick-based reset
    /// (`doc.is_changed()`) backs the pointer key: a later doc could in principle reuse a freed
    /// allocation, so pointer equality alone is not trusted across doc changes.
    key: Option<usize>,
    rows: Vec<HierarchyRow>,
}

/// The shared "Hierarchy" panel (runs in `EguiPrimaryContextPass`): an indented, clickable list of every
/// comp in the loaded doc. Geometry-less rows (reference links / frames) get a dimmed "◇ … (frame)"
/// marker; geometry-carrying rows get a solid "▣". Clicking a row sets [`Selected`] to the entity whose
/// [`CompEntity`] comp_index matches the row, exactly what mesh picking would do, so geometry-less comps
/// that picking can never reach become selectable here (and, in dendrite_build, editable in the
/// Inspector). Selection keys on comp_index (not name) because HCDF comp names are neither unique nor
/// required.
///
/// Read-only: it only writes `Selected`, never the doc. Anchored bottom-right so it never collides with
/// the hcdviz/dendrite panel (top-left), the Inspector (top-right), or the Joints/Import panels
/// (bottom-left). The row list is cached per doc ([`HierarchyRowCache`]) and computed inside the window
/// body, so a collapsed window costs nothing and an open one rebuilds the tree only when the doc changes.
pub fn hierarchy_panel(
    mut contexts: EguiContexts,
    doc: Res<HcdfDoc>,
    comps: Query<(Entity, &CompEntity)>,
    mut selected: ResMut<Selected>,
    mut cache: Local<HierarchyRowCache>,
) -> Result {
    // Invalidate BEFORE any early-out: this runs every frame, so a doc republished while the window
    // is collapsed still drops the stale rows even though the body below never recomputes them.
    if doc.is_changed() {
        cache.key = None;
    }
    let Some(h) = doc.0.as_ref() else {
        return Ok(()); // no doc loaded, nothing to list.
    };
    // Every comp appears in the rows exactly once, so no comps ⇒ no rows ⇒ no window (the same
    // early-out as checking `rows.is_empty()`, without building the kinematic tree to find out).
    if h.comp.is_empty() {
        return Ok(());
    }
    // The comp_index of the currently selected comp (if any), to mark its row.
    let selected_index = selected
        .0
        .and_then(|e| comps.get(e).ok())
        .map(|(_, ce)| ce.comp_index);

    let ctx = contexts.ctx_mut()?;
    // Collected outside the closure so the borrow on `selected` is free while drawing. We capture the
    // row's comp_index (the authoritative identity that hierarchy_rows, is_selected, and CompEntity all
    // key on) rather than its name: comp names are neither unique nor required in HCDF, so name-matching
    // would select the wrong comp for duplicate/empty names.
    let mut clicked: Option<usize> = None;

    // A sane default height (40% of the viewport) so the window opens usefully large yet on-screen; the
    // window stays egui-default-resizable and the fill-height ScrollArea below lets the tree follow a
    // resize. egui remembers the dragged size by window Id for the session, so the size sticks.
    let default_h = (ctx.content_rect().height() * 0.4).max(160.0);
    egui::Window::new("Hierarchy")
        .default_width(240.0)
        .default_height(default_h)
        .default_open(false)
        .anchor(egui::Align2::RIGHT_BOTTOM, [-8.0, -8.0])
        .show(ctx, |ui| {
            // Rows are computed HERE (inside the window body) so a collapsed window skips the tree
            // build entirely, and the cache makes an open window recompute only on a fresh doc.
            let key = Arc::as_ptr(h) as usize;
            if cache.key != Some(key) {
                cache.rows = hierarchy_rows(h);
                cache.key = Some(key);
            }
            // Fill the window's available height (no hard cap) so resizing the window resizes the tree view
            // and the size the user drags to sticks; auto_shrink=false keeps it filling even for a short tree.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    for row in &cache.rows {
                        ui.horizontal(|ui| {
                            ui.add_space(row.depth as f32 * INDENT_PER_DEPTH);
                            let is_selected = Some(row.comp_index) == selected_index;
                            // ▣ = has geometry (3D-pickable); ◇ + dim + "(frame)" = geometry-less reference
                            // link the user can only reach from this list.
                            let label = if row.has_geometry {
                                egui::RichText::new(format!("▣ {}", row.name))
                            } else {
                                egui::RichText::new(format!("◇ {} (frame)", row.name)).weak()
                            };
                            if ui.selectable_label(is_selected, label).clicked() {
                                clicked = Some(row.comp_index);
                            }
                        });
                    }
                });
        });

    // Apply the click after the window closes (the borrow on `selected` was idle during drawing). Match
    // the row's comp_index to its live CompEntity; if the doc was just reloaded the entity may be absent,
    // in which case the click is a harmless no-op until the next frame's entity exists. Matching on
    // comp_index (not name) keeps selection consistent with the is_selected highlight even when comp
    // names collide or are empty.
    if let Some(ci) = clicked {
        if let Some((entity, _)) = comps.iter().find(|(_, ce)| ce.comp_index == ci) {
            selected.0 = Some(entity);
        }
    }

    Ok(())
}
