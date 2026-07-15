//! Real scene build from the loaded HCDF, plus the built-in rviz-style displays.
//!
//! Build-once / render-static: the entity graph is spawned on `Changed<HcdfDoc>` and then
//! idle. Persistent meshes (visuals, collision, FOV frustums, the selection-triad arrows) are spawned
//! entities sharing cached handles; ephemera (grid, frame triads, world axes, constraint links, the
//! selection highlight box) are immediate-mode gizmos gated by `run_if(display_enabled(id))`.
//!
//! Scene-graph mapping:
//!   WorldRoot(world→Bevy basis)
//!     └─ BodyBasis(body→world basis) per root comp
//!         └─ CompEntity ── visual/collision/sensor/port children at their own poses
//!             └─ child CompEntity via joint.origin (the kinematic tree edge)
use bevy::camera::primitives::Aabb;
use bevy::gltf::GltfAssetLabel;
use bevy::mesh::PrimitiveTopology;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::world_serialization::WorldAssetRoot;

use crate::camera::SceneFitted;
use crate::display::{display_enabled, Display, DisplayRegistry};
use crate::doc::HcdfDoc;
use crate::frame::{pose_to_transform, FrameConvention, WorldRoot};
use crate::geometry::{
    frustum_optical_to_body_rotation, resolve_collision_geometry, resolve_frustum_mesh,
    resolve_general_primitive, resolve_lidar_scan_mesh, resolve_visual_geometry, with_extra_scale,
    GeometryCache,
};
use crate::pick::{
    HighlightSet, IsolateSelection, IsolateSet, Selected, SelectionOverrides, SensorVizGlobal,
    SensorVizOverrides,
};
use crate::schema::model::enums::AxisValue;
use crate::schema::model::Color as SColor;
use crate::schema::model::{AxisAlign, LidarParams, SensorDriver};
use crate::schema::{Comp, Hcdf, VisualAppearance};

// Display ids (stable public contract).
pub const ID_VISUAL: &str = "hcdviz.visual";
pub const ID_KINEMATICS: &str = "hcdviz.kinematics";
pub const ID_FRAMES: &str = "hcdviz.frames";
pub const ID_SENSORS: &str = "hcdviz.sensors";
pub const ID_SENSOR_AXIS_ALIGN: &str = "hcdviz.sensor_axis_align";
pub const ID_CONNECTIVITY: &str = "hcdviz.connectivity";
pub const ID_NETWORK: &str = "hcdviz.network";
pub const ID_COLLISION: &str = "hcdviz.collision";
pub const ID_INERTIAL: &str = "hcdviz.inertial";
pub const ID_GRID: &str = "hcdviz.grid";

/// The six comp-owned display kinds offered as per-link overrides in the inspector "This link" section
/// (Kinematics + Grid are scene-level and intentionally excluded). `(id, label)` so the UI can
/// reuse it for its checkboxes; the sync systems can pair each kind's id with its own helper call.
pub const PER_LINK_KINDS: [(&str, &str); 6] = [
    (ID_VISUAL, "Visual"),
    (ID_COLLISION, "Collision"),
    (ID_FRAMES, "Frames"),
    (ID_SENSORS, "Sensors"),
    (ID_CONNECTIVITY, "Connectivity"),
    (ID_INERTIAL, "Inertial"),
];

/// Ordering label for the scene rebuild, so downstream `Update` systems (e.g. joint articulation) can
/// run `.after(SceneSet::Rebuild)` without depending on the rebuild system's private signature.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneSet {
    /// `rebuild_on_change`: despawns + respawns the entity graph on `Changed<HcdfDoc>`.
    Rebuild,
}

pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        // `ArticulatedJoints` is populated here (the scene owns the catalogue), so init it in this
        // plugin too, since `rebuild_on_change` writes it whether or not the UI/JointsPlugin is present.
        // `init_resource` is idempotent, so JointsPlugin re-initing it is harmless.
        app.init_resource::<GeometryCache>()
            .init_resource::<crate::joints::ArticulatedJoints>()
            // `rebuild_on_change` reads these to pick each sensor's effective drawn extent, so the scene
            // owns their init too (idempotent with `PickPlugin`): the rebuild must run without PickPlugin.
            .init_resource::<SensorVizOverrides>()
            .init_resource::<SensorVizGlobal>()
            .add_message::<SceneFitted>()
            .init_resource::<RenderShininess>()
            .init_resource::<VisualToggleGroups>()
            .add_systems(Startup, setup_lighting)
            .add_systems(Update, rebuild_on_change.in_set(SceneSet::Rebuild))
            .add_systems(Update, refresh_toggle_groups)
            // Visual submesh selection: filter a spawned glTF scene by node name as its tree materializes,
            // then (for a lone `@center` include) recenter once its Aabbs are ready. Gated on scene
            // growth so quiet frames are free; the center pass runs after the visibility pass so it
            // measures the drawn subtree, not the whole model.
            .add_systems(
                Update,
                (apply_submesh_visibility, apply_submesh_center)
                    .chain()
                    .run_if(submesh_scene_growing),
            )
            .add_systems(
                Update,
                (
                    compute_missing_normals,
                    // Scanning every StandardMaterial each frame was waste: run only when the toggle
                    // flips or material assets change (Added messages cover late-loading glTF
                    // materials, so they still converge whenever they appear). Eager OR so the
                    // message reader is drained even on frames where the toggle also flipped.
                    apply_shininess.run_if(
                        resource_changed::<RenderShininess>
                            .or_eager(on_message::<AssetEvent<StandardMaterial>>),
                    ),
                ),
            );
    }
}

/// Opt-in "render shininess": when ON, mesh materials get a metallic/low-roughness sheen (the RViz-style
/// metallic look); when OFF, they render at the faithful matte the bake produced. DAE/COLLADA robot
/// materials are Lambert (matte), so this is purely a viewer aesthetic, not source data, hence opt-in,
/// default OFF. Toggled from the Displays panel.
#[derive(Resource, Default)]
pub struct RenderShininess(pub bool);

/// The metallic/roughness applied to every mesh `StandardMaterial` per [`RenderShininess`]. ON gives a
/// dielectric specular sheen from the directional light (no env map needed); OFF restores matte. Gated
/// (see [`ScenePlugin`]) on the toggle changing or `AssetEvent<StandardMaterial>` messages, so it never
/// scans on quiet frames. Only materials whose values DIFFER from the target are touched (a read scan +
/// targeted `get_mut`), so the Modified messages its own writes emit cost one extra no-op pass before
/// the system goes quiet.
fn apply_shininess(setting: Res<RenderShininess>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let (metallic, roughness) = if setting.0 {
        (0.2_f32, 0.4_f32)
    } else {
        (0.0_f32, 1.0_f32)
    };
    let stale: Vec<_> = materials
        .iter()
        .filter(|(_, m)| {
            (m.metallic - metallic).abs() > 1e-4
                || (m.perceptual_roughness - roughness).abs() > 1e-4
        })
        .map(|(id, _)| id)
        .collect();
    for id in stale {
        if let Some(mut m) = materials.get_mut(id) {
            m.metallic = metallic;
            m.perceptual_roughness = roughness;
        }
    }
}

/// Loaded glTF/GLB visual meshes commonly arrive POSITION-only: the bake omits the `NORMAL` attribute
/// for STL / scaled sources (to stay byte-identical with trimesh), and Bevy's glTF loader does not
/// synthesize normals. Without normals the directional light contributes nothing (N·L is undefined), so
/// the surface renders flat and washed-out: the "clay" look, even when the GLB carries correct baked
/// colours. Compute FLAT normals on any newly-spawned mesh that lacks them (matching the STL loader's
/// flat shading), restoring proper shading. Meshes that already carry normals (hcdviz's own primitives /
/// STL, or a GLB whose authored normals survived the bake) are skipped by the guard, and the FOV
/// frustums ([`SensorMarker`] entities) are excluded by the query (they render unlit and are built
/// position-only on purpose) so this only ever touches the normal-less GLB visuals.
fn compute_missing_normals(
    new_meshes: Query<&Mesh3d, (Added<Mesh3d>, Without<SensorMarker>)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for handle in &new_meshes {
        if let Some(mut mesh) = meshes.get_mut(&handle.0) {
            ensure_flat_normals(&mut mesh);
        }
    }
}

/// Give `mesh` flat per-face normals IF it has none. No-op when normals already exist, or when the mesh
/// is not a triangle list with positions (`compute_flat_normals` requires both). `duplicate_vertices`
/// drops the shared index buffer so each triangle gets its own crisp face normal (flat shading, matching
/// the STL loader); it is a no-op on an already-non-indexed mesh.
fn ensure_flat_normals(mesh: &mut Mesh) {
    if mesh.contains_attribute(Mesh::ATTRIBUTE_NORMAL)
        || mesh.primitive_topology() != PrimitiveTopology::TriangleList
        || !mesh.contains_attribute(Mesh::ATTRIBUTE_POSITION)
    {
        return;
    }
    mesh.duplicate_vertices();
    mesh.compute_flat_normals();
}

#[cfg(test)]
mod normals_tests {
    use super::*;
    use bevy::asset::RenderAssetUsages;
    use bevy::mesh::{Indices, VertexAttributeValues};

    fn tri_no_normals() -> Mesh {
        // One indexed triangle in the XY plane (POSITION only): its flat face normal is +Z.
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        );
        mesh.insert_indices(Indices::U32(vec![0, 1, 2]));
        mesh
    }

    #[test]
    fn flat_normals_added_when_missing() {
        let mut mesh = tri_no_normals();
        assert!(!mesh.contains_attribute(Mesh::ATTRIBUTE_NORMAL));
        ensure_flat_normals(&mut mesh);
        let Some(VertexAttributeValues::Float32x3(n)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("normals should have been computed");
        };
        assert!(!n.is_empty());
        assert!(
            (n[0][2] - 1.0).abs() < 1e-6,
            "flat normal of an XY triangle is +Z, got {:?}",
            n[0]
        );
    }

    #[test]
    fn existing_normals_left_untouched() {
        let mut mesh = tri_no_normals();
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3]);
        ensure_flat_normals(&mut mesh); // must be a no-op (still indexed, 3 verts)
        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("positions present");
        };
        assert_eq!(pos.len(), 3, "skip path must not duplicate vertices");
    }
}

// ── components describing spawned HCDF content ───────────────────────────────

/// An HCDF `<comp>` entity. Carries its comp index into the loaded doc for the inspector.
#[derive(Component, Debug, Clone)]
pub struct CompEntity {
    pub comp_index: usize,
    pub name: String,
}

/// Marks everything spawned from a load so a reload can clear it in one query.
#[derive(Component)]
pub struct SceneItem;

/// The owning [`CompEntity`] of a toggleable spawned item, stamped at spawn time. Lets the visibility
/// systems decide "is this item's comp the selected one?" with an `Entity` compare instead of a
/// per-frame ancestor walk: every show/hide system queries `(&OwnerComp, &mut Visibility)`.
#[derive(Component, Clone, Copy)]
pub struct OwnerComp(pub Entity);

/// Pure visibility decision for one toggleable item (unit-testable headless; no World/GPU).
///
/// Is `owner` hidden by the active isolation mode? Two mutually exclusive modes:
///   * `isolate_set = Some(set)` (comp-set isolate): hidden unless `owner` is in the set. The single
///     `isolate`/`selected` inputs are ignored; an empty set hides everything (isolate to nothing).
///   * `isolate_set = None` (the default, single-selection path): hidden only while `isolate` is on
///     AND a DIFFERENT comp is selected. No selection ⇒ nothing is isolated away (isolate is a no-op),
///     which is why deselect auto-reverts.
///
/// Pure so both the item and the tree-edge isolation rules unit-test headless (no World/GPU).
pub fn isolate_hides(
    isolate: bool,
    selected: Option<Entity>,
    isolate_set: Option<&HashSet<Entity>>,
    owner: Entity,
) -> bool {
    match isolate_set {
        Some(set) => !set.contains(&owner),
        None => isolate && selected.is_some_and(|s| s != owner),
    }
}

/// Is a kinematic tree edge (parent→child comp) hidden by the active isolation mode? Comp-set isolate
/// keeps an edge only when BOTH endpoints are in the set (so a joint's own edge survives isolate-to-its
/// -comps); the single-selection path keeps only the edge INTO the selected comp, matching the item rule
/// against the edge's child. Pure for headless tests.
pub fn isolate_hides_edge(
    isolate: bool,
    selected: Option<Entity>,
    isolate_set: Option<&HashSet<Entity>>,
    parent: Entity,
    child: Entity,
) -> bool {
    match isolate_set {
        Some(set) => !(set.contains(&parent) && set.contains(&child)),
        None => isolate && selected.is_some_and(|s| s != child),
    }
}

/// An item shows iff its display kind is enabled AND it is not isolated away ([`isolate_hides`]). Uses
/// [`Visibility::Inherited`] for "show" (matching the show systems, so the item still follows its parent
/// hierarchy's visibility) and [`Visibility::Hidden`] for "hide".
pub fn item_visibility(
    kind_enabled: bool,
    isolate: bool,
    selected: Option<Entity>,
    owner: Entity,
    isolate_set: Option<&HashSet<Entity>>,
) -> Visibility {
    let show = kind_enabled && !isolate_hides(isolate, selected, isolate_set, owner);
    if show {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    }
}

/// Pure per-link "is this kind enabled for this item?" decision (unit-testable headless).
///
/// The SELECTED comp's items honor its per-link override (if the user set one for this kind); every
/// other comp (and any kind the selection hasn't overridden) follows the global toggle live. Feed the
/// result into [`item_visibility`] in place of the raw global value: isolate + selection still compose
/// exactly as before.
///
/// - not the selected owner ⇒ `global` (overrides never touch other comps)
/// - selected owner, no override for this kind ⇒ `global`
/// - selected owner, override `Some(v)` ⇒ `v`
pub fn effective_kind_enabled(
    global: bool,
    override_kind: Option<bool>,
    is_selected_owner: bool,
) -> bool {
    if is_selected_owner {
        override_kind.unwrap_or(global)
    } else {
        global
    }
}

/// A visual entity (glTF subtree root or primitive mesh) toggled by the visual display.
#[derive(Component)]
pub struct VisualItem;

/// The `toggle="…"` group name of a [`VisualItem`], stamped at spawn time for visuals that carry the
/// attribute (legacy dendrite semantics: e.g. a `case` group hides the enclosure over a bare PCB).
/// Ungrouped visuals never get one, so they are unaffected by group toggling.
#[derive(Component)]
pub struct ToggleGroup(pub String);

/// The doc's visual toggle groups plus which ones the user has hidden (legacy per-group
/// show/hide checkboxes, ported into the Displays panel as a sub-section under Visual).
///
/// `groups` is every DISTINCT non-empty `toggle` name in the loaded doc, sorted for a stable UI
/// order; `hidden` is the user's hide set. Rebuilt by `refresh_toggle_groups` on every doc change
/// with `hidden` CLEARED: all groups start visible (the legacy default), and stale names from a
/// previous doc can't linger. `sync_visual_visibility` reads it as one more AND-term of the
/// per-item decision, with `resource_changed` as an extra run trigger so quiet frames stay free.
#[derive(Resource, Default)]
pub struct VisualToggleGroups {
    pub groups: Vec<String>,
    pub hidden: HashSet<String>,
}

/// Pure: every distinct non-empty visual `toggle` group name in the doc, sorted (unit-testable
/// headless; no World/GPU).
pub fn collect_toggle_groups(h: &Hcdf) -> Vec<String> {
    let mut names: Vec<String> = h
        .comp
        .iter()
        .flat_map(|c| &c.visual)
        .filter_map(|v| v.toggle.as_deref())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Pure: is an item's toggle group currently shown? Ungrouped items (`None`) are always shown; a
/// grouped item shows unless its group is in the hide set. ANDed into the visual-kind decision by
/// `sync_visual_visibility`, so isolate/overrides still compose exactly as before.
pub fn toggle_group_visible(group: Option<&str>, hidden: &HashSet<String>) -> bool {
    group.is_none_or(|g| !hidden.contains(g))
}

/// Recompute [`VisualToggleGroups`] from the freshly loaded doc (all groups visible). Writes the
/// resource only on a doc change, so its change tick doubles as the resync trigger for
/// [`sync_visual_visibility`].
fn refresh_toggle_groups(doc: Res<HcdfDoc>, mut groups: ResMut<VisualToggleGroups>) {
    if !doc.is_changed() {
        return;
    }
    groups.groups = doc
        .0
        .as_deref()
        .map(collect_toggle_groups)
        .unwrap_or_default();
    groups.hidden.clear();
}

/// Frame triad to draw as a gizmo (frames display). `optical` ⇒ Z-forward convention.
#[derive(Component)]
pub struct FrameMarker {
    pub label: String,
    pub optical: bool,
    /// Human-facing `<frame @type>` label, drawn alongside the name so a frame's role reads in the
    /// scene. `None` when the frame declares no type. Computed once at spawn via [`frame_type_label`]
    /// (which spells out `tcp` per the UI text rule), so the per-frame gizmo draw never re-maps it.
    pub type_label: Option<String>,
}

/// Human-facing label for a `<frame @type>`: the schema token verbatim, EXCEPT `tcp` reads
/// "TCP (tool center point)" (the UI text rule; a bare "tcp" is opaque to a reader). `None` when the
/// frame declares no type. Pure, so the mapping is unit-testable headless.
pub fn frame_type_label(type_: Option<&str>) -> Option<String> {
    type_.map(|t| match t {
        "tcp" => "TCP (tool center point)".to_string(),
        other => other.to_string(),
    })
}

/// Sensor pose marker (for the sensors display triad + label).
#[derive(Component)]
pub struct SensorMarker {
    pub label: String,
}

/// Aligned-axes marker on a sensor pose node whose `<driver><axis-align>` remaps the raw driver axes.
/// Holds the [`axis_align_rotation`] of that remap (columns = where each raw axis lands); drawn by
/// [`SensorAxisAlignDisplay`] as a dashed, dimmed triad alongside the raw pose triad so IMU/mag
/// mounting can be verified at a glance. Only sensors that actually carry `<axis-align>` get one.
#[derive(Component)]
pub struct AlignedAxesMarker(pub Mat3);

/// Body-frame unit direction for one `<axis-align>` `AxisValue` literal.
fn axis_value_dir(v: AxisValue) -> Vec3 {
    match v {
        AxisValue::X => Vec3::X,
        AxisValue::NegX => Vec3::NEG_X,
        AxisValue::Y => Vec3::Y,
        AxisValue::NegY => Vec3::NEG_Y,
        AxisValue::Z => Vec3::Z,
        AxisValue::NegZ => Vec3::NEG_Z,
    }
}

/// The `<axis-align x= y= z=>` remap as a matrix whose COLUMNS are the body-frame directions the
/// sensor's raw X/Y/Z axes map to (an absent attribute means "no remap", per the XSD defaults), so
/// `m * Vec3::X` is the aligned X direction. Legacy dendrite's `DeviceAxisAlign::to_rotation_matrix`
/// expressed the same mapping as ROWS `[x; y; z]`; this is its transpose, identical semantics, but
/// column-major so it applies directly to Bevy vectors. Degenerate remaps (two axes mapped onto the
/// same direction) are drawn as authored rather than rejected, matching the legacy per-axis arrows.
pub fn axis_align_rotation(a: &AxisAlign) -> Mat3 {
    Mat3::from_cols(
        axis_value_dir(a.x.unwrap_or(AxisValue::X)),
        axis_value_dir(a.y.unwrap_or(AxisValue::Y)),
        axis_value_dir(a.z.unwrap_or(AxisValue::Z)),
    )
}

/// Collision overlay mesh (toggled translucent by the collision display).
#[derive(Component)]
pub struct CollisionItem;

/// Inertial CG marker (toggled by the inertial display).
#[derive(Component)]
pub struct InertialMarker;

/// A canonical functional-endpoint annotation toggled by the connectivity display.
#[derive(Component)]
pub struct ConnectorMarker;

/// A loop/parallel-mechanism constraint link to draw as a dashed gizmo line: child & parent entities,
/// plus the closure joint's index into `hcdf.joint`: the stable key that pairs this gizmo with its
/// [`crate::loop_solver::LoopClosureStatus`] entry, so a closure the solver reports open beyond
/// tolerance can flag warning-red instead of the usual orange.
#[derive(Component)]
pub struct ConstraintLinkMarker {
    pub parent: Entity,
    pub child: Entity,
    pub joint: usize,
}

/// A spanning-tree kinematic edge (parent comp → child comp) to draw as a thin connector line, so the
/// kinematic skeleton is visible for tree robots (which have no loop constraints). Distinct color from
/// [`ConstraintLinkMarker`].
#[derive(Component)]
pub struct TreeEdgeMarker {
    pub parent: Entity,
    pub child: Entity,
}

fn setup_lighting(mut commands: Commands) {
    commands.insert_resource(bevy::light::GlobalAmbientLight {
        color: Color::srgb(0.9, 0.95, 1.0),
        brightness: 250.0,
        ..default()
    });
    commands.spawn((
        DirectionalLight {
            illuminance: 6000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Default sRGBA used when a primitive visual gives neither inline nor resolvable named color.
const DEFAULT_VISUAL_RGBA: [f32; 4] = [0.6, 0.62, 0.66, 1.0];

/// Resolve a `<color>` reference for a visual to sRGBA, using the root palette for named-only refs.
/// Inline `@rgba` wins; named-only looks up the palette; neither ⇒ a sensible default (NOT black).
fn resolve_color(c: Option<&SColor>, palette: &HashMap<String, [f32; 4]>) -> [f32; 4] {
    let Some(c) = c else {
        return DEFAULT_VISUAL_RGBA;
    };
    if let Some(rgba) = c.rgba.as_deref().and_then(parse_rgba) {
        return rgba;
    }
    if let Some(name) = c.name.as_deref() {
        if let Some(rgba) = palette.get(name) {
            return *rgba;
        }
    }
    DEFAULT_VISUAL_RGBA
}

fn parse_rgba(s: &str) -> Option<[f32; 4]> {
    let v: Vec<f32> = s
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    match v.len() {
        3 => Some([v[0], v[1], v[2], 1.0]),
        4 => Some([v[0], v[1], v[2], v[3]]),
        _ => None,
    }
}

fn build_palette(h: &Hcdf) -> HashMap<String, [f32; 4]> {
    let mut m = HashMap::new();
    for c in &h.color {
        if let (Some(name), Some(rgba)) =
            (c.name.as_deref(), c.rgba.as_deref().and_then(parse_rgba))
        {
            m.insert(name.to_string(), rgba);
        }
    }
    m
}

/// The asset writers + cache the scene build needs, grouped into one [`SystemParam`] so the rebuild
/// system stays within the argument budget (idiomatic Bevy; no lint suppression needed).
#[derive(bevy::ecs::system::SystemParam)]
struct SceneAssets<'w> {
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    cache: ResMut<'w, GeometryCache>,
    asset_server: Res<'w, AssetServer>,
    /// Per-sensor viz overrides + the global full-extent toggle, read at build time to pick each
    /// sensor's EFFECTIVE drawn extent (capped vs true). Folded into this `SystemParam` so
    /// `rebuild_on_change` stays within the argument budget without a lint suppression.
    sensor_overrides: Res<'w, SensorVizOverrides>,
    sensor_global: Res<'w, SensorVizGlobal>,
    /// The in-memory asset store, consulted to translate each document uri into its generation-stamped
    /// load path ([`crate::mem_assets::MemAssetStore::asset_path`]): Bevy caches loaded assets BY PATH,
    /// so a plain-uri load after a clean document open could return the PREVIOUS document's cached
    /// bytes for a shared uri. Optional because only the bins register the store; a world without it
    /// (embedder/tests) loads plain uris, which is correct there since nothing swaps their bytes.
    mem_store: Option<Res<'w, crate::mem_assets::MemAssetStore>>,
}

/// Full scene rebuild whenever the read-only doc changes.
fn rebuild_on_change(
    doc: Res<HcdfDoc>,
    existing: Query<Entity, With<SceneItem>>,
    mut commands: Commands,
    mut assets: SceneAssets,
    mut selected: ResMut<Selected>,
    mut fitted: MessageWriter<SceneFitted>,
    mut articulated: ResMut<crate::joints::ArticulatedJoints>,
) {
    if !doc.is_changed() {
        return;
    }
    // `despawn()` is recursive in Bevy 0.19, and SceneItem is on nested entities (WorldRoot → comps →
    // visuals …), so despawning a parent already removes its children. `try_despawn` then no-ops on the
    // children the loop revisits, instead of erroring "entity invalid" (which spammed on every rebuild
    // after the first, e.g. dendrite_build republishing on each edit, or a reload here).
    for e in &existing {
        commands.entity(e).try_despawn();
    }
    selected.0 = None;
    // Reset the joint catalogue every load: it is rebuilt below from the new tree's edges. Always
    // marking it changed (even when emptied) lets the articulation system run on the fresh scene.
    articulated.0.clear();

    let Some(h) = &doc.0 else {
        return;
    };
    let convention = FrameConvention::from_hcdf(h);
    let palette = build_palette(h);
    let tree = crate::kinematics::build_kinematic_tree(h);

    // Root entity carrying the world→Bevy basis.
    let world_root = commands
        .spawn((
            SceneItem,
            WorldRoot { convention },
            convention.world.to_bevy_transform(),
            Visibility::default(),
            Name::new("WorldRoot"),
        ))
        .id();

    // For each ROOT comp, a body-basis wrapper that carries body→world; non-root comps inherit it
    // through the tree (their joint.origin poses are authored in the same body frame).
    let body_basis_t = convention.body.to_world_transform();

    // Spawn comp entities, building the parent/child hierarchy from the tree.
    let mut comp_entities: HashMap<usize, Entity> = HashMap::new();

    // Roots: child of a per-root body-basis wrapper under WorldRoot, at identity local transform.
    for &ci in &tree.roots {
        let basis = commands
            .spawn((
                SceneItem,
                body_basis_t,
                Visibility::default(),
                Name::new("BodyBasis"),
            ))
            .id();
        commands.entity(world_root).add_child(basis);
        let e = spawn_comp(&mut commands, h, ci, Transform::IDENTITY);
        commands.entity(basis).add_child(e);
        comp_entities.insert(ci, e);
    }

    // Non-root comps in dependency order: a child is spawned once its parent exists. Iterate to a
    // fixed point (tree depth is small; bounded by comp count).
    let mut remaining: Vec<&crate::kinematics::TreeEdge> = tree.edges.iter().collect();
    let mut progressed = true;
    while progressed && !remaining.is_empty() {
        progressed = false;
        remaining.retain(|edge| {
            let Some(&parent_e) = comp_entities.get(&edge.parent) else {
                return true; // parent not spawned yet; retry next pass.
            };
            let joint = &h.joint[edge.joint];
            let origin = joint
                .origin
                .as_ref()
                .map(pose_to_transform)
                .unwrap_or(Transform::IDENTITY);
            let e = spawn_comp(&mut commands, h, edge.child, origin);
            commands.entity(parent_e).add_child(e);
            comp_entities.insert(edge.child, e);
            // Catalogue this edge's joint so the articulation system can drive its child from the
            // commanded position map (the child's local Transform == joint_local_transform(origin,…)).
            articulated
                .0
                .push(crate::joints::make_articulated_joint(joint, e, origin));
            progressed = true;
            false
        });
    }
    // INVARIANT: `remaining` is empty here. build_kinematic_tree guarantees a forest (every child is
    // claimed by exactly one edge and cycles are demoted to constraints) so each edge's parent chain
    // terminates at a root spawned above, and the fixed point spawns every edge.

    // Constraint links (loops / parallel mechanisms) → markers for the kinematics display gizmo.
    for c in &tree.constraints {
        if let (Some(&p), Some(&ch)) = (comp_entities.get(&c.parent), comp_entities.get(&c.child)) {
            commands.spawn((
                SceneItem,
                ConstraintLinkMarker {
                    parent: p,
                    child: ch,
                    joint: c.joint,
                },
            ));
        }
    }

    // Spanning-tree edges → markers so the kinematics display can draw the skeleton for tree robots
    // (which have no loop constraints). Distinct from the loop constraint links above.
    for e in &tree.edges {
        if let (Some(&p), Some(&ch)) = (comp_entities.get(&e.parent), comp_entities.get(&e.child)) {
            commands.spawn((
                SceneItem,
                TreeEdgeMarker {
                    parent: p,
                    child: ch,
                },
            ));
        }
    }

    // Populate each comp's per-element children (visuals, collision, etc.).
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    // Frame `@relative-to` resolution: a `<frame>` pose is authored in the frame named by
    // `@relative-to` (a sibling frame, a comp, or a joint: the union validate::check_frames accepts),
    // yet the frame stays rigidly attached to its OWN comp. Resolve every frame to a comp-relative
    // LOCAL transform up front: sibling chains compose directly, cross-comp/joint refs compose through
    // the comps' rest-pose world placement, so the builder spawns each frame at its composed pose.
    let comp_world = comp_world_transforms(h, &tree, body_basis_t);
    let frame_locals = resolve_frame_locals(h, &comp_world);
    {
        let mut builder = SceneBuilder {
            commands: &mut commands,
            meshes: &mut assets.meshes,
            materials: &mut assets.materials,
            cache: &mut assets.cache,
            asset_server: &assets.asset_server,
            mem_store: assets.mem_store.as_deref(),
            palette: &palette,
            sensor_overrides: &assets.sensor_overrides,
            sensor_full_extent_global: assets.sensor_global.full_extent,
            frame_locals: &frame_locals,
        };
        for (&ci, &entity) in &comp_entities {
            builder.populate_comp(ci, entity, &h.comp[ci]);
            // Coarse bound from visual primitive poses (gltf bounds unknown until loaded).
            accumulate_comp_bounds(h, ci, &tree, convention, &mut min, &mut max);
        }
    }

    // Fit camera to the accumulated bounds (fallback to a small default if nothing measured).
    if min.x <= max.x {
        let center = (min + max) * 0.5;
        let radius = ((max - min) * 0.5).length().max(0.1);
        fitted.write(SceneFitted { center, radius });
    } else {
        fitted.write(SceneFitted {
            center: Vec3::ZERO,
            radius: 0.5,
        });
    }
}

/// Rest-pose world transform of each comp UNDER `WorldRoot`, mirroring the spawn placement in
/// [`rebuild_on_change`]: a root comp sits at the shared body-basis; a child rides its parent's
/// transform composed with the joint origin. Keyed by comp index. Consulted ONLY to express a
/// cross-comp (`comp`/`joint`) `@relative-to` frame reference in the referring comp's own frame (see
/// [`resolve_frame_locals`]); the sibling-frame and self-comp cases never consult it.
fn comp_world_transforms(
    h: &Hcdf,
    tree: &crate::kinematics::KinematicTree,
    body_basis: Transform,
) -> HashMap<usize, Transform> {
    let mut world: HashMap<usize, Transform> = HashMap::new();
    for &r in &tree.roots {
        world.insert(r, body_basis);
    }
    // Children in dependency order: place a child once its parent's world transform exists, the same
    // fixed-point walk `rebuild_on_change` uses to spawn the hierarchy, so the two never disagree.
    let mut remaining: Vec<&crate::kinematics::TreeEdge> = tree.edges.iter().collect();
    let mut progressed = true;
    while progressed && !remaining.is_empty() {
        progressed = false;
        remaining.retain(|e| {
            let Some(&pw) = world.get(&e.parent) else {
                return true; // parent not placed yet; retry next pass.
            };
            let origin = h.joint[e.joint]
                .origin
                .as_ref()
                .map(pose_to_transform)
                .unwrap_or(Transform::IDENTITY);
            world.insert(e.child, pw.mul_transform(origin));
            progressed = true;
            false
        });
    }
    world
}

/// Resolves each `<frame>`'s `@relative-to` to a comp-relative LOCAL transform. Bundles the name
/// lookups + comp world placement so the recursion (a frame relative to a sibling frame relative to
/// another…) stays one method call and this stays within the clippy argument budget.
struct FrameResolver<'a> {
    h: &'a Hcdf,
    /// comp name → comp index.
    comp_by_name: HashMap<&'a str, usize>,
    /// joint name → its CHILD comp index (a joint's frame is its child link frame).
    joint_child: HashMap<&'a str, usize>,
    /// comp index → rest-pose world transform (cross-comp/joint composition only).
    comp_world: &'a HashMap<usize, Transform>,
}

impl<'a> FrameResolver<'a> {
    fn new(h: &'a Hcdf, comp_world: &'a HashMap<usize, Transform>) -> Self {
        let comp_by_name: HashMap<&str, usize> = h
            .comp
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.as_str(), i))
            .collect();
        let joint_child = h
            .joint
            .iter()
            .filter_map(|j| {
                let name = j.name.as_deref()?;
                let child = j.child.as_ref().and_then(|c| c.comp.as_deref())?;
                comp_by_name.get(child).map(|&ci| (name, ci))
            })
            .collect();
        Self {
            h,
            comp_by_name,
            joint_child,
            comp_world,
        }
    }

    /// The comp-relative LOCAL transform of frame `fi` on comp `ci`, composing `@relative-to`.
    /// `visiting` holds the sibling-frame indices currently being resolved (seeded with `fi`) to break
    /// `@relative-to` cycles (A→B→A): a cyclic/self sibling ref falls back to the raw comp-relative
    /// pose with a warn, matching the general unresolvable case.
    fn local(&self, ci: usize, fi: usize, visiting: &mut Vec<usize>) -> Transform {
        let comp = &self.h.comp[ci];
        let f = &comp.frame[fi];
        let pose = f
            .pose
            .as_ref()
            .map(pose_to_transform)
            .unwrap_or(Transform::IDENTITY);
        let Some(rel) = f.relative_to.as_deref() else {
            return pose;
        };
        // Self-reference to the OWN comp: comp-relative, i.e. the raw pose (the common real case, e.g.
        // test-minimal's `relative-to="test-board"` on test-board).
        if rel == comp.name {
            return pose;
        }
        // A sibling frame on the SAME comp: compose the sibling's own resolved local, guarding cycles.
        if let Some(sfi) = comp.frame.iter().position(|sf| sf.name == rel) {
            if visiting.contains(&sfi) {
                warn!(
                    "frame {:?} on comp {:?}: relative-to {:?} is a self/cyclic frame reference, \
                     placing comp-relative",
                    f.name, comp.name, rel
                );
                return pose;
            }
            visiting.push(sfi);
            let sib = self.local(ci, sfi, visiting);
            visiting.pop();
            return sib.mul_transform(pose);
        }
        // Another comp, or a joint (its child link frame): express that element's frame relative to THIS
        // comp via their rest-pose world transforms, then compose the authored pose.
        let target = self
            .comp_by_name
            .get(rel)
            .or_else(|| self.joint_child.get(rel));
        if let Some(&refc) = target {
            if let (Some(&wc), Some(&wr)) = (self.comp_world.get(&ci), self.comp_world.get(&refc)) {
                let rel_t = GlobalTransform::from(wr).reparented_to(&GlobalTransform::from(wc));
                return rel_t.mul_transform(pose);
            }
        }
        // Unresolvable (matches no sibling frame, comp, or joint): keep the earlier behaviour, place the
        // frame comp-relative, and warn. The hcdformat validator owns ERRORING on a dangling
        // `@relative-to` (E_FRAME_RELATIVE_TO); this overlay stays lenient.
        warn!(
            "frame {:?} on comp {:?}: relative-to {:?} matches no sibling frame, comp, or joint, \
             placing comp-relative",
            f.name, comp.name, rel
        );
        pose
    }
}

/// Resolve every `<frame>`'s `@relative-to` to a comp-relative LOCAL transform, keyed
/// `(comp index, frame index)`. A frame stays rigidly attached to its own comp; `@relative-to` only
/// reinterprets which frame the authored `<pose>` is expressed in, resolved here against the union of
/// sibling frames ∪ comps ∪ joints (the namespace `validate::check_frames` accepts). Sibling chains
/// compose directly; cross-comp/joint refs compose through the rest-pose comp world placement.
fn resolve_frame_locals(
    h: &Hcdf,
    comp_world: &HashMap<usize, Transform>,
) -> HashMap<(usize, usize), Transform> {
    let resolver = FrameResolver::new(h, comp_world);
    let mut out = HashMap::new();
    for (ci, c) in h.comp.iter().enumerate() {
        for fi in 0..c.frame.len() {
            let mut visiting = vec![fi];
            out.insert((ci, fi), resolver.local(ci, fi, &mut visiting));
        }
    }
    out
}

fn spawn_comp(commands: &mut Commands, h: &Hcdf, ci: usize, local: Transform) -> Entity {
    let name = h.comp[ci].name.clone();
    commands
        .spawn((
            SceneItem,
            CompEntity {
                comp_index: ci,
                name: name.clone(),
            },
            local,
            Visibility::default(),
            Name::new(name),
        ))
        .id()
}

/// Borrows the spawn-side resources so the per-element helpers take one `&mut self` instead of a long
/// argument list (mirrors dendrite-viewer's `SystemParam` grouping; idiomatic and removes the
/// `too_many_arguments` lint without suppressing it).
struct SceneBuilder<'a, 'w, 's> {
    commands: &'a mut Commands<'w, 's>,
    meshes: &'a mut Assets<Mesh>,
    materials: &'a mut Assets<StandardMaterial>,
    cache: &'a mut GeometryCache,
    asset_server: &'a AssetServer,
    /// Translates document uris to their generation-stamped load paths (see [`SceneAssets`]); absent
    /// in worlds that never registered the store, where plain uris load directly.
    mem_store: Option<&'a crate::mem_assets::MemAssetStore>,
    palette: &'a HashMap<String, [f32; 4]>,
    /// Per-sensor viz overrides, read for each sensor's per-sensor full-extent flag.
    sensor_overrides: &'a SensorVizOverrides,
    /// The global full-extent toggle; ORed with the per-sensor flag to pick the drawn extent.
    sensor_full_extent_global: bool,
    /// Resolved comp-relative LOCAL transform for every `<frame>`, keyed `(comp index, frame index)`
    /// ([`resolve_frame_locals`]): folds `@relative-to` composition in up front so `populate_comp`
    /// spawns each frame at its composed pose rather than the raw comp-relative `<pose>`.
    frame_locals: &'a HashMap<(usize, usize), Transform>,
}

impl SceneBuilder<'_, '_, '_> {
    /// The EFFECTIVE full-extent for a sensor: the global toggle OR this sensor's per-sensor override.
    /// When true the FOV/lidar meshes resolve at their TRUE (uncapped) extent; otherwise the display cap.
    fn sensor_full_extent(&self, comp: &str, sensor: &str) -> bool {
        self.sensor_full_extent_global || self.sensor_overrides.full_extent(comp, sensor)
    }

    /// The path to load a document asset uri under: the in-memory store's generation-stamped
    /// translation when the store serves the uri, the plain uri otherwise (native disk fallback, or a
    /// world without the store). Every `asset_server.load` of a document uri MUST go through this: Bevy
    /// caches loaded assets by path, so a plain-uri load could return a PREVIOUS document's cached
    /// bytes for a shared uri after a clean open.
    fn load_path(&self, uri: &str) -> String {
        match self.mem_store {
            Some(store) => store.asset_path(uri),
            None => uri.to_string(),
        }
    }
}

impl SceneBuilder<'_, '_, '_> {
    /// Spawn the structural and sensor children of a component.
    /// `ci` is the comp's index into `hcdf.comp`, used to look up its frames' resolved local transforms.
    fn populate_comp(&mut self, ci: usize, parent: Entity, comp: &Comp) {
        // VISUALS: ARM A (model/glTF) or ARM B (primitive + color).
        for v in &comp.visual {
            let pose = v
                .pose
                .as_ref()
                .map(pose_to_transform)
                .unwrap_or(Transform::IDENTITY);
            match &v.appearance {
                VisualAppearance::Model { model, geometry } => {
                    if let Some(uri) = model.uri.as_deref() {
                        let gltf = self
                            .commands
                            .spawn((
                                SceneItem,
                                VisualItem,
                                OwnerComp(parent),
                                WorldAssetRoot(self.asset_server.load(
                                    GltfAssetLabel::Scene(0).from_asset(self.load_path(uri)),
                                )),
                                pose,
                                // Explicit (default Visible/Inherited) so the visual display can flip
                                // this subtree root to Hidden: inherited visibility hides the whole
                                // async-spawned glTF mesh tree below it (verified: WorldAssetRoot
                                // spawns the scene as ChildOf this entity, children stay Inherited).
                                Visibility::default(),
                                Name::new(format!("visual:{}", v.name)),
                            ))
                            .id();
                        // `toggle="…"` group membership (legacy show/hide groups): only when present.
                        if let Some(group) = &v.toggle {
                            self.commands
                                .entity(gltf)
                                .insert(ToggleGroup(group.clone()));
                        }
                        // Visual submesh selection: stamp the include/exclude filter ONLY when the model
                        // carries selectors (a selector-less model gets nothing, so its spawn stays
                        // byte-identical). `apply_submesh_visibility` reads it once the glTF materializes.
                        if let Some(filter) = model_submesh_filter(model) {
                            self.commands.entity(gltf).insert(filter);
                        }
                        self.commands.entity(parent).add_child(gltf);
                    } else if let Some(g) = geometry {
                        // Model arm with no uri but a fallback primitive: render the primitive.
                        self.spawn_primitive_visual(
                            parent,
                            g,
                            None,
                            pose,
                            &v.name,
                            v.toggle.as_deref(),
                        );
                    }
                }
                VisualAppearance::Primitive { geometry, color } => {
                    if let Some(g) = geometry {
                        self.spawn_primitive_visual(
                            parent,
                            g,
                            color.as_ref(),
                            pose,
                            &v.name,
                            v.toggle.as_deref(),
                        );
                    }
                }
            }
        }

        // COLLISION: primitive/mesh overlay (default hidden; toggled by collision display). A primitive
        // resolves to a cached mesh with the translucent collision tint; a `<mesh uri>` dispatches by
        // extension: glTF/GLB via the SCENE loader (exactly like visuals), `.stl` via our Mesh loader,
        // anything else (or an absent uri) falls back to a LOUD translucent bounds box + a console warn:
        // no more silently-missing collision geometry (the Perseverance `.gltf`/`.glb` collision case).
        //
        // Note: a comp with a `<collision>` but NO `<visual>` (openarm-thor's drive comps) renders this
        // collision overlay AS its body: the saturated teal is the shared collision tint on the only
        // geometry the comp has, NOT a highlight, bleed, or per-frame churn. Correct, doc-truthful; the
        // extra apparent saturation is just overlapping translucent front/back faces.
        for col in &comp.collision {
            let pose = col
                .pose
                .as_ref()
                .map(pose_to_transform)
                .unwrap_or(Transform::IDENTITY);
            let Some(g) = &col.geometry else { continue };
            let cname = col.name.clone().unwrap_or_default();
            let name = Name::new(format!("collision:{cname}"));
            let mat = self.cache.material([0.2, 0.8, 0.9, 0.25], self.materials);

            let child = if let Some(rm) = resolve_collision_geometry(g, self.cache, self.meshes) {
                // Primitive collision (box/cylinder/sphere/…): cached primitive mesh + collision tint.
                self.spawn_collision_mesh(
                    parent,
                    rm.mesh,
                    mat,
                    with_extra_scale(pose, rm.scale),
                    name,
                )
            } else if let Some(uri) = g.mesh.as_ref().and_then(|m| m.uri.as_deref()) {
                let scale = parse_mesh_scale(g.mesh.as_ref().and_then(|m| m.scale.as_deref()));
                let t = with_extra_scale(pose, scale);
                match collision_mesh_kind(uri) {
                    // glTF/GLB: the SCENE loader (the visuals' pattern): a `WorldAssetRoot` under
                    // `CollisionItem`, so the whole async-spawned subtree toggles/isolates with the
                    // Collision display (children stay `Inherited`; flipping this root hides them all).
                    // The translucent collision tint applies only to the primitive/STL overlays; a scene
                    // mesh carries its OWN baked PBR materials (accepted, not re-tinted: a subtree walk to
                    // recolor every child material is deliberately out of scope here).
                    CollisionMeshKind::Scene => {
                        let scene_root = self
                            .commands
                            .spawn((
                                SceneItem,
                                CollisionItem,
                                OwnerComp(parent),
                                WorldAssetRoot(self.asset_server.load(
                                    GltfAssetLabel::Scene(0).from_asset(self.load_path(uri)),
                                )),
                                t,
                                Visibility::Hidden,
                                name,
                            ))
                            .id();
                        // Honor the collision single `<submesh name center>` (previously ignored): a
                        // lone include filters the loaded scene to that subtree, with optional `@center`.
                        if let Some(filter) = g
                            .mesh
                            .as_ref()
                            .and_then(|m| m.submesh.as_ref())
                            .and_then(collision_submesh_filter)
                        {
                            self.commands.entity(scene_root).insert(filter);
                        }
                        scene_root
                    }
                    // STL: our Mesh asset loader, tinted with the translucent collision material.
                    CollisionMeshKind::Stl => {
                        let handle: Handle<Mesh> = self.asset_server.load(self.load_path(uri));
                        self.spawn_collision_mesh(parent, handle, mat, t, name)
                    }
                    // Unsupported extension: no loader turns this into geometry, so restore a LOUD fallback
                    // (a translucent unit bounds box at the collision pose + a console warn) instead of the
                    // silent nothing this path used to produce.
                    CollisionMeshKind::Unsupported => {
                        warn!(
                            "collision mesh {uri:?} on comp {:?}: unsupported extension (only \
                             .glb/.gltf/.stl collision meshes load): drawing a fallback bounds box",
                            comp.name
                        );
                        let cube = self.cache_box(Vec3::ONE);
                        self.spawn_collision_mesh(parent, cube, mat, t, name)
                    }
                }
            } else {
                continue;
            };
            self.commands.entity(parent).add_child(child);
        }

        // INERTIAL: CG marker at inertia_origin (default hidden).
        if let Some(inertial) = &comp.inertial {
            let pose = inertial
                .inertia_origin
                .as_ref()
                .map(pose_to_transform)
                .unwrap_or(Transform::IDENTITY);
            let mesh = self.cache_sphere(0.01);
            let mat = self.cache.material([1.0, 0.85, 0.1, 0.9], self.materials);
            let child = self
                .commands
                .spawn((
                    SceneItem,
                    InertialMarker,
                    OwnerComp(parent),
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    pose,
                    Visibility::Hidden,
                    Name::new("inertial:cg"),
                ))
                .id();
            self.commands.entity(parent).add_child(child);
        }

        // FRAMES: gizmo triad markers (default hidden; drawn by frames display gizmo). The pose is the
        // `@relative-to`-resolved comp-relative LOCAL transform (see [`resolve_frame_locals`]); a frame
        // with no (or an unresolvable) `@relative-to` falls back to its raw comp-relative `<pose>`.
        for (fi, f) in comp.frame.iter().enumerate() {
            let pose = self
                .frame_locals
                .get(&(ci, fi))
                .copied()
                .unwrap_or_else(|| {
                    f.pose
                        .as_ref()
                        .map(pose_to_transform)
                        .unwrap_or(Transform::IDENTITY)
                });
            let optical = f.type_.as_deref() == Some("optical");
            let child = self
                .commands
                .spawn((
                    SceneItem,
                    FrameMarker {
                        label: f.name.clone(),
                        optical,
                        type_label: frame_type_label(f.type_.as_deref()),
                    },
                    OwnerComp(parent),
                    pose,
                    Visibility::Hidden,
                    Name::new(format!("frame:{}", f.name)),
                ))
                .id();
            self.commands.entity(parent).add_child(child);
        }

        // SENSORS: pose markers + FOV frustums.
        for s in &comp.sensor {
            let sname = s.name.clone().unwrap_or_default();
            // Effective full-extent for THIS sensor (global toggle OR its per-sensor override): drives
            // whether the FOV/lidar meshes resolve capped or at their true extent.
            let full_extent = self.sensor_full_extent(&comp.name, &sname);
            for opt in &s.optical {
                let spose = opt
                    .pose
                    .as_ref()
                    .map(pose_to_transform)
                    .unwrap_or(Transform::IDENTITY);
                self.spawn_sensor_node(parent, &sname, spose, opt.driver.as_ref());
                for fov in &opt.fov {
                    if let Some(g) = &fov.geometry {
                        let fov_local = fov
                            .pose
                            .as_ref()
                            .map(pose_to_transform)
                            .unwrap_or(Transform::IDENTITY);
                        self.spawn_fov(
                            parent,
                            g,
                            spose * fov_local,
                            fov.color.as_deref(),
                            &sname,
                            full_extent,
                        );
                    }
                }
                // A lidar/ray sensor's angular sweep (typed <lidar-params><scan-pattern>) becomes a
                // lightweight scan wireframe at the sensor pose: the FoV analogue for a ranging sensor.
                if let Some(lp) = &opt.lidar_params {
                    self.spawn_lidar_scan(parent, lp, spose, &sname, full_extent);
                }
            }
            for cat in
                s.em.iter()
                    .chain(&s.rf)
                    .chain(&s.chemical)
                    .chain(&s.force)
                    .chain(&s.encoder)
                    .chain(&s.temperature)
                    .chain(&s.radiation)
                    .chain(&s.audio)
                    .chain(&s.tactile)
            {
                let spose = cat
                    .pose
                    .as_ref()
                    .map(pose_to_transform)
                    .unwrap_or(Transform::IDENTITY);
                self.spawn_sensor_node(parent, &sname, spose, cat.driver.as_ref());
                if let Some(g) = &cat.geometry {
                    self.spawn_fov(parent, g, spose, None, &sname, full_extent);
                }
            }
            for ins in &s.inertial {
                let spose = ins
                    .pose
                    .as_ref()
                    .map(pose_to_transform)
                    .unwrap_or(Transform::IDENTITY);
                self.spawn_sensor_node(parent, &sname, spose, ins.driver.as_ref());
            }
            // FLUID (barometer/airspeed/depth/flow/…): pressure/flow measurands have no field of view,
            // so (exactly like the em/rf/force categories) they get a labeled pose triad, plus a glyph
            // mesh only when the sensor authored explicit `<geometry>`. `FluidSensor` is its own struct
            // (not `SensorCategory`), so it can't join the chain above and needs its own loop.
            for fl in &s.fluid {
                let spose = fl
                    .pose
                    .as_ref()
                    .map(pose_to_transform)
                    .unwrap_or(Transform::IDENTITY);
                self.spawn_sensor_node(parent, &sname, spose, fl.driver.as_ref());
                if let Some(g) = &fl.geometry {
                    self.spawn_fov(parent, g, spose, None, &sname, full_extent);
                }
            }
        }
    }

    fn spawn_primitive_visual(
        &mut self,
        parent: Entity,
        g: &crate::schema::model::geometry::VisualGeometry,
        inline_color: Option<&SColor>,
        pose: Transform,
        name: &str,
        toggle: Option<&str>,
    ) {
        if let Some(rm) = resolve_visual_geometry(g, self.cache, self.meshes) {
            let rgba = resolve_color(inline_color, self.palette);
            let mat = self.cache.material(rgba, self.materials);
            let t = with_extra_scale(pose, rm.scale);
            let child = self
                .commands
                .spawn((
                    SceneItem,
                    VisualItem,
                    OwnerComp(parent),
                    Mesh3d(rm.mesh),
                    MeshMaterial3d(mat),
                    t,
                    Visibility::default(),
                    Name::new(format!("visual:{name}")),
                ))
                .id();
            // `toggle="…"` group membership (legacy show/hide groups): only when present.
            if let Some(group) = toggle {
                self.commands
                    .entity(child)
                    .insert(ToggleGroup(group.to_string()));
            }
            self.commands.entity(parent).add_child(child);
        }
    }

    fn spawn_sensor_node(
        &mut self,
        parent: Entity,
        name: &str,
        pose: Transform,
        driver: Option<&SensorDriver>,
    ) {
        let mut node = self.commands.spawn((
            SceneItem,
            SensorMarker {
                label: name.to_string(),
            },
            OwnerComp(parent),
            pose,
            Visibility::Hidden,
            Name::new(format!("sensor:{name}")),
        ));
        // A `<driver><axis-align>` remap gets an aligned-axes marker so the sensor axis-align display
        // can draw the post-remap triad next to the raw one (legacy semantics: the mapped directions
        // are drawn in the sensor-pose frame, which IS the comp body frame for the usual
        // identity-rotation sensor pose).
        if let Some(align) = driver.and_then(|d| d.axis_align.as_ref()) {
            node.insert(AlignedAxesMarker(axis_align_rotation(align)));
        }
        let child = node.id();
        self.commands.entity(parent).add_child(child);
    }

    /// `base` is the already-composed sensor·fov placement transform (the caller folds any `<fov><pose>`
    /// into the sensor pose), keeping the arg count within the clippy budget without a lint suppression.
    fn spawn_fov(
        &mut self,
        parent: Entity,
        g: &crate::schema::model::geometry::Geometry,
        base: Transform,
        color: Option<&str>,
        name: &str,
        full_extent: bool,
    ) {
        let rgba = color.and_then(parse_rgba).unwrap_or([0.3, 0.8, 1.0, 0.15]);
        // Frustum geometry → cached custom mesh; otherwise fall back to a general primitive (e.g. cone).
        // The frustum mesh is authored in the OPTICAL frame (+Z forward), but the sensor `<pose>` places
        // the sensor BODY frame (+X forward), so it is reoriented optical→body to project out the front.
        // General-primitive glyphs (em/rf/fluid markers) have no viewing axis and are placed as authored.
        let (mesh, t) =
            if let Some(mesh) = resolve_frustum_mesh(g, full_extent, self.cache, self.meshes) {
                (
                    mesh,
                    base * Transform::from_rotation(frustum_optical_to_body_rotation()),
                )
            } else if let Some(rm) = resolve_general_primitive(g, self.cache, self.meshes) {
                (rm.mesh, with_extra_scale(base, rm.scale))
            } else {
                return;
            };
        let mat = self.cache.material_unlit(rgba, self.materials);
        let child = self
            .commands
            .spawn((
                SceneItem,
                SensorMarker {
                    label: name.to_string(),
                },
                OwnerComp(parent),
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                t,
                Visibility::Hidden,
                Name::new(format!("fov:{name}")),
            ))
            .id();
        self.commands.entity(parent).add_child(child);
    }

    /// Lidar/ray scan-extent viz (translucent filled annulus/sector/band + a thin boundary ring) at the
    /// sensor pose, built from the typed `<lidar-params>`. Two sibling `SensorMarker` + `Mesh3d` entities
    /// (fill + ring) so both ride the exact same sensor-visibility toggle as the camera FOV frustums
    /// (`sync_sensor_visibility`). `full_extent` draws the true range; otherwise the display cap.
    fn spawn_lidar_scan(
        &mut self,
        parent: Entity,
        params: &LidarParams,
        sensor_pose: Transform,
        name: &str,
        full_extent: bool,
    ) {
        let Some(meshes) = resolve_lidar_scan_mesh(params, full_extent, self.cache, self.meshes)
        else {
            return;
        };
        // Subdued translucent fill (FOV-frustum-like) + a slightly stronger opaque-ish boundary ring.
        let fill_mat = self
            .cache
            .material_unlit([0.3, 0.8, 1.0, 0.20], self.materials);
        let ring_mat = self
            .cache
            .material_unlit([0.3, 0.8, 1.0, 0.8], self.materials);
        for (mesh, mat, tag) in [
            (meshes.fill, fill_mat, "scan"),
            (meshes.boundary, ring_mat, "scan-ring"),
        ] {
            let child = self
                .commands
                .spawn((
                    SceneItem,
                    SensorMarker {
                        label: name.to_string(),
                    },
                    OwnerComp(parent),
                    Mesh3d(mesh),
                    MeshMaterial3d(mat),
                    sensor_pose,
                    Visibility::Hidden,
                    Name::new(format!("{tag}:{name}")),
                ))
                .id();
            self.commands.entity(parent).add_child(child);
        }
    }

    /// Spawn one Mesh-backed collision overlay item (primitive, STL, or fallback box): a `CollisionItem`
    /// carrying the given mesh + translucent collision material at `t`, hidden until the Collision display
    /// enables it. Returns the entity so the caller parents it under its comp.
    fn spawn_collision_mesh(
        &mut self,
        parent: Entity,
        mesh: Handle<Mesh>,
        mat: Handle<StandardMaterial>,
        t: Transform,
        name: Name,
    ) -> Entity {
        self.commands
            .spawn((
                SceneItem,
                CollisionItem,
                OwnerComp(parent),
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                t,
                Visibility::Hidden,
                name,
            ))
            .id()
    }

    fn cache_sphere(&mut self, r: f32) -> Handle<Mesh> {
        use crate::schema::model::geometry::{Sphere as SSphere, VisualGeometry};
        let g = VisualGeometry {
            sphere: Some(SSphere {
                radius: Some(r.to_string()),
            }),
            ..default()
        };
        resolve_visual_geometry(&g, self.cache, self.meshes)
            .map(|rm| rm.mesh)
            .unwrap_or_else(|| self.meshes.add(Sphere::new(r)))
    }

    fn cache_box(&mut self, size: Vec3) -> Handle<Mesh> {
        use crate::schema::model::geometry::{Box_, VisualGeometry};
        let g = VisualGeometry {
            box_: Some(Box_ {
                size: Some(format!("{} {} {}", size.x, size.y, size.z)),
            }),
            ..default()
        };
        resolve_visual_geometry(&g, self.cache, self.meshes)
            .map(|rm| rm.mesh)
            .unwrap_or_else(|| self.meshes.add(Cuboid::new(size.x, size.y, size.z)))
    }
}

/// How a collision `<mesh uri>` resolves to something spawnable, dispatched on the URI's extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionMeshKind {
    /// `.glb`/`.gltf`: the Bevy SCENE loader (a whole node tree), exactly like a visual model.
    Scene,
    /// `.stl`: hcdviz's STL → [`Mesh`] asset loader (a single mesh, translucent-tinted).
    Stl,
    /// Anything else: no loader handles it; the caller draws a LOUD fallback bounds box.
    Unsupported,
}

/// Classify a collision mesh URI by its (case-insensitive) file extension. glTF/GLB load via the scene
/// loader (their only Mesh path: the STL loader is the ONLY `Handle<Mesh>` loader, so a glTF asked for
/// as a `Handle<Mesh>` would never resolve); `.stl` keeps the Mesh path; everything else falls back.
pub fn collision_mesh_kind(uri: &str) -> CollisionMeshKind {
    let base = uri
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(uri)
        .to_ascii_lowercase();
    if base.ends_with(".glb") || base.ends_with(".gltf") {
        CollisionMeshKind::Scene
    } else if base.ends_with(".stl") {
        CollisionMeshKind::Stl
    } else {
        CollisionMeshKind::Unsupported
    }
}

/// Parse a mesh `@scale` ("sx sy sz", or a single uniform value) to a [`Vec3`]; default `ONE`.
fn parse_mesh_scale(s: Option<&str>) -> Vec3 {
    match s {
        Some(t) => {
            let v = crate::geometry::parse_floats(t);
            match v.len() {
                1 => Vec3::splat(v[0]),
                3 => Vec3::new(v[0], v[1], v[2]),
                _ => Vec3::ONE,
            }
        }
        None => Vec3::ONE,
    }
}

/// Accumulate a coarse world-space bound for the camera fit from a comp's visual primitive sizes,
/// placed at the comp's accumulated joint translation (rotation ignored for the coarse bound).
fn accumulate_comp_bounds(
    h: &Hcdf,
    ci: usize,
    tree: &crate::kinematics::KinematicTree,
    convention: FrameConvention,
    min: &mut Vec3,
    max: &mut Vec3,
) {
    // Accumulate translation up the tree to a world-ish position.
    let mut pos = Vec3::ZERO;
    let mut cur = ci;
    let mut guard = 0;
    while let Some(&edge_idx) = tree.primary_edge_of.get(&cur) {
        let edge = &tree.edges[edge_idx];
        if let Some(o) = &h.joint[edge.joint].origin {
            let oxyz = o.xyz_or_zero();
            pos += Vec3::new(oxyz[0] as f32, oxyz[1] as f32, oxyz[2] as f32);
        }
        cur = edge.parent;
        guard += 1;
        if guard > h.comp.len() {
            break;
        }
    }
    let world_pos = convention.body_point_to_bevy(pos);
    let half = comp_extent(&h.comp[ci]).max(0.02);
    *min = min.min(world_pos - Vec3::splat(half));
    *max = max.max(world_pos + Vec3::splat(half));
}

/// Rough half-extent of a comp from its first visual primitive (meters), for camera framing.
fn comp_extent(comp: &Comp) -> f32 {
    for v in &comp.visual {
        if let VisualAppearance::Primitive {
            geometry: Some(g), ..
        }
        | VisualAppearance::Model {
            geometry: Some(g), ..
        } = &v.appearance
        {
            if let Some(b) = &g.box_ {
                let f = crate::geometry::parse_floats(b.size.as_deref().unwrap_or(""));
                if f.len() == 3 {
                    return f[0].max(f[1]).max(f[2]) * 0.5;
                }
            }
            if let Some(c) = &g.cylinder {
                return c
                    .length
                    .as_deref()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.05)
                    * 0.5;
            }
            if let Some(s) = &g.sphere {
                return s
                    .radius
                    .as_deref()
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.05);
            }
        }
    }
    0.05
}

// ── built-in displays ────────────────────────────────────────────────────────

/// Shared run condition for the `Visibility`-writing sync systems: the visibility they wrote last time
/// stays correct until a display/selection input changes or new items spawn (`Added<T>`: a doc
/// rebuild), so quiet frames skip the full query walk. Consults the live [`DisplayRegistry`] (seeded
/// from each display's `default_enabled()`), so no assumption about which kinds start enabled is baked
/// in here.
fn vis_inputs_changed<T: Component>(
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    added: Query<(), Added<T>>,
) -> bool {
    registry.is_changed()
        || isolation.is_changed()
        || selected.is_changed()
        || overrides.is_changed()
        || !added.is_empty()
}

/// The isolation inputs bundled into one `SystemParam` so both isolate modes travel together (and the
/// per-frame draw systems stay within the argument budget): the single-selection [`IsolateSelection`]
/// flag plus the comp-set [`IsolateSet`] that supersedes it when populated. Every isolation decision
/// goes through [`isolate_hides`]/[`isolate_hides_edge`], so the single- vs set-mode branch lives in one
/// place.
#[derive(bevy::ecs::system::SystemParam)]
pub struct Isolation<'w> {
    isolate: Res<'w, IsolateSelection>,
    set: Res<'w, IsolateSet>,
}

impl Isolation<'_> {
    /// Visibility for an owned item, folding the kind toggle with the active isolation mode.
    fn visibility(
        &self,
        kind_enabled: bool,
        selected: Option<Entity>,
        owner: Entity,
    ) -> Visibility {
        item_visibility(
            kind_enabled,
            self.isolate.0,
            selected,
            owner,
            self.set.0.as_ref(),
        )
    }

    /// Whether a gizmo owned by `owner` is isolated away (immediate-mode draws check this then `continue`).
    fn hides(&self, selected: Option<Entity>, owner: Entity) -> bool {
        isolate_hides(self.isolate.0, selected, self.set.0.as_ref(), owner)
    }

    /// Whether a kinematic tree edge is isolated away (both-endpoints rule under comp-set isolate).
    fn edge_hides(&self, selected: Option<Entity>, parent: Entity, child: Entity) -> bool {
        isolate_hides_edge(self.isolate.0, selected, self.set.0.as_ref(), parent, child)
    }

    /// Change tick for the run conditions: either isolate input flipping must re-run the sync systems.
    fn is_changed(&self) -> bool {
        self.isolate.is_changed() || self.set.is_changed()
    }
}

/// The per-sensor override lookup bundled into one `SystemParam` (so the sensor-visibility systems stay
/// within the argument budget): resolves an item's owner comp entity to its name, then asks
/// [`SensorVizOverrides`]. Mirrors [`Isolation`]. An unresolvable owner defaults to shown, preserving the
/// pre-override behaviour.
#[derive(bevy::ecs::system::SystemParam)]
pub struct SensorViz<'w, 's> {
    overrides: Res<'w, SensorVizOverrides>,
    comps: Query<'w, 's, &'static CompEntity>,
}

impl SensorViz<'_, '_> {
    /// Whether the sensor `label` owned by `owner` should show its viz per its per-sensor override,
    /// BEFORE the global Sensors toggle / isolate rules apply. Default on; unresolvable owner ⇒ on.
    fn allows(&self, owner: Entity, label: &str) -> bool {
        self.comps
            .get(owner)
            .map_or(true, |c| self.overrides.visible(&c.name, label))
    }
}

/// Run condition for the immediate-mode gizmo draws (frame/sensor triads): gizmos must be re-submitted
/// every frame while anything shows, so they cannot be change-gated like the sync systems above. The
/// safe skip is "nothing can pass [`effective_kind_enabled`] for this kind": the global toggle is off
/// AND no per-link override forces it on (overrides only ever apply to the selected comp).
pub(crate) fn kind_drawable(
    id: &'static str,
) -> impl Fn(Res<DisplayRegistry>, Res<SelectionOverrides>) -> bool + Clone {
    move |registry: Res<DisplayRegistry>, overrides: Res<SelectionOverrides>| {
        registry.enabled(id) || overrides.kinds.get(id).copied().unwrap_or(false)
    }
}

/// VisualDisplay: visuals are spawned meshes/glTF; this display toggles their visibility. Flipping the
/// `VisualItem` root entity's `Visibility` propagates through inherited visibility to the whole glTF
/// subtree (the async-spawned meshes are `ChildOf` the `WorldAssetRoot` entity and stay `Inherited`).
pub struct VisualDisplay;
impl Display for VisualDisplay {
    fn id(&self) -> &'static str {
        ID_VISUAL
    }
    fn label(&self) -> &str {
        "Visual"
    }
    fn build(&self, app: &mut App) {
        // No display_enabled gate: the global kind-toggle and the isolate/selection state are folded
        // into the per-item decision by `item_visibility`, so isolate mode can hide non-selected items
        // while the kind stays globally enabled. `vis_inputs_changed` keeps quiet frames free instead;
        // the toggle-group hide set is one more input, so its change tick is ORed in (idempotent
        // `init_resource` so this display also works standalone, without ScenePlugin).
        app.init_resource::<VisualToggleGroups>().add_systems(
            Update,
            sync_visual_visibility.run_if(
                vis_inputs_changed::<VisualItem>.or_else(resource_changed::<VisualToggleGroups>),
            ),
        );
    }
}

fn sync_visual_visibility(
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    groups: Res<VisualToggleGroups>,
    mut q: Query<(&OwnerComp, Option<&ToggleGroup>, &mut Visibility), With<VisualItem>>,
) {
    let global = registry.enabled(ID_VISUAL);
    let override_kind = overrides.kinds.get(ID_VISUAL).copied();
    for (owner, group, mut v) in &mut q {
        // The selected comp honors its per-link override for this kind; every other comp follows
        // global. A hidden toggle group then hides its members no matter what enabled the kind.
        let enabled = effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0))
            && toggle_group_visible(group.map(|g| g.0.as_str()), &groups.hidden);
        let want = isolation.visibility(enabled, selected.0, owner.0);
        if *v != want {
            *v = want; // write only on change to avoid change-detection churn
        }
    }
}

// ── visual submesh selection ──────────────────────────────────────────────
//
// A `<model uri sha>` may carry selector children that choose WHICH subtrees of the loaded glTF this
// visual draws (schema): none = the whole model (backward compatible); `<submesh name>`* = the UNION
// of the named subtrees (include mode); `<exclude-submesh name>`* = the whole model MINUS them (exclude
// mode). The COLLISION side already had a single-`<submesh name center>` selector that the viewer used
// to ignore silently; it is honored here through the SAME machinery (a lone include, optionally
// recentered per SDF `@center`).
//
// Realization: we still spawn the whole glTF scene under the `WorldAssetRoot` holder (byte-identical
// spawn), then VISIBILITY-FILTER the async-materialized node tree by NAME, hiding the complement for
// include mode, hiding the named subtrees for exclude mode. The filtered nodes are the glTF child nodes
// (which carry NO `VisualItem`/`CollisionItem` marker), so `sync_visual_visibility` (which only writes
// the holder) never fights them: a node left `Inherited` follows the holder's display toggle, a node we
// set `Hidden` stays hidden regardless. We NEVER set a node `Visibility::Visible` (that would override a
// toggled-off holder and leak the geometry), so the visual/collision toggle keeps working unchanged.

/// The visibility filter a submesh selection imposes on a spawned glTF scene, stamped on the
/// [`VisualItem`]/[`CollisionItem`] holder at spawn time ONLY when the model carries selectors. A
/// selector-less `<model>` gets NO such component, so its spawn and render are byte-identical to before
/// (the entire existing corpus is untouched; the whole-model path is unchanged).
#[derive(Component, Debug, Clone, PartialEq)]
pub enum SubmeshFilter {
    /// Include mode: draw ONLY the union of these named subtrees; every other node is hidden. `center`
    /// is the lone-include `@center=true` recenter request (SDF parity, set only when there is exactly
    /// ONE name that authored `@center` truthy); it drives [`apply_submesh_center`]. The visual side
    /// never sets it (selections keep their model-root-relative transforms per spec); the
    /// collision single-submesh does, matching SDF `<submesh><center>` semantics.
    Include { names: Vec<String>, center: bool },
    /// Exclude mode: hide these named subtrees, draw everything else, including nodes that spawn LATER
    /// (the plan is re-derived idempotently as the async glTF tree grows).
    Exclude { names: Vec<String> },
}

/// Marks a holder whose submesh selector named a node NOT present in its glTF, so the loud
/// unresolvable-selector `warn!` fires exactly once (not every re-derive frame). Fail-visible: while
/// unresolved, the model renders WHOLE, never blank.
#[derive(Component)]
struct SubmeshWarned;

/// Marks a lone-include holder already recentered by [`apply_submesh_center`], so the one-shot `@center`
/// translation is applied exactly once (idempotent across the frames the async glTF keeps spawning).
#[derive(Component)]
struct SubmeshCentered;

/// A trimmed, non-empty `@name`/`@center` string, or `None` (a nameless/blank selector is meaningless,
/// nothing to resolve, and must NOT collapse an include to "hide everything").
fn clean_str(v: Option<&str>) -> Option<String> {
    v.map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// SDF-style boolean truthiness for `@center` (accepts the usual affirmative tokens; everything else,
/// including absent, is false).
fn center_truthy(v: Option<&str>) -> bool {
    matches!(
        clean_str(v).map(|s| s.to_ascii_lowercase()).as_deref(),
        Some("true" | "1" | "yes" | "on")
    )
}

/// The [`SubmeshFilter`] a `<model>`'s selectors resolve to, or `None` for a selector-less model (the
/// unchanged whole-model path: no component attached, spawn stays byte-identical). Include and exclude
/// are mutually exclusive per the schema/validator; if a malformed doc carries BOTH, include wins here
/// (the validator already flags the mix; "show only" is the conservative reading). `@center` is honored
/// only on a LONE include (SDF parity); multiple includes never recenter.
fn model_submesh_filter(model: &crate::schema::model::ModelRef) -> Option<SubmeshFilter> {
    let include: Vec<String> = model
        .submesh
        .iter()
        .filter_map(|s| clean_str(s.name.as_deref()))
        .collect();
    if !include.is_empty() {
        let center = include.len() == 1
            && model
                .submesh
                .iter()
                .find(|s| clean_str(s.name.as_deref()).is_some())
                .is_some_and(|s| center_truthy(s.center.as_deref()));
        return Some(SubmeshFilter::Include {
            names: include,
            center,
        });
    }
    let exclude: Vec<String> = model
        .exclude_submesh
        .iter()
        .filter_map(|s| clean_str(s.name.as_deref()))
        .collect();
    (!exclude.is_empty()).then_some(SubmeshFilter::Exclude { names: exclude })
}

/// The [`SubmeshFilter`] a collision `<mesh>`'s single `<submesh name center>` resolves to (a lone
/// include, honoring `@center` with SDF semantics), or `None` when it is nameless/absent: the
/// previously-silently-ignored collision selector, now honored through the same visibility machinery.
fn collision_submesh_filter(submesh: &crate::schema::model::Submesh) -> Option<SubmeshFilter> {
    let name = clean_str(submesh.name.as_deref())?;
    Some(SubmeshFilter::Include {
        names: vec![name],
        center: center_truthy(submesh.center.as_deref()),
    })
}

/// The visibility plan for one submesh-filtered holder: which descendant node entities to HIDE, the full
/// set walked (so the apply resets the rest to `Inherited`, staying idempotent as the tree grows), and
/// any selector names that matched NO node (the fail-visible warn set).
struct SubmeshPlan {
    /// Descendant node entities to set `Visibility::Hidden`.
    hide: Vec<Entity>,
    /// Every descendant node entity walked under the holder (the holder itself is excluded).
    all: Vec<Entity>,
    /// Selector names that resolved to no node. When non-empty, `hide` is EMPTY (whole model): an
    /// unresolvable name shows everything rather than hiding the model (fail-visible, not fail-invisible).
    unresolved: Vec<String>,
}

/// Compute the [`SubmeshPlan`] for a holder's spawned glTF subtree. `holder_children` are the holder's
/// direct children (the scene's root node entities); `name_of`/`children_of` expose the node tree as
/// closures so this is unit-testable headless (no World/GPU): the same subtree-walk idiom as
/// [`crate::connector::resolve_connector_meshes`].
///
/// Include mode KEEPS a node iff it is a named target, a DESCENDANT of one (the drawn geometry), or an
/// ANCESTOR of one (the structural path that must stay visible for its descendants to render); the rest
/// is hidden. Exclude mode hides each named node (its descendants inherit the hidden state). A single
/// unresolved name trips fail-visible: `hide` comes back empty so the whole model shows.
fn submesh_plan(
    holder_children: &[Entity],
    filter: &SubmeshFilter,
    name_of: impl Fn(Entity) -> Option<String>,
    children_of: impl Fn(Entity) -> Vec<Entity>,
) -> SubmeshPlan {
    // One DFS of the holder subtree: record every node, its within-subtree parent, its name, and its
    // children (so include mode can walk ancestors/descendants without re-invoking `children_of`).
    let mut all: Vec<Entity> = Vec::new();
    let mut parent_of: HashMap<Entity, Entity> = HashMap::new();
    let mut child_map: HashMap<Entity, Vec<Entity>> = HashMap::new();
    let mut node_name: HashMap<Entity, String> = HashMap::new();
    let mut stack: Vec<Entity> = holder_children.to_vec();
    while let Some(cur) = stack.pop() {
        all.push(cur);
        if let Some(n) = name_of(cur) {
            node_name.insert(cur, n);
        }
        let kids = children_of(cur);
        for &ch in &kids {
            parent_of.insert(ch, cur);
            stack.push(ch);
        }
        child_map.insert(cur, kids);
    }

    let (sel_names, exclude) = match filter {
        SubmeshFilter::Include { names, .. } => (names, false),
        SubmeshFilter::Exclude { names } => (names, true),
    };

    // Match each selector name to every node bearing it; a name matching nothing is unresolved.
    let mut matched: Vec<Entity> = Vec::new();
    let mut unresolved: Vec<String> = Vec::new();
    for want in sel_names {
        let hits: Vec<Entity> = all
            .iter()
            .copied()
            .filter(|e| node_name.get(e).is_some_and(|n| n == want))
            .collect();
        if hits.is_empty() {
            unresolved.push(want.clone());
        } else {
            matched.extend(hits);
        }
    }

    // Fail-visible: any unresolved name ⇒ hide nothing (whole model shows), let the caller warn.
    if !unresolved.is_empty() {
        return SubmeshPlan {
            hide: Vec::new(),
            all,
            unresolved,
        };
    }

    if exclude {
        // Hide each named node; its descendants inherit the hidden state.
        return SubmeshPlan {
            hide: matched,
            all,
            unresolved,
        };
    }

    // Include: KEEP = matched ∪ descendants(matched) ∪ ancestors(matched); hide everything else.
    let mut keep: HashSet<Entity> = HashSet::new();
    let mut desc: Vec<Entity> = matched.clone();
    while let Some(e) = desc.pop() {
        if keep.insert(e) {
            if let Some(kids) = child_map.get(&e) {
                desc.extend(kids.iter().copied());
            }
        }
    }
    for &m in &matched {
        let mut cur = m;
        while let Some(&p) = parent_of.get(&cur) {
            keep.insert(p);
            cur = p;
        }
    }
    let hide: Vec<Entity> = all.iter().copied().filter(|e| !keep.contains(e)).collect();
    SubmeshPlan {
        hide,
        all,
        unresolved,
    }
}

/// Run condition for the submesh passes: a submesh-filtered model just spawned, the async glTF scene
/// grew new meshes (its node tree materializes over frames), or new `Aabb`s appeared (mesh bounds land a
/// frame or two after their `Mesh3d`, and the `@center` recenter needs them). Quiet frames skip the
/// walk, so the filter costs nothing once every scene has settled.
fn submesh_scene_growing(
    added_mesh: Query<(), Added<Mesh3d>>,
    added_aabb: Query<(), Added<Aabb>>,
    added_filter: Query<(), Added<SubmeshFilter>>,
) -> bool {
    !added_mesh.is_empty() || !added_aabb.is_empty() || !added_filter.is_empty()
}

/// Apply each holder's [`SubmeshFilter`] to its spawned glTF node tree by NAME. Re-derived idempotently
/// whenever the tree grows ([`submesh_scene_growing`]) so late-arriving nodes are caught; writes
/// `Visibility` only on change to avoid churn. A holder whose scene has not spawned yet (no children) is
/// skipped WITHOUT warning: that is the still-loading case, not a missing name. Once the scene is
/// present, an unresolved selector name warns ONCE ([`SubmeshWarned`]) and renders the whole model.
fn apply_submesh_visibility(
    holders: Query<(Entity, &SubmeshFilter, Has<SubmeshWarned>)>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut vis: Query<&mut Visibility>,
    mut commands: Commands,
) {
    for (holder, filter, warned) in &holders {
        let holder_children: Vec<Entity> = children
            .get(holder)
            .map(|c| c.iter().collect())
            .unwrap_or_default();
        if holder_children.is_empty() {
            continue; // scene not spawned yet, still loading, so neither filter nor warn.
        }
        let plan = submesh_plan(
            &holder_children,
            filter,
            |e| names.get(e).ok().map(|n| n.as_str().to_string()),
            |e| {
                children
                    .get(e)
                    .map(|c| c.iter().collect())
                    .unwrap_or_default()
            },
        );
        // Fail-visible warn (once): the glTF is present but a selector name is not among its nodes: the
        // validator can't see into a GLB, so this is the only place the mismatch surfaces.
        if !plan.unresolved.is_empty() && !warned {
            warn!(
                "submesh selector name(s) {:?} not found in the model's glTF nodes: rendering the \
                 whole model (check the node names inside the GLB)",
                plan.unresolved
            );
            commands.entity(holder).try_insert(SubmeshWarned);
        }
        let hide: HashSet<Entity> = plan.hide.iter().copied().collect();
        for e in plan.all {
            let want = if hide.contains(&e) {
                Visibility::Hidden
            } else {
                Visibility::Inherited
            };
            if let Ok(mut v) = vis.get_mut(e) {
                if *v != want {
                    *v = want;
                }
            } else {
                // A pure transform node may lack a `Visibility`; give it one so the filter still binds.
                commands.entity(e).try_insert(want);
            }
        }
    }
}

/// The new holder-local translation that recenters a lone-include subtree so its bounding-box center
/// `c_local` (already expressed in the holder's own frame) maps to the holder's ORIGINAL origin: SDF
/// `<submesh><center>true` semantics. The holder's rotation/scale are preserved; only its translation
/// shifts by `-R·(S·c_local)`, so the drawn geometry's center lands at the collision/model pose origin.
fn recenter_translation(orig: Vec3, rot: Quat, scale: Vec3, c_local: Vec3) -> Vec3 {
    orig - rot * (scale * c_local)
}

/// The lone-include holders [`apply_submesh_center`] recenters, aliased so the `Query` type stays simple
/// (the codebase's convention for keeping the `type_complexity` lint clean without a suppression).
type CenterHolders<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static SubmeshFilter, &'static mut Transform),
    (With<WorldAssetRoot>, Without<SubmeshCentered>),
>;

/// One-shot `@center` recenter for a lone-include holder (the collision single-submesh case, and any
/// visual lone include that authored `@center`). Once the named subtree's meshes have `Aabb`s, compute
/// their combined center in the holder's LOCAL frame (walking local `Transform`s down from the holder,
/// so no `GlobalTransform` propagation is required) and shift the holder's translation to place that
/// center at the holder origin, tagging [`SubmeshCentered`] so it happens exactly once.
fn apply_submesh_center(
    mut holders: CenterHolders,
    children: Query<&Children>,
    names: Query<&Name>,
    transforms: Query<&Transform, Without<WorldAssetRoot>>,
    aabbs: Query<&Aabb>,
    mut commands: Commands,
) {
    for (holder, filter, mut holder_tf) in &mut holders {
        let SubmeshFilter::Include {
            names: sel,
            center: true,
        } = filter
        else {
            continue;
        };
        let holder_children: Vec<Entity> = children
            .get(holder)
            .map(|c| c.iter().collect())
            .unwrap_or_default();
        if holder_children.is_empty() {
            continue; // scene not spawned yet.
        }
        // Bounds of the named subtree(s) in holder-local space (accumulating local transforms as we
        // descend). `None` until at least one drawn mesh has an `Aabb`; retry next growth frame.
        let Some((min, max)) = submesh_local_bounds(
            &holder_children,
            sel,
            |e| names.get(e).ok().map(|n| n.as_str().to_string()),
            |e| transforms.get(e).copied().unwrap_or(Transform::IDENTITY),
            |e| {
                aabbs
                    .get(e)
                    .ok()
                    .map(|a| (Vec3::from(a.center), Vec3::from(a.half_extents)))
            },
            |e| {
                children
                    .get(e)
                    .map(|c| c.iter().collect())
                    .unwrap_or_default()
            },
        ) else {
            continue;
        };
        let c_local = (min + max) * 0.5;
        holder_tf.translation = recenter_translation(
            holder_tf.translation,
            holder_tf.rotation,
            holder_tf.scale,
            c_local,
        );
        commands.entity(holder).try_insert(SubmeshCentered);
    }
}

/// The bounding box (min, max) of the named subtree(s) in the holder's LOCAL frame, or `None` when no
/// drawn mesh under a named node has an `Aabb` yet. Walks local `Transform`s from the holder down,
/// composing them so each mesh's `Aabb` is expressed in the holder's frame; only nodes within a named
/// subtree contribute (matching the include-mode drawn geometry). Pure/headless-testable via closures.
fn submesh_local_bounds(
    holder_children: &[Entity],
    sel_names: &[String],
    name_of: impl Fn(Entity) -> Option<String>,
    transform_of: impl Fn(Entity) -> Transform,
    aabb_of: impl Fn(Entity) -> Option<(Vec3, Vec3)>,
    children_of: impl Fn(Entity) -> Vec<Entity>,
) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut found = false;
    // DFS carrying (entity, accumulated holder→node transform, whether an ancestor was a named target).
    let mut stack: Vec<(Entity, Transform, bool)> = holder_children
        .iter()
        .map(|&c| (c, transform_of(c), false))
        .collect();
    while let Some((cur, acc, in_named)) = stack.pop() {
        let named = in_named || name_of(cur).is_some_and(|n| sel_names.contains(&n));
        if named {
            if let Some((center, half)) = aabb_of(cur) {
                // The Aabb's 8 corners, transformed into holder-local space, extend the running box.
                for sx in [-1.0_f32, 1.0] {
                    for sy in [-1.0_f32, 1.0] {
                        for sz in [-1.0_f32, 1.0] {
                            let corner = center + half * Vec3::new(sx, sy, sz);
                            // Affine map holder-local ← node-local (explicit T·R·S, no GlobalTransform).
                            let p = acc.translation + acc.rotation * (acc.scale * corner);
                            min = min.min(p);
                            max = max.max(p);
                            found = true;
                        }
                    }
                }
            }
        }
        for ch in children_of(cur) {
            stack.push((ch, acc.mul_transform(transform_of(ch)), named));
        }
    }
    found.then_some((min, max))
}

#[cfg(test)]
mod submesh_tests {
    use super::*;
    use crate::schema::model::geometry::{ExcludeSubmesh, Submesh};
    use crate::schema::model::ModelRef;

    /// A synthetic glTF node tree (entity → name, parent → children): the same headless idiom as
    /// `connector::resolve_tests`, so the plan logic tests without a World/GPU.
    #[derive(Default)]
    struct Tree {
        names: HashMap<Entity, String>,
        children: HashMap<Entity, Vec<Entity>>,
        next: u32,
    }
    impl Tree {
        fn node(&mut self, name: &str) -> Entity {
            let e = Entity::from_raw_u32(self.next).expect("valid test entity");
            self.next += 1;
            self.names.insert(e, name.to_string());
            e
        }
        fn parent(&mut self, p: Entity, c: Entity) {
            self.children.entry(p).or_default().push(c);
        }
        fn plan(&self, roots: &[Entity], f: &SubmeshFilter) -> SubmeshPlan {
            submesh_plan(
                roots,
                f,
                |e| self.names.get(&e).cloned(),
                |e| self.children.get(&e).cloned().unwrap_or_default(),
            )
        }
    }

    /// A QDD-actuator shape: one GLB root whose children are the case, a connector nested under the
    /// case, and a flange. Include {case, flange-sibling...} exercises union + nested descendants.
    fn actuator() -> (Tree, Entity, [Entity; 4]) {
        let mut t = Tree::default();
        let root = t.node("Scene");
        let case = t.node("case");
        let connector = t.node("xt30"); // nested UNDER the case (a connector anchor)
        let flange = t.node("flange");
        t.parent(root, case);
        t.parent(root, flange);
        t.parent(case, connector);
        (t, root, [case, connector, flange, root])
    }

    #[test]
    fn include_union_shows_named_subtrees_hides_the_rest() {
        // Include {case}: the case AND its nested connector stay visible (descendants kept), the flange
        // is hidden, and the structural root (an ancestor of case) stays visible.
        let (t, root, [case, connector, flange, scene_root]) = actuator();
        let plan = t.plan(
            &[scene_root],
            &SubmeshFilter::Include {
                names: vec!["case".into()],
                center: false,
            },
        );
        assert!(plan.unresolved.is_empty());
        let hide: HashSet<Entity> = plan.hide.into_iter().collect();
        assert!(hide.contains(&flange), "flange (unnamed) is hidden");
        assert!(!hide.contains(&case), "the named case shows");
        assert!(
            !hide.contains(&connector),
            "the case's nested connector shows"
        );
        assert!(
            !hide.contains(&root),
            "the ancestor scene root stays visible"
        );
    }

    #[test]
    fn include_union_of_two_names() {
        // Include {case, flange}: both subtrees show; nothing left to hide but their absence-of-others.
        let (t, _root, [case, connector, flange, scene_root]) = actuator();
        let plan = t.plan(
            &[scene_root],
            &SubmeshFilter::Include {
                names: vec!["case".into(), "flange".into()],
                center: false,
            },
        );
        let hide: HashSet<Entity> = plan.hide.into_iter().collect();
        for e in [case, connector, flange] {
            assert!(!hide.contains(&e), "union member {e:?} shows");
        }
        assert!(
            hide.is_empty(),
            "case+flange cover the whole model: {hide:?}"
        );
    }

    #[test]
    fn exclude_hides_named_leaves_everything_else() {
        // Exclude {flange}: only the flange node is hidden (its subtree inherits); case + connector show.
        let (t, _root, [case, connector, flange, scene_root]) = actuator();
        let plan = t.plan(
            &[scene_root],
            &SubmeshFilter::Exclude {
                names: vec!["flange".into()],
            },
        );
        assert!(plan.unresolved.is_empty());
        assert_eq!(plan.hide, vec![flange], "only the flange is hidden");
        let hide: HashSet<Entity> = plan.hide.into_iter().collect();
        assert!(!hide.contains(&case) && !hide.contains(&connector));
    }

    #[test]
    fn unresolvable_name_is_fail_visible() {
        // A typo'd include name resolves to nothing: hide is EMPTY (whole model shows), name reported.
        let (t, _root, [.., scene_root]) = actuator();
        let plan = t.plan(
            &[scene_root],
            &SubmeshFilter::Include {
                names: vec!["flangee".into()],
                center: false,
            },
        );
        assert_eq!(plan.unresolved, vec!["flangee".to_string()]);
        assert!(plan.hide.is_empty(), "fail-visible: nothing hidden");
    }

    #[test]
    fn partial_include_resolution_is_fail_visible() {
        // One good name + one typo ⇒ still fail-visible (don't hide the flange because of a typo).
        let (t, _root, [.., scene_root]) = actuator();
        let plan = t.plan(
            &[scene_root],
            &SubmeshFilter::Include {
                names: vec!["case".into(), "typo".into()],
                center: false,
            },
        );
        assert_eq!(plan.unresolved, vec!["typo".to_string()]);
        assert!(plan.hide.is_empty(), "one typo ⇒ whole model shows");
    }

    #[test]
    fn model_filter_none_when_no_selectors() {
        // The whole-model path: a selector-less model yields NO filter (spawn stays byte-identical).
        let model = ModelRef {
            uri: Some("a.glb".into()),
            ..Default::default()
        };
        assert!(model_submesh_filter(&model).is_none());
    }

    #[test]
    fn model_filter_include_and_exclude_and_center() {
        // Include list with a LONE @center ⇒ Include{center:true}; two includes ⇒ center dropped.
        let lone = ModelRef {
            submesh: vec![Submesh {
                name: Some("flange".into()),
                center: Some("true".into()),
            }],
            ..Default::default()
        };
        assert_eq!(
            model_submesh_filter(&lone),
            Some(SubmeshFilter::Include {
                names: vec!["flange".into()],
                center: true
            })
        );
        let two = ModelRef {
            submesh: vec![
                Submesh {
                    name: Some("a".into()),
                    center: Some("true".into()),
                },
                Submesh {
                    name: Some("b".into()),
                    center: None,
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            model_submesh_filter(&two),
            Some(SubmeshFilter::Include {
                names: vec!["a".into(), "b".into()],
                center: false
            })
        );
        let exc = ModelRef {
            exclude_submesh: vec![ExcludeSubmesh {
                name: Some("flange".into()),
            }],
            ..Default::default()
        };
        assert_eq!(
            model_submesh_filter(&exc),
            Some(SubmeshFilter::Exclude {
                names: vec!["flange".into()]
            })
        );
    }

    #[test]
    fn collision_filter_is_a_lone_include_honoring_center() {
        // The collision single `<submesh>` becomes a lone include, carrying @center (SDF semantics).
        let sm = Submesh {
            name: Some("hull".into()),
            center: Some("true".into()),
        };
        assert_eq!(
            collision_submesh_filter(&sm),
            Some(SubmeshFilter::Include {
                names: vec!["hull".into()],
                center: true
            })
        );
        // A nameless collision submesh resolves to nothing (whole mesh).
        assert!(collision_submesh_filter(&Submesh {
            name: None,
            center: Some("true".into())
        })
        .is_none());
    }

    #[test]
    fn recenter_places_subtree_center_at_holder_origin() {
        // A subtree centered at (1,2,3) in holder-local space, holder at identity ⇒ shift by -(1,2,3),
        // so center + new_translation = origin.
        let c = Vec3::new(1.0, 2.0, 3.0);
        let t = recenter_translation(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, c);
        assert!((t + c).length() < 1e-6, "center lands at the holder origin");
    }

    #[test]
    fn local_bounds_measures_the_named_subtree_only() {
        // holder → flange (Aabb unit box at origin) with a translated child mesh at +X (also part of the
        // named subtree) and an EXCLUDED sibling far away that must NOT widen the box.
        let mut names: HashMap<Entity, String> = HashMap::new();
        let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
        let mut tf: HashMap<Entity, Transform> = HashMap::new();
        let mut aabb: HashMap<Entity, (Vec3, Vec3)> = HashMap::new();
        let mut n = 0u32;
        let mut mk = |name: &str| {
            let e = Entity::from_raw_u32(n).expect("valid entity");
            n += 1;
            names.insert(e, name.to_string());
            e
        };
        let flange = mk("flange");
        let bolt = mk("bolt"); // child of flange (in the named subtree)
        let elsewhere = mk("elsewhere"); // sibling, NOT named
        tf.insert(flange, Transform::IDENTITY);
        tf.insert(bolt, Transform::from_translation(Vec3::new(2.0, 0.0, 0.0)));
        tf.insert(
            elsewhere,
            Transform::from_translation(Vec3::new(100.0, 0.0, 0.0)),
        );
        aabb.insert(flange, (Vec3::ZERO, Vec3::splat(0.5)));
        aabb.insert(bolt, (Vec3::ZERO, Vec3::splat(0.5)));
        aabb.insert(elsewhere, (Vec3::ZERO, Vec3::splat(0.5)));
        children.insert(flange, vec![bolt]);

        let (min, max) = submesh_local_bounds(
            &[flange, elsewhere],
            &["flange".to_string()],
            |e| names.get(&e).cloned(),
            |e| tf.get(&e).copied().unwrap_or(Transform::IDENTITY),
            |e| aabb.get(&e).copied(),
            |e| children.get(&e).cloned().unwrap_or_default(),
        )
        .expect("bounds found");
        // flange box [-0.5,0.5] ∪ bolt box [1.5,2.5] on X ⇒ [-0.5, 2.5]; the far sibling is excluded.
        assert!((min.x - -0.5).abs() < 1e-5, "min.x = {}", min.x);
        assert!((max.x - 2.5).abs() < 1e-5, "max.x = {}", max.x);
    }

    #[test]
    fn ecs_include_hides_complement_exclude_catches_late_nodes() {
        // The system writes Visibility on the real node entities. Include hides the complement; a
        // separately-built exclude holder hides only the named node and leaves a LATER-added node shown.
        let mut app = App::new();
        app.add_systems(Update, apply_submesh_visibility);

        let world = app.world_mut();
        let inc_holder = world
            .spawn(SubmeshFilter::Include {
                names: vec!["case".into()],
                center: false,
            })
            .id();
        let case = world.spawn((Name::new("case"), Visibility::Inherited)).id();
        let flange = world
            .spawn((Name::new("flange"), Visibility::Inherited))
            .id();
        world.entity_mut(inc_holder).add_child(case);
        world.entity_mut(inc_holder).add_child(flange);

        let exc_holder = world
            .spawn(SubmeshFilter::Exclude {
                names: vec!["skin".into()],
            })
            .id();
        let skin = world.spawn((Name::new("skin"), Visibility::Inherited)).id();
        world.entity_mut(exc_holder).add_child(skin);

        app.update();
        let vis = |a: &App, e: Entity| *a.world().get::<Visibility>(e).unwrap();
        assert_eq!(
            vis(&app, flange),
            Visibility::Hidden,
            "include complement hidden"
        );
        assert_eq!(
            vis(&app, case),
            Visibility::Inherited,
            "named include shown"
        );
        assert_eq!(vis(&app, skin), Visibility::Hidden, "excluded node hidden");

        // A node that spawns AFTER the first apply must be caught (visible) on the next derive.
        let bracket = world_child(&mut app, exc_holder, "bracket");
        app.update();
        assert_eq!(
            vis(&app, bracket),
            Visibility::Inherited,
            "a later-added, non-excluded node is visible"
        );
        assert_eq!(
            vis(&app, skin),
            Visibility::Hidden,
            "excluded node still hidden"
        );
    }

    #[test]
    fn ecs_unresolvable_warns_once_and_shows_whole_model() {
        let mut app = App::new();
        app.add_systems(Update, apply_submesh_visibility);
        let world = app.world_mut();
        let holder = world
            .spawn(SubmeshFilter::Include {
                names: vec!["nope".into()],
                center: false,
            })
            .id();
        let case = world.spawn((Name::new("case"), Visibility::Inherited)).id();
        world.entity_mut(holder).add_child(case);

        app.update();
        // Fail-visible: nothing hidden, and the holder is tagged so the warn does not repeat.
        assert_eq!(
            *app.world().get::<Visibility>(case).unwrap(),
            Visibility::Inherited
        );
        assert!(app.world().get::<SubmeshWarned>(holder).is_some());
    }

    /// Spawn `name` as a fresh child of `parent` (a node that arrives after the first filter pass).
    fn world_child(app: &mut App, parent: Entity, name: &str) -> Entity {
        let world = app.world_mut();
        let e = world
            .spawn((Name::new(name.to_string()), Visibility::Inherited))
            .id();
        world.entity_mut(parent).add_child(e);
        e
    }
}

/// KinematicsDisplay: draw the kinematic skeleton, spanning-tree edges (so tree robots show
/// something) plus loop/parallel constraint links in a distinct color.
pub struct KinematicsDisplay;
impl Display for KinematicsDisplay {
    fn id(&self) -> &'static str {
        ID_KINEMATICS
    }
    fn label(&self) -> &str {
        "Kinematics"
    }
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (draw_tree_edges, draw_constraint_links).run_if(display_enabled(self.id())),
        );
    }
}

/// Spanning-tree edges (parent comp → child comp) as thin cyan connector lines: the visible skeleton
/// for any robot, including pure trees with no loop constraints.
fn draw_tree_edges(
    edges: Query<&TreeEdgeMarker>,
    transforms: Query<&GlobalTransform>,
    isolation: Isolation,
    selected: Res<Selected>,
    mut gizmos: Gizmos,
) {
    for edge in &edges {
        // Isolate: single-selection keeps only the edge INTO the selected comp; comp-set isolate keeps
        // an edge only when BOTH endpoints are in the set. No-op when nothing's selected / set is None.
        if isolation.edge_hides(selected.0, edge.parent, edge.child) {
            continue;
        }
        if let (Ok(p), Ok(c)) = (transforms.get(edge.parent), transforms.get(edge.child)) {
            gizmos.line(p.translation(), c.translation(), Color::srgb(0.3, 0.8, 1.0));
        }
    }
}

/// Loop / parallel-mechanism constraint links: orange while closed (and while the solver is off or
/// absent: exactly the historical look), warning-red when the loop-closure solver reports the
/// closure open beyond tolerance (its limits/geometry genuinely prevent assembly). The status is
/// `Option<Res>` because it lives in `JointsPlugin` (core): the composed apps always have it, but an
/// embedder registering displays à la carte without the joints core must degrade to orange, not panic.
fn draw_constraint_links(
    links: Query<&ConstraintLinkMarker>,
    transforms: Query<&GlobalTransform>,
    status: Option<Res<crate::loop_solver::LoopClosureStatus>>,
    mut gizmos: Gizmos,
) {
    for link in &links {
        let open = status.as_ref().is_some_and(|s| {
            s.0.iter()
                .any(|c| c.doc_joint == link.joint && c.error.is_some_and(|e| e.open()))
        });
        let color = if open {
            Color::srgb(0.95, 0.2, 0.1)
        } else {
            Color::srgb(1.0, 0.4, 0.0)
        };
        if let (Ok(p), Ok(c)) = (transforms.get(link.parent), transforms.get(link.child)) {
            gizmos.line(p.translation(), c.translation(), color);
        }
    }
}

/// FramesDisplay: triad gizmos + world-space text labels for `<frame>` and sensor markers.
pub struct FramesDisplay;
impl Display for FramesDisplay {
    fn id(&self) -> &'static str {
        ID_FRAMES
    }
    fn label(&self) -> &str {
        "Frames"
    }
    fn default_enabled(&self) -> bool {
        false
    }
    fn build(&self, app: &mut App) {
        // No display_enabled gate: per-link overrides mean a comp can have Frames ON while the global
        // toggle is OFF (or vice-versa), so the per-comp effective decision lives inside. As an
        // immediate-mode gizmo draw it must run every frame while visible; `kind_drawable` skips the
        // walk only when nothing can show at all.
        app.add_systems(Update, draw_frames.run_if(kind_drawable(self.id())));
    }
}

/// World-space height of gizmo text labels, in metres. `gizmos.text`'s 3D variant sizes glyphs in
/// world units (not pixels; that doc note is for the 2D variant), so this is small for a ~1 m robot.
const LABEL_SIZE: f32 = 0.02;

fn draw_frames(
    frames: Query<(&GlobalTransform, &FrameMarker, &OwnerComp)>,
    comps: Query<(Entity, &GlobalTransform, &CompEntity)>,
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    mut gizmos: Gizmos,
) {
    let global = registry.enabled(ID_FRAMES);
    let override_kind = overrides.kinds.get(ID_FRAMES).copied();

    // Per-link kinematic frames: a triad + name at every comp's GlobalTransform. Tree robots have no
    // explicit <frame> elements, so this is what makes the Frames display non-blank for them.
    for (entity, gt, comp) in &comps {
        // Per-comp effective Frames toggle: the selected comp honors its override; others follow global.
        if !effective_kind_enabled(global, override_kind, selected.0 == Some(entity)) {
            continue;
        }
        // Isolate: only the selected comp's own frame (or the set's comps); no-op when nothing's isolated.
        if isolation.hides(selected.0, entity) {
            continue;
        }
        gizmos.axes(gt.compute_transform(), 0.04);
        gizmos.text(
            Isometry3d::from_translation(gt.translation()),
            &comp.name,
            LABEL_SIZE,
            Vec2::ZERO,
            Color::srgb(0.85, 0.85, 0.9),
        );
    }

    // Explicit <frame> triads (e.g. optical frames), drawn on top.
    for (gt, marker, owner) in &frames {
        if !effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0)) {
            continue;
        }
        if isolation.hides(selected.0, owner.0) {
            continue;
        }
        let mut t = gt.compute_transform();
        if marker.optical {
            // optical frames are Z-forward; show the triad rotated so X stays right, Z points fwd.
            t.rotation *= Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        }
        gizmos.axes(t, 0.03);
        // Label with the frame name plus, when typed, its human-facing type (tcp → the spelled-out
        // form, per the UI text rule; see `frame_type_label`), so the frame's role reads in-scene.
        let text = match &marker.type_label {
            Some(ty) => format!("{} ({})", marker.label, ty),
            None => marker.label.clone(),
        };
        gizmos.text(
            Isometry3d::from_translation(t.translation),
            &text,
            LABEL_SIZE,
            Vec2::ZERO,
            Color::WHITE,
        );
    }
}

/// SensorsDisplay: show sensor markers (triads/labels) and FOV frustum meshes.
pub struct SensorsDisplay;
impl Display for SensorsDisplay {
    fn id(&self) -> &'static str {
        ID_SENSORS
    }
    fn label(&self) -> &str {
        "Sensors & FOV"
    }
    fn build(&self, app: &mut App) {
        // Sensor FOV meshes are entities (isolate-aware via `sync_sensor_visibility`); the sensor triads
        // are immediate-mode gizmos. Neither takes a display_enabled gate: per-link overrides mean a
        // comp can have Sensors ON while the global toggle is OFF (or vice-versa), so the per-comp
        // effective decision lives inside each system, but each gets the cheapest safe skip: the sync
        // re-runs only when its inputs change, and the per-frame triad draw skips only when nothing can
        // pass `effective_kind_enabled` at all.
        app.add_systems(
            Update,
            sync_sensor_visibility.run_if(
                vis_inputs_changed::<SensorMarker>.or_else(resource_changed::<SensorVizOverrides>),
            ),
        )
        .add_systems(Update, draw_sensor_triads.run_if(kind_drawable(self.id())));
    }
}

/// The sensor FOV-mesh visibility query (FOV frustums are the `SensorMarker` entities that carry a
/// mesh; the triad-only markers have no `Mesh3d`). Aliased to keep the system signature under the
/// `type_complexity` lint without suppressing it.
type SensorMeshVis<'w, 's> = Query<
    'w,
    's,
    (
        &'static OwnerComp,
        &'static SensorMarker,
        &'static mut Visibility,
    ),
    With<Mesh3d>,
>;

/// Sync FoV/lidar-scan mesh visibility with the global Sensors toggle, per-link + per-sensor overrides,
/// and isolate. Public so headless tests can drive it without the immediate-mode `draw_sensor_triads`
/// (which needs a GizmoConfigStore that MinimalPlugins doesn't provide); the [`SensorsDisplay`] wires it
/// with its change-gated run condition in the real app.
pub fn sync_sensor_visibility(
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    sensor_viz: SensorViz,
    mut q: SensorMeshVis,
) {
    let global = registry.enabled(ID_SENSORS);
    let override_kind = overrides.kinds.get(ID_SENSORS).copied();
    for (owner, marker, mut v) in &mut q {
        let enabled = sensor_viz.allows(owner.0, &marker.label)
            && effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0));
        let want = isolation.visibility(enabled, selected.0, owner.0);
        if *v != want {
            *v = want;
        }
    }
}

fn draw_sensor_triads(
    sensors: Query<(&GlobalTransform, &SensorMarker, &OwnerComp), Without<Mesh3d>>,
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    sensor_viz: SensorViz,
    mut gizmos: Gizmos,
) {
    let global = registry.enabled(ID_SENSORS);
    let override_kind = overrides.kinds.get(ID_SENSORS).copied();
    for (gt, marker, owner) in &sensors {
        // Per-sensor override (this ONE sensor's viz on/off) ANDs with everything below.
        if !sensor_viz.allows(owner.0, &marker.label) {
            continue;
        }
        // Per-comp effective Sensors toggle: the selected comp honors its override; others follow global.
        if !effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0)) {
            continue;
        }
        // Isolate: skip triads owned by any comp outside the selection / set (no-op when nothing's
        // isolated, so isolate has no effect on deselect).
        if isolation.hides(selected.0, owner.0) {
            continue;
        }
        gizmos.axes(gt.compute_transform(), SENSOR_TRIAD_LEN);
        gizmos.text(
            Isometry3d::from_translation(gt.translation()),
            &marker.label,
            LABEL_SIZE,
            Vec2::ZERO,
            Color::srgb(0.6, 0.9, 1.0),
        );
    }
}

/// Raw sensor triad gizmo length (metres): also the base the aligned triad scales from.
const SENSOR_TRIAD_LEN: f32 = 0.02;

/// The aligned triad draws 1.5× the raw triad, so an identity remap (aligned == raw directions) still
/// reads: its dashes extend past the raw arrowheads instead of hiding underneath them.
const ALIGNED_TRIAD_LEN: f32 = SENSOR_TRIAD_LEN * 1.5;

/// Dashes per aligned axis; the last dash is drawn as an arrowhead so the direction reads.
const ALIGNED_DASHES: u32 = 4;

/// Dimmed X/Y/Z sRGB colors for the aligned triad, deliberately softer than the full-saturation
/// RED/GREEN/BLUE `gizmos.axes` uses for the raw triad, so raw vs aligned read apart at a glance.
const ALIGNED_AXIS_COLORS: [[f32; 3]; 3] =
    [[0.75, 0.35, 0.35], [0.35, 0.65, 0.35], [0.40, 0.45, 0.85]];

/// SensorAxisAlignDisplay: for every sensor carrying a `<driver><axis-align>` remap, draw a second,
/// dashed + dimmed triad showing the axes AFTER the remap alongside the raw pose triad: the legacy
/// dendrite tool for verifying IMU/mag mounting.
///
/// Legacy toggled raw vs aligned per sensor; here the DisplayRegistry is a flat toggle list, so this
/// is one global "Sensor axis-align" sub-toggle instead, and because BOTH triads draw at once
/// (visually distinguished), no per-sensor switch is needed to compare them. Visibility follows the
/// Sensors display: the draw is additionally gated on `kind_drawable(ID_SENSORS)` and applies the
/// SAME per-comp effective toggle + isolate rules as `draw_sensor_triads`, so aligned triads only
/// ever show where the raw sensor triads do. Default ON: it renders nothing unless a shown sensor
/// actually carries `<axis-align>`, so it is silent on documents without remaps and surfaces the
/// mounting check the moment one appears.
pub struct SensorAxisAlignDisplay;
impl Display for SensorAxisAlignDisplay {
    fn id(&self) -> &'static str {
        ID_SENSOR_AXIS_ALIGN
    }
    fn label(&self) -> &str {
        "Sensor axis-align"
    }
    fn build(&self, app: &mut App) {
        // Two chained conditions AND together: this display's own master toggle, plus "some sensor
        // triad could draw at all" (the per-comp Sensors decision still runs inside the system).
        app.add_systems(
            Update,
            draw_sensor_axis_align
                .run_if(display_enabled(self.id()))
                .run_if(kind_drawable(ID_SENSORS)),
        );
    }
}

fn draw_sensor_axis_align(
    sensors: Query<(
        &GlobalTransform,
        &AlignedAxesMarker,
        &SensorMarker,
        &OwnerComp,
    )>,
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    sensor_viz: SensorViz,
    mut gizmos: Gizmos,
) {
    // Follows the SENSORS kind (not its own id): the aligned triad is an annotation on the raw sensor
    // triad, so it obeys the same global toggle / per-link override / isolate / per-sensor rules.
    let global = registry.enabled(ID_SENSORS);
    let override_kind = overrides.kinds.get(ID_SENSORS).copied();
    for (gt, marker, sensor, owner) in &sensors {
        if !sensor_viz.allows(owner.0, &sensor.label) {
            continue;
        }
        if !effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0)) {
            continue;
        }
        if isolation.hides(selected.0, owner.0) {
            continue;
        }
        let t = gt.compute_transform();
        for (i, rgb) in ALIGNED_AXIS_COLORS.into_iter().enumerate() {
            // Column i of the remap = where raw axis i lands, expressed in the sensor-pose frame
            // (legacy composition: the aligned triad is drawn under the sensor pose).
            let dir = t.rotation * marker.0.col(i);
            draw_dashed_arrow(
                &mut gizmos,
                t.translation,
                dir,
                Color::srgb(rgb[0], rgb[1], rgb[2]),
            );
        }
    }
}

/// One dashed aligned-triad axis from `origin` along the unit `dir`: [`ALIGNED_DASHES`] dashes with
/// gaps between them ending exactly at [`ALIGNED_TRIAD_LEN`], the final dash drawn as an arrowhead.
/// Dashing is the raw-vs-aligned distinction: `gizmos.axes` draws the raw triad solid.
fn draw_dashed_arrow(gizmos: &mut Gizmos, origin: Vec3, dir: Vec3, color: Color) {
    // n dashes + (n-1) gaps of equal length tile the axis, so the last dash ends at the tip.
    let seg = ALIGNED_TRIAD_LEN / (2 * ALIGNED_DASHES - 1) as f32;
    for k in 0..ALIGNED_DASHES {
        let a = origin + dir * (seg * (2 * k) as f32);
        let b = a + dir * seg;
        if k + 1 == ALIGNED_DASHES {
            gizmos.arrow(a, b, color).with_tip_length(seg * 0.6);
        } else {
            gizmos.line(a, b, color);
        }
    }
}

/// ConnectivityDisplay: canonical endpoint annotations (default off).
pub struct ConnectivityDisplay;
impl Display for ConnectivityDisplay {
    fn id(&self) -> &'static str {
        ID_CONNECTIVITY
    }
    fn label(&self) -> &str {
        "Connectivity endpoints"
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
            sync_connector_visibility.run_if(vis_inputs_changed::<ConnectorMarker>),
        );
    }
}

fn sync_connector_visibility(
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    mut q: Query<(&OwnerComp, &mut Visibility), With<ConnectorMarker>>,
) {
    let global = registry.enabled(ID_CONNECTIVITY);
    let override_kind = overrides.kinds.get(ID_CONNECTIVITY).copied();
    for (owner, mut v) in &mut q {
        let enabled = effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0));
        let want = isolation.visibility(enabled, selected.0, owner.0);
        if *v != want {
            *v = want;
        }
    }
}

/// CollisionDisplay: translucent collision overlay (default off).
pub struct CollisionDisplay;
impl Display for CollisionDisplay {
    fn id(&self) -> &'static str {
        ID_COLLISION
    }
    fn label(&self) -> &str {
        "Collision"
    }
    fn default_enabled(&self) -> bool {
        false
    }
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (
                sync_collision_visibility.run_if(vis_inputs_changed::<CollisionItem>),
                // Runs every frame (NOT change-gated): a collision glTF/GLB scene spawns its meshes as
                // descendants over several frames AFTER the rebuild, so the tint must keep catching newly
                // arrived children until all are swapped (each swap is one-shot via CollisionSceneTinted).
                tint_collision_scene_materials,
            ),
        );
    }
}

fn sync_collision_visibility(
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    mut q: Query<(&OwnerComp, &mut Visibility), With<CollisionItem>>,
) {
    let global = registry.enabled(ID_COLLISION);
    let override_kind = overrides.kinds.get(ID_COLLISION).copied();
    for (owner, mut v) in &mut q {
        let enabled = effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0));
        let want = isolation.visibility(enabled, selected.0, owner.0);
        if *v != want {
            *v = want;
        }
    }
}

/// Marks a mesh entity inside a collision glTF/GLB scene whose baked material has already been overridden
/// with the shared translucent collision material by [`tint_collision_scene_materials`], so the swap is a
/// one-shot (idempotent across frames as the async scene keeps spawning children).
#[derive(Component)]
struct CollisionSceneTinted;

/// Override the baked PBR materials of every mesh spawned under a COLLISION glTF/GLB scene root with the
/// shared translucent collision material (the same look as primitive/STL collisions) so a `.glb`/`.gltf`
/// collision renders as a uniform collision overlay rather than its authored textures (the batteries/cables
/// that showed under the Collision display). A collision scene root carries BOTH [`CollisionItem`] and
/// [`WorldAssetRoot`] (see the `CollisionMeshKind::Scene` spawn), and its glTF meshes arrive as descendants
/// over several frames; this walks each such root's subtree and swaps any not-yet-tinted mesh material,
/// tagging it [`CollisionSceneTinted`] so it happens once. VISUAL scenes ([`SceneItem`] + [`WorldAssetRoot`]
/// WITHOUT `CollisionItem`) are never matched, so they keep their authored materials. The `.after` on the
/// visibility system is unnecessary: this only mutates materials, never visibility.
fn tint_collision_scene_materials(
    roots: Query<Entity, (With<CollisionItem>, With<WorldAssetRoot>)>,
    children: Query<&Children>,
    has_material: Query<(), With<MeshMaterial3d<StandardMaterial>>>,
    already_tinted: Query<(), With<CollisionSceneTinted>>,
    mut commands: Commands,
    mut cache: ResMut<GeometryCache>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Nothing to do without a collision scene root; avoid materializing the shared material otherwise.
    if roots.is_empty() {
        return;
    }
    // The shared translucent collision material: the SAME color-deduped handle primitive/STL collisions
    // use (see the `[0.2, 0.8, 0.9, 0.25]` tint in `spawn_element_children`).
    let mat = cache.material([0.2, 0.8, 0.9, 0.25], &mut materials);
    for root in &roots {
        // DFS the collision scene root's descendants; swap+tag each not-yet-tinted mesh material.
        let mut stack: Vec<Entity> = Vec::new();
        if let Ok(ch) = children.get(root) {
            stack.extend(ch.iter());
        }
        while let Some(e) = stack.pop() {
            if let Ok(ch) = children.get(e) {
                stack.extend(ch.iter());
            }
            if has_material.contains(e) && !already_tinted.contains(e) {
                commands
                    .entity(e)
                    .insert((MeshMaterial3d(mat.clone()), CollisionSceneTinted));
            }
        }
    }
}

#[cfg(test)]
mod collision_tint_tests {
    use super::*;
    use bevy::world_serialization::WorldAssetRoot;

    /// A collision glTF scene's baked child materials are overridden with the shared translucent collision
    /// material, while a VISUAL scene's child material is left untouched.
    #[test]
    fn collision_scene_materials_overridden_visual_untouched() {
        let mut app = App::new();
        app.init_resource::<GeometryCache>()
            .init_resource::<Assets<StandardMaterial>>()
            .add_systems(Update, tint_collision_scene_materials);

        // Two distinct opaque baked materials (NOT the collision tint).
        let (baked_col, baked_vis) = {
            let mut mats = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
            (
                mats.add(StandardMaterial::from_color(Color::srgb(1.0, 0.0, 0.0))),
                mats.add(StandardMaterial::from_color(Color::srgb(0.0, 1.0, 0.0))),
            )
        };

        // A COLLISION scene root (CollisionItem + WorldAssetRoot) with a baked mesh child, and a VISUAL
        // scene root (SceneItem + WorldAssetRoot, NO CollisionItem) with its own baked mesh child.
        let world = app.world_mut();
        let col_root = world
            .spawn((CollisionItem, SceneItem, WorldAssetRoot(default())))
            .id();
        let col_child = world.spawn(MeshMaterial3d(baked_col.clone())).id();
        world.entity_mut(col_root).add_child(col_child);

        let vis_root = world.spawn((SceneItem, WorldAssetRoot(default()))).id();
        let vis_child = world.spawn(MeshMaterial3d(baked_vis.clone())).id();
        world.entity_mut(vis_root).add_child(vis_child);

        // Two frames: the command-queued swap applies at the end of the first update.
        app.update();
        app.update();

        // The collision child's material was swapped to the translucent collision tint + tagged done.
        let col_mat = app
            .world()
            .get::<MeshMaterial3d<StandardMaterial>>(col_child)
            .expect("collision child keeps a material")
            .0
            .clone();
        assert_ne!(col_mat, baked_col, "collision baked material was replaced");
        assert!(
            app.world().get::<CollisionSceneTinted>(col_child).is_some(),
            "collision child tagged tinted"
        );
        let srgba = app
            .world()
            .resource::<Assets<StandardMaterial>>()
            .get(&col_mat)
            .expect("swapped material exists")
            .base_color
            .to_srgba();
        assert!(
            (srgba.red - 0.2).abs() < 0.02
                && (srgba.green - 0.8).abs() < 0.02
                && (srgba.blue - 0.9).abs() < 0.02
                && (srgba.alpha - 0.25).abs() < 0.02,
            "swapped to the shared collision tint, got {srgba:?}"
        );

        // The visual child is untouched: same baked handle, no tint tag.
        let vis_mat = app
            .world()
            .get::<MeshMaterial3d<StandardMaterial>>(vis_child)
            .expect("visual child keeps a material")
            .0
            .clone();
        assert_eq!(vis_mat, baked_vis, "visual scene material untouched");
        assert!(
            app.world().get::<CollisionSceneTinted>(vis_child).is_none(),
            "visual child never tagged"
        );
    }
}

/// InertialDisplay: CG markers (default off).
pub struct InertialDisplay;
impl Display for InertialDisplay {
    fn id(&self) -> &'static str {
        ID_INERTIAL
    }
    fn label(&self) -> &str {
        "Inertial (CG)"
    }
    fn default_enabled(&self) -> bool {
        false
    }
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            sync_inertial_visibility.run_if(vis_inputs_changed::<InertialMarker>),
        );
    }
}

fn sync_inertial_visibility(
    registry: Res<DisplayRegistry>,
    isolation: Isolation,
    selected: Res<Selected>,
    overrides: Res<SelectionOverrides>,
    mut q: Query<(&OwnerComp, &mut Visibility), With<InertialMarker>>,
) {
    let global = registry.enabled(ID_INERTIAL);
    let override_kind = overrides.kinds.get(ID_INERTIAL).copied();
    for (owner, mut v) in &mut q {
        let enabled = effective_kind_enabled(global, override_kind, selected.0 == Some(owner.0));
        let want = isolation.visibility(enabled, selected.0, owner.0);
        if *v != want {
            *v = want;
        }
    }
}

/// Dedicated gizmo group for the world origin axes. The X/Y axes lie IN the grid's Z=0 plane, so in the
/// default group they are COPLANAR with the grid lines and z-fight them; at grazing camera angles the
/// grid wins the depth test and an in-plane axis disappears. Drawing the axes in their own group with a
/// negative `depth_bias` renders them ON TOP of the coplanar grid, so they stay visible at every angle.
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct WorldAxesGizmos;

/// GridDisplay: world grid + origin axes oriented to the world frame.
pub struct GridDisplay;
impl Display for GridDisplay {
    fn id(&self) -> &'static str {
        ID_GRID
    }
    fn label(&self) -> &str {
        "Grid & axes"
    }
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<WorldAxesGizmos>()
            .add_systems(Startup, setup_world_axes_gizmos)
            .add_systems(Update, draw_grid.run_if(display_enabled(self.id())));
    }
}

/// Put the world axes in front of the coplanar grid (and other Z=0 ephemera) so they never z-fight and
/// vanish; a slightly thicker line also reads clearly above the hairline grid.
fn setup_world_axes_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<WorldAxesGizmos>();
    config.depth_bias = -1.0;
    config.line.width = 2.0;
}

/// Dedicated gizmo group for the selection highlight, configured with a thick line so the box is
/// unmistakable (the default group draws hairline-thin).
#[derive(Default, Reflect, GizmoConfigGroup)]
pub struct HighlightGizmos;

/// Always-on selection feedback: the yellow bounds box (`draw_highlight`) plus the 3D joint-anchor
/// axes triad (`sync_selection_triad`). The highlight is NOT a toggleable display: it must show
/// whenever a comp is selected, regardless of which displays are enabled, so it lives in its own
/// plugin rather than under GridDisplay (where it previously, incorrectly, vanished if the grid was
/// turned off).
pub struct HighlightPlugin;
impl Plugin for HighlightPlugin {
    fn build(&self, app: &mut App) {
        app.init_gizmo_group::<HighlightGizmos>()
            .init_resource::<SelectionTriadState>()
            .add_systems(Startup, setup_highlight_gizmos)
            // The triad sync runs after the rebuild so on a doc-change frame it sees the cleared
            // selection (and the recursively despawned old glyph) rather than re-anchoring to stale
            // entities.
            .add_systems(
                Update,
                (
                    draw_highlight,
                    sync_selection_triad.after(SceneSet::Rebuild),
                ),
            );
    }
}

fn setup_highlight_gizmos(mut store: ResMut<GizmoConfigStore>) {
    let (config, _) = store.config_mut::<HighlightGizmos>();
    config.line.width = 4.0;
}

fn draw_grid(roots: Query<&WorldRoot>, mut gizmos: Gizmos, mut axes: Gizmos<WorldAxesGizmos>) {
    let convention = roots
        .iter()
        .next()
        .map(|r| r.convention)
        .unwrap_or_default();
    // World axes drawn in the world frame, mapped to Bevy by the convention. Drawn in the WorldAxesGizmos
    // group (negative depth bias) so the in-plane X/Y axes render on top of the coplanar grid rather than
    // z-fighting it and vanishing at grazing angles.
    let o = Vec3::ZERO;
    axes.line(
        o,
        convention.world_point_to_bevy(Vec3::X * 0.3),
        Color::srgb(0.9, 0.2, 0.2),
    );
    axes.line(
        o,
        convention.world_point_to_bevy(Vec3::Y * 0.3),
        Color::srgb(0.2, 0.9, 0.2),
    );
    axes.line(
        o,
        convention.world_point_to_bevy(Vec3::Z * 0.3),
        Color::srgb(0.2, 0.2, 0.9),
    );

    // Grid on the world ground plane (world Z=0), mapped to Bevy.
    let half = 5;
    let step = 0.5;
    let extent = half as f32 * step;
    let c = Color::srgb(0.28, 0.28, 0.28);
    for i in -half..=half {
        let o = i as f32 * step;
        let a = convention.world_point_to_bevy(Vec3::new(-extent, o, 0.0));
        let b = convention.world_point_to_bevy(Vec3::new(extent, o, 0.0));
        gizmos.line(a, b, c);
        let a = convention.world_point_to_bevy(Vec3::new(o, -extent, 0.0));
        let b = convention.world_point_to_bevy(Vec3::new(o, extent, 0.0));
        gizmos.line(a, b, c);
    }
}

/// Should this direct child of a comp contribute to the selection bounding box? Only the comp's
/// ACTUAL geometry counts: visual meshes and (when shown) the collision overlay. Everything else that
/// hangs off a comp (sensor FOV frusta, connectivity endpoint annotations, the inertial CG marker,
/// frame markers, labels, and the selection-triad arrows) is annotation, not geometry, and previously
/// ballooned the box. Hidden items don't count either (a toggled-off overlay shouldn't size the box).
/// Pure so the filter unit-tests headless.
pub fn include_in_selection_bounds(is_visual: bool, is_collision: bool, hidden: bool) -> bool {
    (is_visual || is_collision) && !hidden
}

/// Union comp-RELATIVE mesh AABBs into one comp-local `(min, max)` bound. Each entry is a mesh
/// entity's affine relative to the comp (`comp ← mesh`) plus its own AABB center/half-extents.
///
/// Working in comp-local space (instead of unioning world-space boxes) makes the result INVARIANT
/// under the comp's world motion: articulating a joint moves the comp and its meshes together, so
/// every relative affine (and therefore the box) is unchanged, and the highlight no longer swells as
/// a wheel spins. Returns `None` for an empty input (e.g. a glTF still loading). Pure for headless
/// unit tests.
pub fn union_local_aabbs(
    entries: impl IntoIterator<Item = (bevy::math::Affine3A, Vec3, Vec3)>,
) -> Option<(Vec3, Vec3)> {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    let mut any = false;
    for (rel, center, half_extents) in entries {
        any = true;
        // Transform the 8 corners of the mesh-local AABB into comp-local space and grow the bound.
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    let corner = center + half_extents * Vec3::new(sx, sy, sz);
                    let local = rel.transform_point3(corner);
                    min = min.min(local);
                    max = max.max(local);
                }
            }
        }
    }
    any.then_some((min, max))
}

/// The queries needed to measure a selected comp's own-geometry bounds, grouped so the highlight box
/// and the selection-triad sizing share one gather (and each system stays within the argument budget).
#[derive(bevy::ecs::system::SystemParam)]
struct SelectionBounds<'w, 's> {
    comps: Query<'w, 's, &'static GlobalTransform, With<CompEntity>>,
    children: Query<'w, 's, &'static Children>,
    items: Query<'w, 's, (&'static Visibility, Has<VisualItem>, Has<CollisionItem>)>,
    aabbs: Query<'w, 's, (&'static GlobalTransform, &'static Aabb)>,
}

impl SelectionBounds<'_, '_> {
    /// Comp-LOCAL `(min, max)` bound of the comp's own geometry: the mesh AABBs under its visual
    /// items plus its shown collision overlays ([`include_in_selection_bounds`]). Child-comp subtrees
    /// are excluded (each link boxes its own geometry): a child comp carries neither geometry marker,
    /// so the direct-child filter drops it along with the annotation glyphs. `None` when the comp has
    /// no measurable geometry yet (e.g. its glTF is still loading).
    fn local_bounds(&self, comp: Entity) -> Option<(Vec3, Vec3)> {
        let inv = self.comps.get(comp).ok()?.affine().inverse();
        let mut entries = Vec::new();
        let mut stack = Vec::new();
        for &item in self.children.get(comp).into_iter().flatten() {
            let Ok((vis, is_visual, is_collision)) = self.items.get(item) else {
                continue;
            };
            if include_in_selection_bounds(is_visual, is_collision, *vis == Visibility::Hidden) {
                stack.push(item);
            }
        }
        // Collect every mesh AABB in the included subtrees (a glTF visual's meshes are descendants
        // of its `VisualItem` root), expressed relative to the comp.
        while let Some(cur) = stack.pop() {
            if let Ok((gt, aabb)) = self.aabbs.get(cur) {
                entries.push((
                    inv * gt.affine(),
                    Vec3::from(aabb.center),
                    Vec3::from(aabb.half_extents),
                ));
            }
            if let Ok(ch) = self.children.get(cur) {
                stack.extend(ch.iter());
            }
        }
        union_local_aabbs(entries)
    }
}

/// Selection highlight box: an unmistakable yellow wireframe around the selected comp's OWN geometry.
///
/// The box is computed in comp-LOCAL space from the visual (+ shown collision) mesh AABBs only (see
/// [`SelectionBounds::local_bounds`]) then drawn as an oriented box under the comp's transform. That
/// keeps it tight against the geometry (annotation glyphs are excluded) and constant-size under joint
/// articulation (it turns WITH the wheel instead of growing around it). Falls back to a fixed-size box
/// at the comp origin while no mesh AABB is measurable (e.g. glTF still loading). The companion axes
/// triad is the [`SelectionTriad`] mesh glyph, anchored to the joint frame by [`sync_selection_triad`].
fn draw_highlight(
    selected: Res<Selected>,
    highlight_set: Res<HighlightSet>,
    bounds: SelectionBounds,
    mut gizmos: Gizmos<HighlightGizmos>,
) {
    // The plain comp selection stays yellow: exactly the old single-box behaviour. Then every
    // HighlightSet entry gets the SAME bounds math in its own color (empty set ⇒ nothing extra), so the
    // embedder can paint a joint's parent/child comps without duplicating the geometry gather.
    if let Some(e) = selected.0 {
        draw_highlight_box(e, Color::srgb(1.0, 0.85, 0.0), &bounds, &mut gizmos);
    }
    for &(e, color) in &highlight_set.0 {
        draw_highlight_box(e, color, &bounds, &mut gizmos);
    }
}

/// Draw ONE selection-style bounds box around comp `e` in `color`: the comp-local geometry AABB when
/// measurable ([`SelectionBounds::local_bounds`]), else a fixed-size box at the comp origin (e.g. glTF
/// still loading). Shared by the plain yellow [`Selected`] highlight and every [`HighlightSet`] entry.
fn draw_highlight_box(
    e: Entity,
    color: Color,
    bounds: &SelectionBounds,
    gizmos: &mut Gizmos<HighlightGizmos>,
) {
    let Ok(comp_gt) = bounds.comps.get(e) else {
        return;
    };
    gizmos.cube(
        highlight_box_transform(comp_gt.compute_transform(), bounds.local_bounds(e)),
        color,
    );
}

/// PURE: the world transform of the highlight wireframe cube for a comp at `comp` with comp-LOCAL
/// geometry `local_bounds`. With measurable bounds the cube hugs the geometry (a hairline pad so the
/// wireframe reads as an outline, not a margin); with `None` (e.g. glTF still loading) it is a
/// fixed-size 0.08 box at the comp origin. Pure so the bounds→box math unit-tests headless.
pub fn highlight_box_transform(comp: Transform, local_bounds: Option<(Vec3, Vec3)>) -> Transform {
    match local_bounds {
        Some((min, max)) => {
            let half = (max - min) * 0.5;
            let pad = (half * 0.02).max(Vec3::splat(0.001));
            comp * Transform::from_translation((min + max) * 0.5).with_scale((half + pad) * 2.0)
        }
        None => {
            let mut t = comp;
            t.scale = Vec3::splat(0.08);
            t
        }
    }
}

/// Marker: the spawned selection-triad glyph root (three 3D arrow meshes at the selected comp's
/// joint-anchor frame). Excluded from picking (`Pickable::IGNORE` on its meshes) and from the
/// selection bounds (it carries no geometry marker, so the bounds gather skips it).
#[derive(Component)]
pub struct SelectionTriad;

/// What [`sync_selection_triad`] last built, so the glyph is rebuilt only when the selection, the
/// measured comp size, or the scene actually changes, not every frame.
#[derive(Resource, Default)]
struct SelectionTriadState {
    comp: Option<Entity>,
    root: Option<Entity>,
    size: f32,
}

/// Arrow length used while the selected comp has no measurable geometry yet (matches the old fixed
/// gizmo triad length).
const FALLBACK_TRIAD_LEN: f32 = 0.06;

/// Per-axis arrow colors: X red, Y green, Z blue; same semantics as the old line triad.
const TRIAD_COLORS: [[f32; 4]; 3] = [
    [0.85, 0.15, 0.15, 1.0],
    [0.15, 0.7, 0.15, 1.0],
    [0.2, 0.35, 0.95, 1.0],
];

/// Keep the selection-triad glyph in sync with [`Selected`]: three solid 3D arrows (cylinder shaft +
/// cone tip, X/Y/Z in red/green/blue) marking the frame the selected comp's JOINT works in.
///
/// The glyph is parented to the NON-moving joint anchor (the parent comp at the joint's fixed
/// `origin`) because a joint's axis is defined in that frame: it must not spin with the child body
/// as the joint articulates (it still follows upstream links, e.g. a steering knuckle carrying the
/// wheel axle). Root comps have no joint, so their triad sits on the comp itself. Arrows are sized
/// proportional to the comp's own geometry and rebuilt when the selection or measured size changes
/// (a late-loading glTF grows the arrows once its bounds appear).
fn sync_selection_triad(
    selected: Res<Selected>,
    joints: Res<crate::joints::ArticulatedJoints>,
    parents: Query<&ChildOf>,
    bounds: SelectionBounds,
    mut state: ResMut<SelectionTriadState>,
    mut assets: SceneAssets,
    mut commands: Commands,
) {
    // Desired glyph for the current selection: the comp plus an arrow length ~3/4 of its largest
    // geometry extent (clamped so tiny/huge links still get a legible, non-absurd triad).
    let target = selected.0.filter(|&e| bounds.comps.contains(e)).map(|e| {
        let len = bounds
            .local_bounds(e)
            .map(|(min, max)| ((max - min).max_element() * 0.75).clamp(0.03, 1.0))
            .unwrap_or(FALLBACK_TRIAD_LEN);
        (e, len)
    });

    let unchanged = match (target, state.comp) {
        (Some((e, len)), Some(cur)) => cur == e && (len - state.size).abs() <= state.size * 0.01,
        (None, None) => true,
        _ => false,
    };
    if unchanged {
        return;
    }

    // Despawn the stale glyph (fallible: a scene rebuild may have already recursively despawned it).
    if let Some(root) = state.root.take() {
        if let Ok(mut e) = commands.get_entity(root) {
            e.try_despawn();
        }
    }
    *state = SelectionTriadState::default();
    let Some((comp, len)) = target else {
        return;
    };

    // Anchor frame: for a jointed comp, the parent comp at the joint's fixed origin (every tree edge
    // is catalogued, including fixed joints); for a root comp, the comp itself.
    let (anchor, local) = match joints.0.iter().find(|j| j.child == comp) {
        Some(j) => match parents.get(comp) {
            Ok(p) => (p.parent(), j.origin),
            Err(_) => (comp, Transform::IDENTITY),
        },
        None => (comp, Transform::IDENTITY),
    };
    let root = spawn_triad(&mut commands, &mut assets, local, len);
    commands.entity(anchor).add_child(root);
    *state = SelectionTriadState {
        comp: Some(comp),
        root: Some(root),
        size: len,
    };
}

/// Spawn the triad glyph: three arrows (shared unit cylinder/cone meshes, sized via scale) along the
/// local X/Y/Z of `local`, colored per [`TRIAD_COLORS`]. Returns the glyph root (not yet parented).
fn spawn_triad(
    commands: &mut Commands,
    assets: &mut SceneAssets,
    local: Transform,
    len: f32,
) -> Entity {
    use bevy::picking::Pickable;
    let shaft_mesh = crate::geometry::unit_cylinder(&mut assets.cache, &mut assets.meshes);
    let tip_mesh = crate::geometry::unit_cone(&mut assets.cache, &mut assets.meshes);
    let root = commands
        .spawn((
            SceneItem,
            SelectionTriad,
            local,
            Visibility::default(),
            Name::new("selection-triad"),
        ))
        .id();
    // Arrow proportions: 70% shaft + 30% cone tip; slender shaft, clearly wider tip base.
    let (shaft_len, tip_len) = (len * 0.7, len * 0.3);
    let (shaft_r, tip_r) = (len * 0.02, len * 0.05);
    // The unit meshes point along +Z; rotate each arrow onto its axis (+Z→+X, +Z→+Y, +Z stays).
    let rotations = [
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
        Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
        Quat::IDENTITY,
    ];
    for (rot, rgba) in rotations.into_iter().zip(TRIAD_COLORS) {
        let mat = assets.cache.material(rgba, &mut assets.materials);
        let shaft = commands
            .spawn((
                Mesh3d(shaft_mesh.clone()),
                MeshMaterial3d(mat.clone()),
                Transform {
                    translation: rot * (Vec3::Z * (shaft_len * 0.5)),
                    rotation: rot,
                    scale: Vec3::new(shaft_r, shaft_r, shaft_len),
                },
                Pickable::IGNORE,
                Name::new("triad-shaft"),
            ))
            .id();
        let tip = commands
            .spawn((
                Mesh3d(tip_mesh.clone()),
                MeshMaterial3d(mat),
                Transform {
                    translation: rot * (Vec3::Z * (shaft_len + tip_len * 0.5)),
                    rotation: rot,
                    scale: Vec3::new(tip_r, tip_r, tip_len),
                },
                Pickable::IGNORE,
                Name::new("triad-tip"),
            ))
            .id();
        commands.entity(root).add_children(&[shaft, tip]);
    }
    root
}
