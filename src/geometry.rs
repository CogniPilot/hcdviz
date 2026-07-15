//! Primitive geometry → Bevy meshes, plus mesh/material caches so identical primitives and colors
//! share GPU handles (batching). Also the FOV frustum builders ported from dendrite-viewer
//! (`create_frustum_mesh` / `create_conical_frustum_mesh`, modernized to Bevy 0.19 `bevy::mesh`).
//!
//! HCDF keeps primitive dimensions as authored text (e.g. `<size>0.048 0.044 0.012</size>`), so this
//! module owns the parse-from-string helpers too. Unparseable/missing dims fall back to a small
//! sensible default rather than panicking.
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::schema::model::connectivity::PrimitiveRepresentation as ConnectivityPrimitive;
use crate::schema::model::geometry::{
    Capsule as SCapsule, CollisionGeometry, Cone as SCone, Cylinder as SCylinder,
    Ellipsoid as SEllipsoid, Frustum as SFrustum, Geometry as SGeometry, Sphere as SSphere,
    VisualGeometry,
};
use crate::schema::model::{LidarParams, LidarScanAxis, RangeValue};

/// A resolved primitive: a Bevy mesh plus the local scale to apply (ellipsoid renders as a unit
/// sphere scaled by its radii).
pub struct ResolvedMesh {
    pub mesh: Handle<Mesh>,
    pub scale: Vec3,
}

/// Cache keying identical primitive/frustum meshes (by a quantized parameter key) and materials (by
/// quantized rgba plus an unlit flag bit) so they share handles. Stored as a Bevy resource and reused
/// across the whole scene build, and across rebuilds, so republished docs stop churning assets.
#[derive(Resource, Default)]
pub struct GeometryCache {
    meshes: HashMap<String, Handle<Mesh>>,
    materials: HashMap<u64, Handle<StandardMaterial>>,
}

impl GeometryCache {
    /// Get-or-insert a mesh built by `f`, keyed by `key`.
    fn mesh(
        &mut self,
        key: String,
        meshes: &mut Assets<Mesh>,
        f: impl FnOnce() -> Mesh,
    ) -> Handle<Mesh> {
        self.meshes
            .entry(key)
            .or_insert_with(|| meshes.add(f()))
            .clone()
    }

    /// Get-or-insert an opaque standard material for an sRGB color, deduped by quantized rgba.
    pub fn material(
        &mut self,
        rgba: [f32; 4],
        materials: &mut Assets<StandardMaterial>,
    ) -> Handle<StandardMaterial> {
        let key = color_key(rgba);
        self.materials
            .entry(key)
            .or_insert_with(|| {
                let alpha = rgba[3];
                let mut m = StandardMaterial {
                    base_color: Color::srgba(rgba[0], rgba[1], rgba[2], alpha),
                    perceptual_roughness: 0.7,
                    metallic: 0.1,
                    ..default()
                };
                if alpha < 1.0 {
                    m.alpha_mode = AlphaMode::Blend;
                }
                materials.add(m)
            })
            .clone()
    }

    /// Get-or-insert the translucent, double-sided, unlit material the FOV/connector glyphs use,
    /// deduped by quantized rgba with the private `UNLIT_FLAG` bit set (so it can never collide with
    /// [`Self::material`]'s lit entries). These were previously freshly `add`ed per scene build,
    /// freed on despawn, so pure churn on every rebuild.
    pub fn material_unlit(
        &mut self,
        rgba: [f32; 4],
        materials: &mut Assets<StandardMaterial>,
    ) -> Handle<StandardMaterial> {
        let key = color_key(rgba) | UNLIT_FLAG;
        self.materials
            .entry(key)
            .or_insert_with(|| {
                materials.add(StandardMaterial {
                    base_color: Color::srgba(rgba[0], rgba[1], rgba[2], rgba[3]),
                    alpha_mode: AlphaMode::Blend,
                    cull_mode: None,
                    unlit: true,
                    ..default()
                })
            })
            .clone()
    }
}

/// Material-key flag bit distinguishing the unlit/no-cull variant from the lit one for the same rgba
/// ([`color_key`] uses bits 0..48; this sets bit 48).
const UNLIT_FLAG: u64 = 1 << 48;

fn color_key(c: [f32; 4]) -> u64 {
    let q = |x: f32| (x.clamp(0.0, 1.0) * 4095.0).round() as u64;
    (q(c[0]) << 36) | (q(c[1]) << 24) | (q(c[2]) << 12) | q(c[3])
}

/// Parse a whitespace-separated float list (e.g. `"0.048 0.044 0.012"`), dropping any non-finite
/// component (NaN/inf). This mirrors the scalar helper [`f1`]: a non-finite dimension must never reach a
/// Bevy mesh (a NaN box/ellipsoid extent corrupts the whole render). Dropping the bad component leaves the
/// count wrong, so [`vec3_text`] (and [`crate::joints::parse_axis`]) then fall back to their default.
pub fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .collect()
}

fn f1(s: &Option<String>, default: f32) -> f32 {
    s.as_deref()
        .and_then(|t| t.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn vec3_text(s: &Option<String>, default: Vec3) -> Vec3 {
    match s {
        Some(t) => {
            let v = parse_floats(t);
            if v.len() == 3 {
                Vec3::new(v[0], v[1], v[2])
            } else {
                default
            }
        }
        None => default,
    }
}

// ── primitive mesh resolution ────────────────────────────────────────────────

/// The set of primitive shapes shared by all three HCDF geometry containers.
enum Prim {
    Box(Vec3),
    Cylinder { radius: f32, length: f32 },
    Sphere { radius: f32 },
    Capsule { radius: f32, length: f32 },
    Cone { radius: f32, length: f32 },
    Ellipsoid { radii: Vec3 },
}

impl Prim {
    fn cache_key(&self) -> String {
        let q = |x: f32| (x * 1e5).round() as i64;
        match self {
            Prim::Box(s) => format!("box:{}:{}:{}", q(s.x), q(s.y), q(s.z)),
            Prim::Cylinder { radius, length } => format!("cyl:{}:{}", q(*radius), q(*length)),
            Prim::Sphere { radius } => format!("sph:{}", q(*radius)),
            Prim::Capsule { radius, length } => format!("cap:{}:{}", q(*radius), q(*length)),
            Prim::Cone { radius, length } => format!("cone:{}:{}", q(*radius), q(*length)),
            // ellipsoid shares the unit-sphere mesh; scale carries the radii.
            Prim::Ellipsoid { .. } => "ellipsoid-unit".to_string(),
        }
    }

    fn build(&self) -> Mesh {
        match self {
            Prim::Box(s) => Cuboid::new(s.x, s.y, s.z).into(),
            Prim::Cylinder { radius, length } => z_aligned(Cylinder::new(*radius, *length).into()),
            Prim::Sphere { radius } => Sphere::new(*radius).into(),
            Prim::Capsule { radius, length } => z_aligned(Capsule3d::new(*radius, *length).into()),
            Prim::Cone { radius, length } => z_aligned(Cone::new(*radius, *length).into()),
            Prim::Ellipsoid { .. } => Sphere::new(1.0).into(),
        }
    }

    fn local_scale(&self) -> Vec3 {
        match self {
            Prim::Ellipsoid { radii } => *radii,
            _ => Vec3::ONE,
        }
    }
}

/// Bake the HCDF axis convention into a Bevy Y-aligned primitive mesh. HCDF (like URDF/SDF) aligns
/// cylinder/capsule length and the cone base→apex axis with **Z**, but Bevy's `Cylinder`/`Capsule3d`/
/// `Cone` primitives extend along **Y** (cone tip toward +Y), rendering e.g. a lidar `<cylinder>`
/// sideways. Rotating the vertex data (positions, normals, tangents) by +90° about X (+Y→+Z, so the
/// cone apex lands on +Z per the schema "tapering to a point … along the Z axis") fixes the mesh
/// itself once at build time: every consumer (visuals, collision overlays, FOV/connector fallbacks,
/// and the mesh-raycast picker) inherits the correct orientation, with no per-entity compensating
/// transform for gizmos or selection to fight. Box/sphere/ellipsoid need no bake (per-axis or
/// symmetric), and the frustum builders below already construct their vertices along +Z.
fn z_aligned(mesh: Mesh) -> Mesh {
    mesh.rotated_by(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
}

fn prim_box(size: &Option<String>) -> Prim {
    Prim::Box(vec3_text(size, Vec3::splat(0.05)))
}
fn prim_cyl(c: &SCylinder) -> Prim {
    Prim::Cylinder {
        radius: f1(&c.radius, 0.025),
        length: f1(&c.length, 0.05),
    }
}
fn prim_sphere(s: &SSphere) -> Prim {
    Prim::Sphere {
        radius: f1(&s.radius, 0.025),
    }
}
fn prim_capsule(c: &SCapsule) -> Prim {
    Prim::Capsule {
        radius: f1(&c.radius, 0.02),
        length: f1(&c.length, 0.05),
    }
}
fn prim_cone(c: &SCone) -> Prim {
    Prim::Cone {
        radius: f1(&c.radius, 0.025),
        length: f1(&c.length, 0.05),
    }
}
fn prim_ellipsoid(e: &SEllipsoid) -> Prim {
    Prim::Ellipsoid {
        radii: vec3_text(&e.radii, Vec3::splat(0.025)),
    }
}

fn resolve(cache: &mut GeometryCache, meshes: &mut Assets<Mesh>, prim: Prim) -> ResolvedMesh {
    let scale = prim.local_scale();
    let key = prim.cache_key();
    let mesh = cache.mesh(key, meshes, || prim.build());
    ResolvedMesh { mesh, scale }
}

/// Get-or-build the shared UNIT cylinder mesh (radius 1, length 1, Z-aligned) for glyph geometry
/// such as the selection-triad arrow shafts. Callers size it per use via a non-uniform
/// `Transform::scale` (`(r, r, length)`), so every arrow shares this one cached mesh instead of
/// churning a new asset per size. Rendered with unlit or lit materials alike; non-uniform scale is
/// fine because the mesh's normals stay axis-aligned under axis-aligned scaling.
pub fn unit_cylinder(cache: &mut GeometryCache, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
    resolve(
        cache,
        meshes,
        Prim::Cylinder {
            radius: 1.0,
            length: 1.0,
        },
    )
    .mesh
}

/// Get-or-build the shared UNIT cone mesh (base radius 1, length 1, apex toward +Z) for glyph
/// geometry such as the selection-triad arrow tips. Sized per use via `Transform::scale`
/// (`(r, r, length)`), sharing one cached mesh across all arrows (see [`unit_cylinder`]).
pub fn unit_cone(cache: &mut GeometryCache, meshes: &mut Assets<Mesh>) -> Handle<Mesh> {
    resolve(
        cache,
        meshes,
        Prim::Cone {
            radius: 1.0,
            length: 1.0,
        },
    )
    .mesh
}

/// Resolve one canonical connectivity primitive using the same cached, Z-aligned meshes as
/// structural visuals. Connectivity dimensions are already parsed and validated by `hcdformat`, so
/// no string fallback is involved here.
pub fn resolve_connectivity_primitive(
    primitive: &ConnectivityPrimitive,
    cache: &mut GeometryCache,
    meshes: &mut Assets<Mesh>,
) -> ResolvedMesh {
    let primitive = match primitive {
        ConnectivityPrimitive::Box { size } => {
            Prim::Box(Vec3::new(size[0] as f32, size[1] as f32, size[2] as f32))
        }
        ConnectivityPrimitive::Cylinder { radius, length } => Prim::Cylinder {
            radius: *radius as f32,
            length: *length as f32,
        },
        ConnectivityPrimitive::Sphere { radius } => Prim::Sphere {
            radius: *radius as f32,
        },
    };
    resolve(cache, meshes, primitive)
}

/// Resolve a `<visual>` primitive (box/cyl/sphere/capsule/cone/ellipsoid; no mesh/frustum).
pub fn resolve_visual_geometry(
    g: &VisualGeometry,
    cache: &mut GeometryCache,
    meshes: &mut Assets<Mesh>,
) -> Option<ResolvedMesh> {
    let prim = first_visual_prim(g)?;
    Some(resolve(cache, meshes, prim))
}

fn first_visual_prim(g: &VisualGeometry) -> Option<Prim> {
    if let Some(b) = &g.box_ {
        Some(prim_box(&b.size))
    } else if let Some(c) = &g.cylinder {
        Some(prim_cyl(c))
    } else if let Some(s) = &g.sphere {
        Some(prim_sphere(s))
    } else if let Some(c) = &g.capsule {
        Some(prim_capsule(c))
    } else if let Some(c) = &g.cone {
        Some(prim_cone(c))
    } else {
        g.ellipsoid.as_ref().map(prim_ellipsoid)
    }
}

/// Resolve a `<collision_geometry>` primitive (visual prims + mesh, with the mesh handled by the caller).
pub fn resolve_collision_geometry(
    g: &CollisionGeometry,
    cache: &mut GeometryCache,
    meshes: &mut Assets<Mesh>,
) -> Option<ResolvedMesh> {
    let prim = if let Some(b) = &g.box_ {
        prim_box(&b.size)
    } else if let Some(c) = &g.cylinder {
        prim_cyl(c)
    } else if let Some(s) = &g.sphere {
        prim_sphere(s)
    } else if let Some(c) = &g.capsule {
        prim_capsule(c)
    } else if let Some(c) = &g.cone {
        prim_cone(c)
    } else if let Some(e) = &g.ellipsoid {
        prim_ellipsoid(e)
    } else {
        return None;
    };
    Some(resolve(cache, meshes, prim))
}

/// Resolve a general `<geometry>` primitive (prims + mesh + frustum; mesh/frustum handled separately).
pub fn resolve_general_primitive(
    g: &SGeometry,
    cache: &mut GeometryCache,
    meshes: &mut Assets<Mesh>,
) -> Option<ResolvedMesh> {
    let prim = if let Some(b) = &g.box_ {
        prim_box(&b.size)
    } else if let Some(c) = &g.cylinder {
        prim_cyl(c)
    } else if let Some(s) = &g.sphere {
        prim_sphere(s)
    } else if let Some(c) = &g.capsule {
        prim_capsule(c)
    } else if let Some(c) = &g.cone {
        prim_cone(c)
    } else if let Some(e) = &g.ellipsoid {
        prim_ellipsoid(e)
    } else {
        return None;
    };
    Some(resolve(cache, meshes, prim))
}

// ── FOV frustum meshes (ported from dendrite-viewer models.rs, modernized) ───

/// VIZ-ONLY display cap (metres) for a camera FOV frustum's drawn depth, mirroring
/// [`LIDAR_DISPLAY_CAP_M`]: a `<far>` clip plane of e.g. 100 m would otherwise draw a 100 m solid
/// pyramid that dwarfs the robot. The authored `<far>` is NEVER mutated (hfov/vfov/near all preserved);
/// only the DRAWN far is clamped; the "full sensor extents" toggle
/// ([`crate::pick::SensorVizGlobal`]/[`crate::pick::SensorVizState::full_extent`]) draws at the true
/// far instead.
pub const FOV_DISPLAY_CAP_M: f32 = 2.0;

/// Resolved frustum parameters (authored text parsed, defaults applied): both the mesh-build input and
/// the cache identity, so key and geometry can never drift apart (mirrors [`Prim`]). `pyramidal`
/// frustums use hfov/vfov (rectangular cross-section); `conical` uses a single fov (circular).
enum FrustumParams {
    Pyramidal {
        near: f32,
        far: f32,
        hfov: f32,
        vfov: f32,
    },
    Conical {
        near: f32,
        far: f32,
        half_fov: f32,
    },
}

impl FrustumParams {
    /// `full_extent` draws the true authored `<far>`; otherwise the drawn far is capped at
    /// [`FOV_DISPLAY_CAP_M`] (viz only; the doc is never touched). The cap can never pull `far` behind
    /// `near`, so a degenerate frustum is avoided.
    fn resolve(f: &SFrustum, full_extent: bool) -> Self {
        let near = f1(&f.near, 0.05);
        let far_raw = f1(&f.far, 1.0);
        let far = if full_extent {
            far_raw
        } else {
            far_raw.min(FOV_DISPLAY_CAP_M)
        }
        .max(near + 1e-3);
        // `@shape` is a typed `FrustumShape` enum now; default (absent) is pyramidal.
        if matches!(
            f.shape,
            Some(crate::schema::model::enums::FrustumShape::Conical)
        ) {
            FrustumParams::Conical {
                near,
                far,
                half_fov: f1(&f.fov, 1.0) * 0.5,
            }
        } else {
            FrustumParams::Pyramidal {
                near,
                far,
                hfov: f1(&f.hfov, f1(&f.fov, 1.0)),
                vfov: f1(&f.vfov, f1(&f.fov, 0.8)),
            }
        }
    }

    fn cache_key(&self) -> String {
        let q = |x: f32| (x * 1e5).round() as i64;
        match self {
            FrustumParams::Pyramidal {
                near,
                far,
                hfov,
                vfov,
            } => {
                format!("fru-pyr:{}:{}:{}:{}", q(*near), q(*far), q(*hfov), q(*vfov))
            }
            FrustumParams::Conical {
                near,
                far,
                half_fov,
            } => {
                format!("fru-con:{}:{}:{}", q(*near), q(*far), q(*half_fov))
            }
        }
    }

    fn build(&self) -> Mesh {
        match *self {
            FrustumParams::Pyramidal {
                near,
                far,
                hfov,
                vfov,
            } => create_pyramidal_frustum_mesh(near, far, hfov, vfov),
            FrustumParams::Conical {
                near,
                far,
                half_fov,
            } => create_conical_frustum_mesh(near, far, half_fov),
        }
    }
}

/// Resolve a frustum-bearing general `<geometry>` to a cached [`Mesh`] handle: identical resolved
/// parameters share one mesh across scene rebuilds (like the primitives) instead of adding a fresh
/// asset per build. Returns `None` when the geometry carries no `<frustum>`.
///
/// The mesh is in the FOV-local frame pointing along **+Z** (optical forward).
pub fn resolve_frustum_mesh(
    g: &SGeometry,
    full_extent: bool,
    cache: &mut GeometryCache,
    meshes: &mut Assets<Mesh>,
) -> Option<Handle<Mesh>> {
    let f = g.frustum.as_ref()?;
    let params = FrustumParams::resolve(f, full_extent);
    Some(cache.mesh(params.cache_key(), meshes, || params.build()))
}

/// Rectangular (pyramidal) frustum: 8 verts, 12 tris, pointing +Z. (dendrite `create_frustum_mesh`).
///
/// POSITION + indices only: FOV frustums always render unlit (base color, no lighting or texture),
/// so normal/UV attributes would be dead weight; the legacy port carried them, but rendered unlit
/// too. Same for the conical builder below.
fn create_pyramidal_frustum_mesh(near: f32, far: f32, hfov: f32, vfov: f32) -> Mesh {
    let near_half_w = near * (hfov / 2.0).tan();
    let near_half_h = near * (vfov / 2.0).tan();
    let far_half_w = far * (hfov / 2.0).tan();
    let far_half_h = far * (vfov / 2.0).tan();

    let vertices: Vec<[f32; 3]> = vec![
        [-near_half_w, -near_half_h, near],
        [near_half_w, -near_half_h, near],
        [near_half_w, near_half_h, near],
        [-near_half_w, near_half_h, near],
        [-far_half_w, -far_half_h, far],
        [far_half_w, -far_half_h, far],
        [far_half_w, far_half_h, far],
        [-far_half_w, far_half_h, far],
    ];
    let indices: Vec<u32> = vec![
        0, 2, 1, 0, 3, 2, // near
        4, 5, 6, 4, 6, 7, // far
        0, 1, 5, 0, 5, 4, // bottom
        3, 6, 2, 3, 7, 6, // top
        0, 4, 7, 0, 7, 3, // left
        1, 2, 6, 1, 6, 5, // right
    ];

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_indices(Indices::U32(indices))
}

/// Circular (conical) truncated-cone frustum pointing +Z. `half_fov` is the half-angle in radians.
/// (dendrite `create_conical_frustum_mesh`.)
fn create_conical_frustum_mesh(near: f32, far: f32, half_fov: f32) -> Mesh {
    const SEGMENTS: usize = 24;
    let near_radius = near * half_fov.tan();
    let far_radius = far * half_fov.tan();

    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(SEGMENTS * 2 + 2);

    for i in 0..SEGMENTS {
        let angle = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
        let (sin_a, cos_a) = angle.sin_cos();
        vertices.push([cos_a * near_radius, sin_a * near_radius, near]);
    }
    for i in 0..SEGMENTS {
        let angle = (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;
        let (sin_a, cos_a) = angle.sin_cos();
        vertices.push([cos_a * far_radius, sin_a * far_radius, far]);
    }
    vertices.push([0.0, 0.0, near]);
    vertices.push([0.0, 0.0, far]);

    let near_center = (SEGMENTS * 2) as u32;
    let far_center = (SEGMENTS * 2 + 1) as u32;
    let mut indices: Vec<u32> = Vec::with_capacity(SEGMENTS * 12);
    for i in 0..SEGMENTS {
        let next = (i + 1) % SEGMENTS;
        let n0 = i as u32;
        let n1 = next as u32;
        let f0 = (SEGMENTS + i) as u32;
        let f1 = (SEGMENTS + next) as u32;
        indices.extend_from_slice(&[n0, f0, f1, n0, f1, n1]);
        indices.extend_from_slice(&[near_center, n1, n0]);
        indices.extend_from_slice(&[far_center, f0, f1]);
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_indices(Indices::U32(indices))
}

// ── LIDAR scan-extent viz (translucent annulus / sector / conical band) ───────

/// VIZ-ONLY display cap (metres) for a lidar's drawn radius, so a long-range planar scan (e.g. a 20 m
/// lidar on a 0.34 m robot) doesn't swamp the scene. The sensor's true `<range><max>` is NEVER mutated;
/// this only clamps the DRAWN annulus radius; the "full sensor extents" toggle
/// ([`crate::pick::SensorVizGlobal`]/[`crate::pick::SensorVizState::full_extent`]) draws at the uncapped
/// range instead.
pub const LIDAR_DISPLAY_CAP_M: f32 = 2.5;

/// Azimuth/elevation tessellation for a sweep of `span` radians: resolution scales with the span so a
/// full 360° ring gets the full budget and a narrow sector stays cheap. Always ≥1.
fn arc_steps(span: f32) -> usize {
    const ARC_STEPS: usize = 48;
    let steps = ((span.abs() / std::f32::consts::TAU) * ARC_STEPS as f32).ceil() as usize;
    steps.clamp(1, ARC_STEPS)
}

/// A unit-direction closure mapping a sweep angle (radians) to a point on the unit circle in the scan
/// plane: the body XY plane for an azimuth sweep, the XZ plane for a vertical (elevation) line-scan.
type ScanDir = fn(f32) -> Vec3;

/// One angular axis of a resolved lidar scan sweep (radians), parsed from a [`LidarScanAxis`].
#[derive(Clone, Copy)]
struct ScanAxis {
    min: f32,
    max: f32,
}

impl ScanAxis {
    /// Parse `@min-angle`/`@max-angle` (radians). `None` when neither is authored, so nothing to draw.
    /// A degenerate/zero sweep (both angles equal) is also treated as absent.
    fn resolve(a: &LidarScanAxis) -> Option<Self> {
        if a.min_angle.is_none() && a.max_angle.is_none() {
            return None;
        }
        let min = f1(&a.min_angle, 0.0);
        let max = f1(&a.max_angle, 0.0);
        (max > min).then_some(ScanAxis { min, max })
    }

    /// Parse a `<vertical-fov>` [`RangeValue`] (`@min`/`@max`, radians) as an elevation sweep: the
    /// fallback vertical extent when a lidar authors `<vertical-fov>` but no `<scan-pattern><vertical>`.
    fn from_range(r: &RangeValue) -> Option<Self> {
        if r.min.is_none() && r.max.is_none() {
            return None;
        }
        let min = f1(&r.min, 0.0);
        let max = f1(&r.max, 0.0);
        (max > min).then_some(ScanAxis { min, max })
    }

    fn span(&self) -> f32 {
        self.max - self.min
    }

    /// True when the sweep covers (approximately) a full 360°: a closed ring, no radial cap edges.
    fn is_full(&self) -> bool {
        self.span() >= std::f32::consts::TAU - 1e-3
    }
}

/// A resolved lidar scan volume: the sweep extent along the horizontal (azimuth, about body +Z) and
/// optional vertical (elevation, about body +Y) axes, plus the along-beam range (`near`..`far`). Both
/// the mesh-build input and the cache identity, so the key and geometry never drift apart (mirrors
/// [`FrustumParams`]). The DRAWN outer radius is [`Self::display_radius`] (capped unless full-extent);
/// `far` here is the true authored range, uncapped.
struct LidarScanViz {
    horizontal: Option<ScanAxis>,
    vertical: Option<ScanAxis>,
    /// `<range><min>`: the annulus inner radius (the un-sensed hole around the sensor). Default 0.
    near: f32,
    /// `<range><max>`: the true beam length (uncapped). Default 5 m so a range-less scan still reads.
    far: f32,
}

impl LidarScanViz {
    /// Resolve from typed `<lidar-params>`; `None` when there is no drawable angular sweep (no
    /// `<scan-pattern>`, or one whose axes carry no min/max angle). The vertical extent falls back to
    /// `<vertical-fov>` when `<scan-pattern><vertical>` is absent.
    fn resolve(params: &LidarParams) -> Option<Self> {
        let sp = params.scan_pattern.as_ref()?;
        let horizontal = sp.horizontal.as_ref().and_then(ScanAxis::resolve);
        let vertical = sp
            .vertical
            .as_ref()
            .and_then(ScanAxis::resolve)
            .or_else(|| params.vertical_fov.as_ref().and_then(ScanAxis::from_range));
        if horizontal.is_none() && vertical.is_none() {
            return None;
        }
        let range = params.range.as_ref();
        let far = range
            .and_then(|r| r.max.as_ref())
            .map(|m| f1(&m.value, 5.0))
            .filter(|v| *v > 0.0)
            .unwrap_or(5.0);
        let near = range
            .and_then(|r| r.min.as_ref())
            .map(|m| f1(&m.value, 0.0))
            .filter(|v| *v >= 0.0)
            .unwrap_or(0.0);
        Some(LidarScanViz {
            horizontal,
            vertical,
            near,
            far,
        })
    }

    /// The DRAWN outer radius: the true `far`, capped at [`LIDAR_DISPLAY_CAP_M`] unless `full_extent`.
    fn display_radius(&self, full_extent: bool) -> f32 {
        if full_extent {
            self.far
        } else {
            self.far.min(LIDAR_DISPLAY_CAP_M)
        }
    }

    /// The DRAWN inner radius (annulus hole), clamped just inside the outer radius so the cap can never
    /// invert the annulus even for a `<range><min>` larger than the display radius.
    fn inner_radius(&self, full_extent: bool) -> f32 {
        self.near
            .clamp(0.0, self.display_radius(full_extent) * 0.99)
    }

    /// Keyed on the effective (post-cap) DRAWN extent, so a capped and an uncapped scan that land on the
    /// same radius share one mesh, while a full-extent toggle re-resolves to a distinct mesh.
    fn cache_key(&self, full_extent: bool) -> String {
        let q = |x: f32| (x * 1e5).round() as i64;
        let ax = |a: &Option<ScanAxis>| match a {
            Some(s) => format!("{}:{}", q(s.min), q(s.max)),
            None => "-".to_string(),
        };
        format!(
            "lidar:{}:{}:{}:{}",
            ax(&self.horizontal),
            ax(&self.vertical),
            q(self.inner_radius(full_extent)),
            q(self.display_radius(full_extent)),
        )
    }

    /// The primary flat annulus's sweep axis + unit-direction closure: the horizontal sweep in the body
    /// XY plane when present, else the vertical sweep in the body XZ plane (a vertical line-scan lidar).
    fn primary(&self) -> Option<(ScanAxis, ScanDir)> {
        if let Some(h) = self.horizontal {
            Some((h, |a| Vec3::new(a.cos(), a.sin(), 0.0)))
        } else {
            self.vertical
                .map(|v| (v, (|a| Vec3::new(a.cos(), 0.0, a.sin())) as ScanDir))
        }
    }

    /// The filled (translucent) `TriangleList`: the flat annulus/sector in the primary scan plane, plus
    /// (when BOTH a horizontal and a vertical sweep exist, a 3D lidar) a conical band swept between
    /// the elevation extremes at the display radius.
    fn build_fill(&self, full_extent: bool) -> Mesh {
        let r_out = self.display_radius(full_extent);
        let r_in = self.inner_radius(full_extent);
        let mut pos: Vec<[f32; 3]> = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        if let Some((axis, dir)) = self.primary() {
            push_annulus(&mut pos, &mut idx, axis, r_in, r_out, dir);
        }
        if let (Some(h), Some(v)) = (self.horizontal, self.vertical) {
            push_conical_band(&mut pos, &mut idx, h, v, r_out);
        }
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_indices(Indices::U32(idx))
    }

    /// The thin boundary `LineList` at the display radius (the definition ring): the outer arc always,
    /// plus the inner arc and the two radial cap edges when the sweep is a sector rather than a full ring.
    fn build_boundary(&self, full_extent: bool) -> Mesh {
        let r_out = self.display_radius(full_extent);
        let r_in = self.inner_radius(full_extent);
        let mut pos: Vec<[f32; 3]> = Vec::new();
        if let Some((axis, dir)) = self.primary() {
            push_boundary(&mut pos, axis, r_in, r_out, dir);
        }
        Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
    }
}

/// Append a filled annulus/sector (`TriangleList`) as a strip of quads between the inner (`r_in`) and
/// outer (`r_out`) arcs of `axis`, each vertex placed by `dir(angle) * radius`.
fn push_annulus(
    pos: &mut Vec<[f32; 3]>,
    idx: &mut Vec<u32>,
    axis: ScanAxis,
    r_in: f32,
    r_out: f32,
    dir: impl Fn(f32) -> Vec3,
) {
    let steps = arc_steps(axis.span());
    let base = pos.len() as u32;
    for k in 0..=steps {
        let a = axis.min + axis.span() * (k as f32 / steps as f32);
        let d = dir(a);
        pos.push((d * r_in).to_array());
        pos.push((d * r_out).to_array());
    }
    for k in 0..steps {
        let i = base + k as u32 * 2;
        // Quad {inner_k, outer_k, inner_k+1, outer_k+1} = (i, i+1, i+2, i+3), two CCW tris.
        idx.extend_from_slice(&[i, i + 1, i + 3, i, i + 3, i + 2]);
    }
}

/// Append the conical BAND (`TriangleList`) a 3D lidar sweeps between elevation extremes: a strip of
/// quads between the `elev.min` and `elev.max` rings at radius `r`, over the `azimuth` sweep. Beam tip =
/// `r · (cosθ·cosφ, cosθ·sinφ, sinθ)` for elevation θ, azimuth φ: the "inclined ring pair".
fn push_conical_band(
    pos: &mut Vec<[f32; 3]>,
    idx: &mut Vec<u32>,
    azimuth: ScanAxis,
    elev: ScanAxis,
    r: f32,
) {
    let steps = arc_steps(azimuth.span());
    let base = pos.len() as u32;
    let tip = |phi: f32, theta: f32| {
        let (se, ce) = theta.sin_cos();
        let (sa, ca) = phi.sin_cos();
        Vec3::new(r * ce * ca, r * ce * sa, r * se)
    };
    for k in 0..=steps {
        let a = azimuth.min + azimuth.span() * (k as f32 / steps as f32);
        pos.push(tip(a, elev.min).to_array());
        pos.push(tip(a, elev.max).to_array());
    }
    for k in 0..steps {
        let i = base + k as u32 * 2;
        idx.extend_from_slice(&[i, i + 1, i + 3, i, i + 3, i + 2]);
    }
}

/// Append the boundary `LineList` for a flat annulus/sector: the outer definition ring always, plus (for
/// a sector, i.e. a sweep that is NOT a full ring) the inner ring and the two radial cap edges.
fn push_boundary(
    pos: &mut Vec<[f32; 3]>,
    axis: ScanAxis,
    r_in: f32,
    r_out: f32,
    dir: impl Fn(f32) -> Vec3,
) {
    let steps = arc_steps(axis.span());
    let pt = |a: f32, r: f32| (dir(a) * r).to_array();
    let angle = |k: usize| axis.min + axis.span() * (k as f32 / steps as f32);
    for k in 0..steps {
        pos.push(pt(angle(k), r_out));
        pos.push(pt(angle(k + 1), r_out));
    }
    if !axis.is_full() {
        pos.push(pt(axis.min, r_in));
        pos.push(pt(axis.min, r_out));
        pos.push(pt(axis.max, r_in));
        pos.push(pt(axis.max, r_out));
        for k in 0..steps {
            pos.push(pt(angle(k), r_in));
            pos.push(pt(angle(k + 1), r_in));
        }
    }
}

/// The two cached meshes a lidar scan renders: the translucent filled [`LidarScanMeshes::fill`] and the
/// thin [`LidarScanMeshes::boundary`] definition ring, spawned as sibling entities under the sensor.
pub struct LidarScanMeshes {
    pub fill: Handle<Mesh>,
    pub boundary: Handle<Mesh>,
}

/// Resolve a lidar/ray sensor's `<lidar-params>` scan volume to cached [`Mesh`] handles (a translucent
/// filled annulus/sector/band plus a boundary ring, at the sensor origin in the sensor BODY frame:
/// forward +X, left +Y, up +Z). `full_extent` draws the true range; otherwise the radius is capped at
/// [`LIDAR_DISPLAY_CAP_M`]. Identical scan volumes share meshes across rebuilds. `None` when the sensor
/// carries no drawable angular sweep.
pub fn resolve_lidar_scan_mesh(
    params: &LidarParams,
    full_extent: bool,
    cache: &mut GeometryCache,
    meshes: &mut Assets<Mesh>,
) -> Option<LidarScanMeshes> {
    let viz = LidarScanViz::resolve(params)?;
    let base = viz.cache_key(full_extent);
    let fill = cache.mesh(format!("{base}:fill"), meshes, || {
        viz.build_fill(full_extent)
    });
    let boundary = cache.mesh(format!("{base}:ring"), meshes, || {
        viz.build_boundary(full_extent)
    });
    Some(LidarScanMeshes { fill, boundary })
}

/// Rotation taking a FOV frustum mesh from its authored **optical** frame (+Z forward, +X right, +Y
/// down) into the sensor **body** frame (+X forward, +Y left, +Z up): the REP-103 optical→body basis.
///
/// The frustum meshes are built pointing along +Z, but the HCDF/SDF sensor `<pose>` places the sensor
/// BODY frame, whose viewing axis is +X. Without this reorientation the frustum projects out of the
/// sensor's +Z (up) instead of its +X (front). The rectangular (pyramidal) frustum's hfov spans mesh
/// +X and vfov spans mesh +Y, so mapping optical→body also lands hfov on body left/right (±Y) and
/// vfov on body up/down (±Z), keeping the aspect correct.
pub fn frustum_optical_to_body_rotation() -> Quat {
    // Columns = images of the optical basis vectors expressed in the body frame:
    //   optical +X (right)   → body −Y
    //   optical +Y (down)    → body −Z
    //   optical +Z (forward) → body +X
    Quat::from_mat3(&Mat3::from_cols(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(1.0, 0.0, 0.0),
    ))
}

/// Apply a uniform/per-axis scale on top of a transform's existing scale (used for mesh `@scale`).
pub fn with_extra_scale(t: Transform, scale: Vec3) -> Transform {
    let mut t = t;
    t.scale *= scale;
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    const EPS: f32 = 1e-4;

    fn positions(mesh: &Mesh) -> Vec<Vec3> {
        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("mesh has no Float32x3 position attribute");
        };
        pos.iter().map(|p| Vec3::from_array(*p)).collect()
    }

    /// Per-axis extents (max − min) of a mesh's vertex positions.
    fn extents(mesh: &Mesh) -> Vec3 {
        let (min, max) = positions(mesh).iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(min, max), &v| (min.min(v), max.max(v)),
        );
        max - min
    }

    /// HCDF/URDF/SDF cylinders are Z-axis aligned; Bevy's `Cylinder` primitive is Y-aligned. The
    /// built mesh must carry its LENGTH along Z (the regression: an SDF lidar `<cylinder>` rendered
    /// sideways), and its cap normals along ±Z, so the bake covers normals too, not just positions.
    #[test]
    fn cylinder_mesh_length_lies_along_z() {
        let mesh = Prim::Cylinder {
            radius: 0.05,
            length: 0.5,
        }
        .build();
        let e = extents(&mesh);
        assert!(
            (e.z - 0.5).abs() < EPS,
            "length must span Z, got extents {e:?}"
        );
        assert!(
            (e.x - 0.1).abs() < 1e-3 && (e.y - 0.1).abs() < 1e-3,
            "diameter on X/Y, got {e:?}"
        );
        assert!(
            e.z > e.x && e.z > e.y,
            "max extent must lie along Z, got {e:?}"
        );
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("cylinder mesh has no normals");
        };
        assert!(
            normals
                .iter()
                .any(|n| Vec3::from_array(*n).distance(Vec3::Z) < EPS),
            "flat caps must face ±Z after the bake"
        );
    }

    /// Capsule: Z-aligned like the cylinder; total height = length + 2·radius (schema doc).
    #[test]
    fn capsule_mesh_length_lies_along_z() {
        let e = extents(
            &Prim::Capsule {
                radius: 0.05,
                length: 0.3,
            }
            .build(),
        );
        assert!(
            (e.z - 0.4).abs() < 1e-3,
            "length + 2r must span Z, got extents {e:?}"
        );
        assert!(
            (e.x - 0.1).abs() < 1e-3 && (e.y - 0.1).abs() < 1e-3,
            "diameter on X/Y, got {e:?}"
        );
        assert!(
            e.z > e.x && e.z > e.y,
            "max extent must lie along Z, got {e:?}"
        );
    }

    /// Cone: "base circle … tapering to a point at the given length along the Z axis" (hcdf.xsd), i.e.
    /// apex toward +Z like SDF. Bevy's `Cone` tips toward +Y, so the bake must land the apex on the
    /// +Z axis with the base ring below it; direction matters here, unlike the symmetric shapes.
    #[test]
    fn cone_mesh_apex_points_plus_z() {
        let mesh = Prim::Cone {
            radius: 0.05,
            length: 0.4,
        }
        .build();
        let e = extents(&mesh);
        assert!(
            (e.z - 0.4).abs() < EPS,
            "height must span Z, got extents {e:?}"
        );
        assert!(
            e.z > e.x && e.z > e.y,
            "max extent must lie along Z, got {e:?}"
        );
        let pos = positions(&mesh);
        let max_z = pos.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v.z));
        let min_z = pos.iter().fold(f32::INFINITY, |m, v| m.min(v.z));
        let radial = |v: &Vec3| v.truncate().length();
        let apex_r = pos
            .iter()
            .filter(|v| (v.z - max_z).abs() < EPS)
            .map(radial)
            .fold(0.0f32, f32::max);
        let base_r = pos
            .iter()
            .filter(|v| (v.z - min_z).abs() < EPS)
            .map(radial)
            .fold(0.0f32, f32::max);
        assert!(
            apex_r < EPS,
            "apex (max Z) must sit on the Z axis, radial {apex_r}"
        );
        assert!(
            (base_r - 0.05).abs() < 1e-3,
            "base ring (min Z) at full radius, got {base_r}"
        );
    }

    use crate::schema::model::{LidarRange, LidarScanPattern, MeasuredValue};

    fn scan_axis(min: &str, max: &str) -> LidarScanAxis {
        LidarScanAxis {
            min_angle: Some(min.to_string()),
            max_angle: Some(max.to_string()),
            ..Default::default()
        }
    }

    fn lidar_params(
        h: Option<LidarScanAxis>,
        v: Option<LidarScanAxis>,
        near: &str,
        far: &str,
    ) -> LidarParams {
        LidarParams {
            scan_pattern: Some(LidarScanPattern {
                horizontal: h,
                vertical: v,
            }),
            range: Some(LidarRange {
                min: Some(MeasuredValue {
                    unit: None,
                    value: Some(near.to_string()),
                }),
                max: Some(MeasuredValue {
                    unit: None,
                    value: Some(far.to_string()),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Max radial distance in the XY plane over a mesh's vertices.
    fn max_xy_radius(mesh: &Mesh) -> f32 {
        positions(mesh)
            .iter()
            .map(|p| p.truncate().length())
            .fold(0.0f32, f32::max)
    }

    /// A full-360° horizontal sweep with a non-zero `<range><min>` builds a CLOSED filled annulus
    /// (`TriangleList`) in the body XY plane: flat in Z, outer radius at the display radius, inner
    /// radius (the hole) at `<range><min>`, and a boundary ring with NO origin vertex (a closed ring
    /// has no radial cap edges).
    #[test]
    fn lidar_full_sweep_is_a_closed_annulus() {
        // Full circle, 0.3 m inner hole, 2.0 m range (under the 2.5 m cap so it draws un-clamped).
        let p = lidar_params(Some(scan_axis("-3.14159", "3.14159")), None, "0.3", "2.0");
        let viz = LidarScanViz::resolve(&p).expect("drawable scan");
        let fill = viz.build_fill(false);
        assert_eq!(fill.primitive_topology(), PrimitiveTopology::TriangleList);
        let pos = positions(&fill);
        assert!(!pos.is_empty(), "annulus must have geometry");
        assert!(
            extents(&fill).z.abs() < EPS,
            "a horizontal-only sweep is flat in Z"
        );
        assert!(
            (max_xy_radius(&fill) - 2.0).abs() < 1e-2,
            "outer ring at the 2 m range, got {}",
            max_xy_radius(&fill)
        );
        let min_r = pos
            .iter()
            .map(|p| p.truncate().length())
            .fold(f32::INFINITY, f32::min);
        assert!(
            (min_r - 0.3).abs() < 1e-2,
            "inner ring at <range><min> = 0.3 m, got {min_r}"
        );
        // Boundary is a LineList ring; a full sweep has no radial caps ⇒ no origin/inner vertex reached.
        let ring = viz.build_boundary(false);
        assert_eq!(ring.primitive_topology(), PrimitiveTopology::LineList);
        assert!(
            !positions(&ring).iter().any(|p| p.length() < 0.2),
            "a closed ring draws no radial cap edges toward the origin"
        );
    }

    /// A limited-hfov horizontal sweep builds a filled SECTOR spanning only the swept azimuth (here the
    /// front half, all x ≥ 0), and its boundary carries the radial cap edges (reaching the origin, since
    /// the inner radius is 0).
    #[test]
    fn lidar_sector_spans_azimuth_and_has_radial_caps() {
        let p = lidar_params(
            Some(scan_axis("-1.5707963", "1.5707963")),
            None,
            "0.0",
            "2.0",
        );
        let viz = LidarScanViz::resolve(&p).expect("drawable sector");
        let fill = viz.build_fill(false);
        let pos = positions(&fill);
        assert!(
            pos.iter().all(|p| p.x >= -EPS),
            "the ±90° front sector must not sweep behind the sensor (x ≥ 0)"
        );
        assert!(pos.iter().any(|p| p.x > 1.9), "forward (+X) reached");
        assert!(
            pos.iter().any(|p| p.y > 1.9) && pos.iter().any(|p| p.y < -1.9),
            "both ±Y flanks swept"
        );
        let ring = viz.build_boundary(false);
        assert!(
            positions(&ring).iter().any(|p| p.length() < EPS),
            "a sector's radial cap edges reach the origin (inner radius 0)"
        );
    }

    /// A 3D lidar (horizontal + vertical sweep) adds a conical BAND between the elevation extremes, so
    /// the fill gains Z extent even though the flat annulus alone is planar.
    #[test]
    fn lidar_3d_scan_builds_conical_band() {
        let p = lidar_params(
            Some(scan_axis("-3.14159", "3.14159")),
            Some(scan_axis("-0.26", "0.26")),
            "0.0",
            "1.5",
        );
        let viz = LidarScanViz::resolve(&p).expect("drawable 3D scan");
        let e = extents(&viz.build_fill(false));
        assert!(
            e.z > 0.1,
            "the elevation band must give the scan volume Z extent, got {e:?}"
        );
        // The band tips lie on the range sphere at r = 1.5; the flat annulus outer ring is at 1.5 too.
        assert!(
            (max_xy_radius(&viz.build_fill(false)) - 1.5).abs() < 1e-1,
            "band/annulus at the 1.5 m range"
        );
    }

    /// A vertical-only sweep (no horizontal) is a flat sector in the body XZ plane; it stays in Y ≈ 0
    /// and gains Z extent; an angle-less or scan-pattern-less lidar draws nothing.
    #[test]
    fn lidar_vertical_only_is_flat_xz_and_empty_skips() {
        let p = lidar_params(None, Some(scan_axis("-0.26", "0.26")), "0.0", "8.0");
        let viz = LidarScanViz::resolve(&p).expect("drawable vertical scan");
        let e = extents(&viz.build_fill(false));
        assert!(e.z > 0.0, "a vertical sweep spans Z, got {e:?}");
        assert!(
            e.y.abs() < EPS,
            "vertical-only stays in the XZ plane, got {e:?}"
        );

        let empty = LidarParams {
            scan_pattern: Some(LidarScanPattern {
                horizontal: Some(LidarScanAxis::default()),
                vertical: None,
            }),
            ..Default::default()
        };
        assert!(
            LidarScanViz::resolve(&empty).is_none(),
            "no min/max angle ⇒ not drawable"
        );
        assert!(
            LidarScanViz::resolve(&LidarParams::default()).is_none(),
            "no scan-pattern ⇒ none"
        );
    }

    /// The display cap clamps a long-range scan's drawn radius to [`LIDAR_DISPLAY_CAP_M`]; the
    /// full-extent flag draws the true range instead, and the two resolve to distinct cache keys.
    #[test]
    fn lidar_display_cap_applied_and_uncapped_by_full_extent() {
        let p = lidar_params(Some(scan_axis("-3.14159", "3.14159")), None, "0.0", "20.0");
        let viz = LidarScanViz::resolve(&p).expect("drawable scan");
        let capped = max_xy_radius(&viz.build_fill(false));
        assert!(
            (capped - LIDAR_DISPLAY_CAP_M).abs() < 1e-2,
            "a 20 m scan draws capped at {LIDAR_DISPLAY_CAP_M} m, got {capped}"
        );
        let full = max_xy_radius(&viz.build_fill(true));
        assert!(
            (full - 20.0).abs() < 1e-1,
            "full-extent draws the true 20 m range, got {full}"
        );
        assert_ne!(
            viz.cache_key(false),
            viz.cache_key(true),
            "capped and full-extent meshes must cache under distinct keys"
        );
    }

    /// `<vertical-fov>` supplies the elevation extent when `<scan-pattern><vertical>` is absent.
    #[test]
    fn lidar_vertical_fov_fills_in_for_absent_scan_vertical() {
        let mut p = lidar_params(Some(scan_axis("-3.14159", "3.14159")), None, "0.0", "1.5");
        p.vertical_fov = Some(RangeValue {
            min: Some("-0.2".to_string()),
            max: Some("0.2".to_string()),
            ..Default::default()
        });
        let viz = LidarScanViz::resolve(&p).expect("drawable scan");
        assert!(
            extents(&viz.build_fill(false)).z > 0.1,
            "vertical-fov must contribute the elevation band"
        );
    }

    /// The optical→body reorientation must send the frustum's forward axis (+Z) onto the sensor body
    /// frame's viewing axis (+X), with the image horizontal (+X) landing on body left/right (±Y) and
    /// the image vertical (+Y) on body up/down (±Z). This is what makes the camera frustum project out
    /// the front of the sensor instead of straight up.
    #[test]
    fn frustum_optical_to_body_points_forward_x() {
        let r = frustum_optical_to_body_rotation();
        assert!(
            (r * Vec3::Z).distance(Vec3::X) < EPS,
            "optical +Z (forward) must map to body +X"
        );
        assert!(
            (r * Vec3::X).distance(-Vec3::Y) < EPS,
            "optical +X (right) must map to body −Y"
        );
        assert!(
            (r * Vec3::Y).distance(-Vec3::Z) < EPS,
            "optical +Y (down) must map to body −Z"
        );
        // Proper rotation (right-handed, no reflection): the mapped basis stays right-handed.
        let mapped_cross = (r * Vec3::X).cross(r * Vec3::Y);
        assert!(
            (mapped_cross - r * Vec3::Z).length() < EPS,
            "must be a proper rotation"
        );
    }

    /// A non-finite dimension must never reach mesh construction. `parse_floats` drops NaN/inf components
    /// (mirroring the scalar helper `f1`), which makes the surviving count wrong, so a NaN-size box falls
    /// back to the default cube and its built mesh has only finite vertices: the scene stays alive instead
    /// of ingesting a NaN extent.
    #[test]
    fn parse_floats_drops_non_finite_so_box_falls_back() {
        assert_eq!(
            parse_floats("1 nan 3"),
            vec![1.0, 3.0],
            "NaN component dropped"
        );
        assert_eq!(
            parse_floats("inf 2 3"),
            vec![2.0, 3.0],
            "inf component dropped"
        );
        assert!(parse_floats("1 2 3").iter().all(|v| v.is_finite()));

        // A NaN-size box resolves to the fallback cube (wrong count => default), never a NaN extent.
        let Prim::Box(size) = prim_box(&Some("1 nan 3".to_string())) else {
            panic!("expected a box prim");
        };
        assert_eq!(
            size,
            Vec3::splat(0.05),
            "a NaN size must fall back to the default cube, got {size:?}"
        );
        // And the built mesh carries only finite positions, so Bevy renders it without NaN corruption.
        let mesh = Prim::Box(size).build();
        assert!(
            positions(&mesh).iter().all(|p| p.is_finite()),
            "the fallback box mesh must have finite vertex positions"
        );
    }
}
