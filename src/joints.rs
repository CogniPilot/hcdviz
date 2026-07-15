//! Slider-driven joint articulation: pose the robot inside its joint limits to see its range of
//! motion (the rviz `robot_state_publisher`/TF analogue for HCDF).
//!
//! The whole subsystem is built around ONE externally-writable source of truth, [`JointPositions`]:
//! a `joint name → per-DOF commanded coordinates` map. Today the only writer is the "Joints" slider
//! panel ([`crate::ui::joints_panel`]). The deliberate design seam is that a FUTURE transport (a ROS
//! subscriber or a websocket bridge) writes the *same* map and the [`articulate`] system reacts
//! identically; there is no transport here now (no ROS, no sockets). Keeping the commanded state in one
//! resource (rather than wiring sliders straight to entities) is what makes that future driver a
//! drop-in.
//!
//! MULTI-DOF: every joint carries up to [`MAX_JOINT_DOF`] coordinates. A single-DOF joint
//! (revolute/continuous/prismatic/screw) uses index 0 only: byte-for-byte the old behaviour. The
//! multi-DOF kinds (cylindrical/universal/planar 2-DOF, ball 3-DOF) use the higher indices; see
//! [`joint_local_transform`] for the exact per-kind pose composition.
//!
//! Flow per frame:
//!   1. The slider panel (or, later, a topic listener) writes `name → [q0, q1, q2]` into
//!      [`JointPositions`].
//!   2. [`articulate`] (runs only when [`JointPositions`] or [`ArticulatedJoints`] changed) resolves
//!      each joint's per-DOF coordinates (honoring mimic, limits, and the continuous-joint exception)
//!      and sets the child comp entity's local [`Transform`] via the pure [`joint_local_transform`].
//!   3. Bevy's `PostUpdate` transform propagation then refreshes every `GlobalTransform`, so visuals,
//!      collision, frames, the highlight gizmo, and the kinematics skeleton all follow for free.
//!
//! [`ArticulatedJoints`] is the per-load catalogue of movable joints (entity + parsed metadata),
//! populated by the scene rebuild. Both resources reset on document reload (a new robot loads in its
//! zero pose).
use crate::doc::HcdfDoc;
use crate::schema::model::joint::{JointLimit, SwingLimit};
use crate::schema::model::KinematicState;
use crate::schema::{Hcdf, Joint};
use bevy::prelude::*;
use std::collections::HashMap;

/// Maximum number of degrees of freedom any single HCDF joint articulates. Ball is the widest
/// (3 rotational DOFs); cylindrical/universal/planar use 2, everything else 1. Fixed and free-form
/// joints use 0 (held at origin). The per-joint coordinate arrays and limit arrays are sized to this.
pub const MAX_JOINT_DOF: usize = 3;

/// Commanded joint positions: **the single writable source of truth** for the robot's pose.
///
/// Maps a joint name to its per-DOF commanded coordinates `[q0, q1, q2]` (radians for rotational DOFs,
/// metres for translational). A single-DOF joint reads/writes index 0 only; the higher indices default
/// to `0.0` (the zero pose). The slider panel writes it now; a future ROS/websocket subscriber would
/// write the same map, and [`articulate`] reacts to either writer with no other change. A missing joint
/// name defaults to all-zero. Cleared on document reload by [`reset_on_reload`].
#[derive(Resource, Default)]
pub struct JointPositions(pub HashMap<String, [f32; MAX_JOINT_DOF]>);

impl JointPositions {
    /// The commanded coordinate of DOF `i` for `name`, or `0.0` if unset (missing joint or missing DOF).
    pub fn dof(&self, name: &str, i: usize) -> f32 {
        self.0
            .get(name)
            .and_then(|q| q.get(i))
            .copied()
            .unwrap_or(0.0)
    }

    /// The full commanded coordinate array for `name`, all-zero if the joint is unset.
    pub fn dofs(&self, name: &str) -> [f32; MAX_JOINT_DOF] {
        self.0.get(name).copied().unwrap_or([0.0; MAX_JOINT_DOF])
    }

    /// Write DOF `i` for `name`, allocating a zero array on first touch so the other DOFs stay at their
    /// zero pose. Out-of-range `i` is ignored (defensive; DOF indices come from [`JointKind::dof_count`]).
    pub fn set_dof(&mut self, name: &str, i: usize, value: f32) {
        if i < MAX_JOINT_DOF {
            self.0
                .entry(name.to_string())
                .or_insert([0.0; MAX_JOINT_DOF])[i] = value;
        }
    }
}

/// Per-load catalogue of every joint that maps to a spawned tree edge, with the metadata [`articulate`]
/// needs to pose it. Rebuilt from scratch on every document load by the scene's `rebuild_on_change`.
#[derive(Resource, Default)]
pub struct ArticulatedJoints(pub Vec<ArticulatedJoint>);

/// One articulable joint: its driven child entity plus the parsed origin/axes/type/limits/mimic.
pub struct ArticulatedJoint {
    /// Joint name (the key into [`JointPositions`]).
    pub name: String,
    /// The child comp entity whose local `Transform` this joint drives.
    pub child: Entity,
    /// The joint's fixed origin transform (`joint.origin` via [`crate::frame::pose_to_transform`]).
    pub origin: Transform,
    /// Primary axis in the joint frame (normalized; defaults to +X; a zero-length axis also collapses
    /// to +X, matching the URDF default). For [`JointKind::Planar`] this is the plane NORMAL (the
    /// in-plane basis is derived from it in [`joint_local_transform`]).
    pub axis: Vec3,
    /// Secondary axis (`joint.axis2`), used only by [`JointKind::Universal`] as the second rotation
    /// axis. Normalized; defaults to +Y so a malformed universal without an `<axis2>` still yields two
    /// distinct (non-degenerate) rotation axes rather than collapsing onto the primary.
    pub axis2: Vec3,
    /// Joint kind, mapped from the typed `@type`.
    pub kind: JointKind,
    /// Per-DOF lower limits (rad or m), if declared. Index maps to the coordinate the DOF drives; see
    /// [`joint_local_transform`] for which coordinate is which per kind. `None` where no bound applies.
    pub lower: [Option<f32>; MAX_JOINT_DOF],
    /// Per-DOF upper limits (rad or m), if declared. Paired with [`Self::lower`] by index.
    pub upper: [Option<f32>; MAX_JOINT_DOF],
    /// Mimic spec, if this joint follows another.
    pub mimic: Option<JointMimic>,
}

/// The articulation-relevant joint types. `free`/unknown/absent land in [`JointKind::Other`] and are
/// held at origin. Not `Eq` because the screw pitch is an `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JointKind {
    /// Single-axis rotation, bounded by limits.
    Revolute,
    /// Single-axis rotation, unbounded (limits ignored).
    Continuous,
    /// Single-axis translation, bounded by limits.
    Prismatic,
    /// Helical motion coupling rotation and translation about/along a single axis: `q0` radians of
    /// rotation advance the joint `q0 · pitch / 2π` metres along the axis. `pitch` is metres per full
    /// revolution (the HCDF `@thread_pitch` unit; see [`SCREW_PITCH_DEFAULT`]). Slider-bounded by the
    /// joint's `[lower, upper]` limits in radians, exactly like a revolute.
    Screw { pitch: f32 },
    /// Coaxial rotation + translation about/along ONE axis, but the two are INDEPENDENT (unlike screw):
    /// `q0` = rotation (rad), `q1` = translation (m). The two DOFs carry SEPARATE, unit-distinct bounds:
    /// `<limit>` is the TRANSLATION bound (metres, `q1`) and `<limit2>` is the ROTATION bound (radians,
    /// `q0`, OPTIONAL: a telescope-that-spins leaves it absent and `q0` runs free).
    Cylindrical,
    /// Two chained rotations about crossed axes (a U-joint): `q0` about `axis`, then `q1` about `axis2`
    /// expressed in the frame left by the first rotation. `<limit>` bounds `q0`, `<limit2>` bounds `q1`.
    Universal,
    /// In-plane translation: `axis` is the plane NORMAL; `q0`/`q1` translate along a stable orthonormal
    /// in-plane basis derived from the normal. `<limit>` bounds `q0`, `<limit2>` bounds `q1`.
    Planar,
    /// Free rotation as intrinsic roll/pitch/yaw about the joint frame: `q0` about X (roll), then `q1`
    /// about Y (pitch), then `q2` about Z (yaw), each in the frame left by the previous. Bounds come from
    /// the ball's swing-cone + twist (NOT `<limit>`/`<limit2>`): `<twist_limit>` (lower/upper radians)
    /// bounds the twist `q0` about X; the `<swing_limit>` cone half-angles bound the swings: `swing2`
    /// about Y (`q1`) and `swing1` about Z (`q2`), mapped CONSERVATIVELY to symmetric `±half-angle`
    /// per-slider clamps (see [`dof_limits`]). Any absent swing/twist leaves that DOF unbounded.
    Ball,
    /// No motion (held at origin).
    Fixed,
    /// Unsupported for articulation (free/unknown); held at origin.
    Other,
}

impl JointKind {
    /// Map the typed `@type` attribute; `free`/unknown and an absent type land in [`JointKind::Other`]
    /// (held at origin). `thread_pitch` is the raw HCDF `@thread_pitch` string, consulted only for
    /// `screw` (see [`parse_screw_pitch`]).
    pub fn from_schema(
        t: Option<crate::schema::model::enums::JointType>,
        thread_pitch: Option<&str>,
    ) -> Self {
        use crate::schema::model::enums::JointType;
        match t {
            Some(JointType::Revolute) => Self::Revolute,
            Some(JointType::Continuous) => Self::Continuous,
            Some(JointType::Prismatic) => Self::Prismatic,
            Some(JointType::Screw) => Self::Screw {
                pitch: parse_screw_pitch(thread_pitch),
            },
            Some(JointType::Cylindrical) => Self::Cylindrical,
            Some(JointType::Universal) => Self::Universal,
            Some(JointType::Planar) => Self::Planar,
            Some(JointType::Ball) => Self::Ball,
            Some(JointType::Fixed) => Self::Fixed,
            _ => Self::Other,
        }
    }

    /// Number of user-driven degrees of freedom: 0 (fixed/other), 1 (revolute/continuous/prismatic/
    /// screw), 2 (cylindrical/universal/planar), or 3 (ball).
    pub fn dof_count(self) -> usize {
        match self {
            Self::Fixed | Self::Other => 0,
            Self::Ball => 3,
            Self::Cylindrical | Self::Universal | Self::Planar => 2,
            _ => 1,
        }
    }

    /// Whether this kind is user-movable (has at least one DOF, so it gets a slider / sliders).
    pub fn is_movable(self) -> bool {
        self.dof_count() > 0
    }

    /// Short per-DOF labels for the multi-DOF slider group, in DOF order. Single-DOF kinds return a
    /// one-element slice (the slider uses the joint's own name instead, so this is unused for them).
    pub fn dof_labels(self) -> &'static [&'static str] {
        match self {
            Self::Cylindrical => &["rot", "slide"],
            Self::Universal => &["rot1", "rot2"],
            Self::Planar => &["x", "y"],
            Self::Ball => &["roll", "pitch", "yaw"],
            _ => &[""],
        }
    }
}

/// Screw pitch used when a `screw` joint's `@thread_pitch` is absent or unparsable. The HCDF schema
/// REQUIRES `@thread_pitch` on a screw joint (validated upstream), so this is a defensive fallback for
/// malformed input: `0.0` metres/rev degrades the joint to pure rotation (no linear advance) rather
/// than inventing a motion, making the missing datum visible instead of silently guessed.
pub const SCREW_PITCH_DEFAULT: f32 = 0.0;

/// Parse a screw joint's `@thread_pitch` (metres per full revolution, per the HCDF schema annotation).
/// Falls back to [`SCREW_PITCH_DEFAULT`] when absent or not a finite float.
pub fn parse_screw_pitch(thread_pitch: Option<&str>) -> f32 {
    parse_f32(thread_pitch).unwrap_or(SCREW_PITCH_DEFAULT)
}

/// A mimic relationship: this joint's every DOF is `multiplier * q_source[i] + offset` (applied
/// element-wise, so a multi-DOF mimic follows its source DOF-by-DOF; single-DOF joints touch index 0).
pub struct JointMimic {
    /// Name of the source joint this one follows.
    pub source: String,
    pub multiplier: f32,
    pub offset: f32,
}

/// Build an [`ArticulatedJoint`] from a schema [`Joint`] and its already-spawned child entity + origin.
///
/// Called by the scene rebuild inside its edge loop. Parses the axes (`"x y z"` → normalized [`Vec3`];
/// `axis` defaults +X, `axis2` defaults +Y) and the per-DOF numeric limits (see [`dof_limits`]), maps
/// the typed `@type`, and lifts the mimic spec. The `name` falls back to an empty string for an unnamed
/// joint (it then simply never matches a [`JointPositions`] key and rests at origin, which is the
/// desired behaviour).
pub fn make_articulated_joint(joint: &Joint, child: Entity, origin: Transform) -> ArticulatedJoint {
    let kind = JointKind::from_schema(joint.type_, joint.thread_pitch.as_deref());
    let (lower, upper) = dof_limits(
        kind,
        joint.limit.as_ref(),
        joint.limit2.as_ref(),
        joint.swing_limit.as_ref(),
        joint.twist_limit.as_ref(),
    );
    ArticulatedJoint {
        name: joint.name.clone().unwrap_or_default(),
        child,
        origin,
        axis: parse_axis(joint.axis.as_ref().and_then(|a| a.xyz.as_deref()), Vec3::X),
        axis2: parse_axis(joint.axis2.as_ref().and_then(|a| a.xyz.as_deref()), Vec3::Y),
        kind,
        lower,
        upper,
        mimic: joint.mimic.as_ref().and_then(|m| {
            // A mimic with no source joint is meaningless; drop it so the joint stays directly driven.
            m.joint.as_deref().map(|src| JointMimic {
                source: src.to_string(),
                multiplier: parse_f32(m.multiplier.as_deref()).unwrap_or(1.0),
                offset: parse_f32(m.offset.as_deref()).unwrap_or(0.0),
            })
        }),
    }
}

/// Map a joint's per-DOF bounds onto the `[lower]`/`[upper]` arrays for its kind, honoring each type's
/// unit-distinct limit convention (the schema's joint semantics).
///
/// * revolute/prismatic/continuous → `<limit>` bounds DOF 0.
/// * screw → `<limit>` bounds DOF 0 in RADIANS (the rotational DOF; translation follows the pitch).
/// * cylindrical → `<limit2>` (rotation, rad) bounds DOF 0; `<limit>` (translation, m) bounds DOF 1.
///   The rotation `<limit2>` is optional (a telescope-that-spins leaves DOF 0 unbounded).
/// * universal → `<limit>` bounds DOF 0, `<limit2>` bounds DOF 1.
/// * planar → `<limit>` (x-box) bounds DOF 0, `<limit2>` (y-box) bounds DOF 1; `<limit2>` may be absent
///   on a URDF-imported planar (its 2nd in-plane DOF is left unbounded, not fabricated).
/// * ball → `<twist_limit>` (rad) bounds the twist DOF 0; the `<swing_limit>` cone half-angles bound the
///   swings: `swing2` → DOF 1, `swing1` → DOF 2, each mapped CONSERVATIVELY to a symmetric
///   `±half-angle` clamp. A circular cone (`swing2` omitted) reuses `swing1` for DOF 1. Absent
///   swing/twist leaves that DOF unbounded (unlimited ball).
/// * fixed/other → no bounds.
fn dof_limits(
    kind: JointKind,
    limit: Option<&JointLimit>,
    limit2: Option<&JointLimit>,
    swing: Option<&SwingLimit>,
    twist: Option<&JointLimit>,
) -> ([Option<f32>; MAX_JOINT_DOF], [Option<f32>; MAX_JOINT_DOF]) {
    match kind {
        JointKind::Cylindrical => {
            // q0 = rotation ← <limit2> (rad, optional); q1 = translation ← <limit> (m).
            let (lr, ur) = parse_limit(limit2);
            let (lt, ut) = parse_limit(limit);
            ([lr, lt, None], [ur, ut, None])
        }
        JointKind::Ball => {
            // q0 = twist (about X) ← <twist_limit>; q1 = pitch (about Y) ← swing2; q2 = yaw (about Z) ←
            // swing1. Elliptic cone half-angles map to symmetric ± per-slider clamps (conservative box
            // inside the true cone); a circular cone (swing2 omitted) reuses swing1 for the pitch DOF.
            let (lt, ut) = parse_limit(twist);
            let (s1, s2) = swing_half_angles(swing);
            let (l_pitch, u_pitch) = symmetric_half_angle(s2.or(s1));
            let (l_yaw, u_yaw) = symmetric_half_angle(s1);
            ([lt, l_pitch, l_yaw], [ut, u_pitch, u_yaw])
        }
        _ => {
            // revolute/prismatic/screw/continuous → <limit> bounds DOF 0.
            // universal/planar → <limit> bounds DOF 0, <limit2> bounds DOF 1.
            let (l0, u0) = parse_limit(limit);
            let (l1, u1) = parse_limit(limit2);
            ([l0, l1, None], [u0, u1, None])
        }
    }
}

/// Parse one `<limit>`'s lower/upper into finite floats (each independently optional).
fn parse_limit(limit: Option<&JointLimit>) -> (Option<f32>, Option<f32>) {
    match limit {
        Some(l) => (parse_f32(l.lower.as_deref()), parse_f32(l.upper.as_deref())),
        None => (None, None),
    }
}

/// Parse a ball `<swing_limit>`'s `swing1`/`swing2` cone half-angles (radians) into finite floats (each
/// independently optional; a missing `swing2` marks a circular cone, so the caller reuses `swing1`).
fn swing_half_angles(swing: Option<&SwingLimit>) -> (Option<f32>, Option<f32>) {
    match swing {
        Some(s) => (
            parse_f32(s.swing1.as_deref()),
            parse_f32(s.swing2.as_deref()),
        ),
        None => (None, None),
    }
}

/// Turn a cone half-angle (radians) into a symmetric `[-|h|, +|h|]` clamp pair: the conservative
/// per-slider bound inside the true (possibly elliptic) swing cone. `None` (no half-angle) => unbounded.
fn symmetric_half_angle(half_angle: Option<f32>) -> (Option<f32>, Option<f32>) {
    match half_angle {
        Some(h) => (Some(-h.abs()), Some(h.abs())),
        None => (None, None),
    }
}

/// Parse an `axis xyz="x y z"` string into a normalized [`Vec3`]. Missing/malformed/zero-length input
/// (and anything not exactly three finite floats) collapses to `default` (URDF's +X for the primary
/// axis; +Y for a universal's secondary axis).
fn parse_axis(xyz: Option<&str>, default: Vec3) -> Vec3 {
    let v = xyz.map(crate::geometry::parse_floats).unwrap_or_default();
    let raw = if v.len() == 3 {
        Vec3::new(v[0], v[1], v[2])
    } else {
        default
    };
    let n = raw.normalize_or_zero();
    if n.length_squared() > 1e-9 {
        n
    } else {
        default
    }
}

/// Parse one optional float attribute, rejecting non-finite values.
fn parse_f32(s: Option<&str>) -> Option<f32> {
    s.and_then(|t| t.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite())
}

/// A stable orthonormal in-plane basis `(u, v)` for a plane with the given `normal`.
///
/// Used by [`JointKind::Planar`]: `q0` translates along `u`, `q1` along `v`. The construction picks the
/// world axis least aligned with the normal as a reference, takes `u = normal × reference` and
/// `v = normal × u`, so `(u, v, normal)` is right-handed and `u`/`v` are unit and orthogonal to the
/// normal. Deterministic for a given normal (no dependence on call order), degenerate inputs fall back
/// to the XY plane.
fn plane_basis(normal: Vec3) -> (Vec3, Vec3) {
    let n = {
        let m = normal.normalize_or_zero();
        if m.length_squared() > 1e-9 {
            m
        } else {
            Vec3::Z
        }
    };
    let reference = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = {
        let c = n.cross(reference).normalize_or_zero();
        if c.length_squared() > 1e-9 {
            c
        } else {
            Vec3::X
        }
    };
    let v = n.cross(u).normalize_or_zero();
    (u, v)
}

/// PURE joint kinematics: place a child relative to its parent given the joint's commanded coordinates.
///
/// The axes are expressed in the JOINT frame and applied AFTER the fixed origin: the standard URDF
/// convention `T = origin · motion(...)` (so the motion rides along with the origin's orientation).
/// [`Transform::mul_transform`] composes self-then-arg, and each successive factor is interpreted in
/// the frame left by the previous one (intrinsic composition). `q` supplies the per-DOF coordinates in
/// DOF order; missing entries default to `0.0`.
///
///   * Revolute/Continuous → rotate `q0` about `axis`.
///   * Prismatic           → translate `axis · q0` (metres).
///   * Screw{pitch}        → rotate `q0` about `axis` and translate `axis · q0 · pitch / 2π` along it
///     (`pitch` = metres per full revolution); at `q0 = 2π` that is one full turn and exactly `pitch` m.
///   * Cylindrical         → rotate `q0` about `axis`, THEN translate `axis · q1` (independent coaxial
///     rotation and slide).
///   * Universal           → rotate `q0` about `axis`, THEN rotate `q1` about `axis2` in the
///     post-rotated frame (the standard crossed-axis U-joint composition; order is axis-then-axis2).
///   * Planar              → translate `u · q0 + v · q1`, where `(u, v)` is [`plane_basis`] of the plane
///     whose normal is `axis`.
///   * Ball                → intrinsic roll/pitch/yaw: rotate `q0` about local X, THEN `q1` about the
///     resulting local Y, THEN `q2` about the resulting local Z.
///   * Fixed/Other         → no motion; the child rests at `origin`.
///
/// `axis`/`axis2` are assumed already normalized (axes are normalized at parse time); an all-zero `q`
/// yields `origin`.
pub fn joint_local_transform(
    origin: Transform,
    axis: Vec3,
    axis2: Vec3,
    kind: JointKind,
    q: &[f32],
) -> Transform {
    let q0 = q.first().copied().unwrap_or(0.0);
    let q1 = q.get(1).copied().unwrap_or(0.0);
    let q2 = q.get(2).copied().unwrap_or(0.0);
    let rot = |a: Vec3, angle: f32| Transform::from_rotation(Quat::from_axis_angle(a, angle));
    let tr = |v: Vec3| Transform::from_translation(v);
    match kind {
        JointKind::Revolute | JointKind::Continuous => origin.mul_transform(rot(axis, q0)),
        JointKind::Prismatic => origin.mul_transform(tr(axis * q0)),
        JointKind::Screw { pitch } => origin
            .mul_transform(rot(axis, q0))
            .mul_transform(tr(axis * (q0 * pitch / std::f32::consts::TAU))),
        JointKind::Cylindrical => origin
            .mul_transform(rot(axis, q0))
            .mul_transform(tr(axis * q1)),
        JointKind::Universal => origin
            .mul_transform(rot(axis, q0))
            .mul_transform(rot(axis2, q1)),
        JointKind::Planar => {
            let (u, v) = plane_basis(axis);
            origin.mul_transform(tr(u * q0 + v * q1))
        }
        JointKind::Ball => origin
            .mul_transform(rot(Vec3::X, q0))
            .mul_transform(rot(Vec3::Y, q1))
            .mul_transform(rot(Vec3::Z, q2)),
        JointKind::Fixed | JointKind::Other => origin,
    }
}

/// Clamp one DOF's coordinate to its declared range, honoring the continuous-joint exception.
///
/// The value is clamped to `[lower[i], upper[i]]` only when BOTH bounds are present and the DOF is not
/// the unbounded rotation of a [`JointKind::Continuous`] joint (which is unbounded by definition).
/// `pub(crate)` because the loop-closure solver ([`crate::loop_solver`]) projects its iterates to the
/// SAME limit box after every step: one clamp rule, applied in two places, so the solver's write-backs
/// can never disagree with what [`articulate`] would clamp them to.
pub(crate) fn clamp_dof(
    kind: JointKind,
    i: usize,
    lower: &[Option<f32>; MAX_JOINT_DOF],
    upper: &[Option<f32>; MAX_JOINT_DOF],
    q: f32,
) -> f32 {
    if matches!(kind, JointKind::Continuous) {
        return q;
    }
    if let (Some(lo), Some(hi)) = (lower[i], upper[i]) {
        return q.clamp(lo, hi);
    }
    q
}

/// Which joints the shared resolver clamps to their declared limit box (see
/// [`resolve_with_catalogue`]). The mode threads through the whole mimic recursion, so a chain is
/// resolved under ONE consistent rule.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LimitClamp {
    /// Every joint: the display semantics [`articulate`] renders.
    Every,
    /// Mimic FOLLOWERS only; directly-commanded coordinates ride through raw, the loop-closure
    /// solver's evaluation semantics (see [`resolve_all_for_solver`] for why the asymmetry).
    MimicFollowers,
}

/// Resolve the commanded DOF-0 coordinate for one joint (mimic then limit). Convenience for single-DOF
/// callers; multi-DOF callers use [`resolve_all`], which resolves every DOF.
pub fn resolve_q(joint: &ArticulatedJoint, positions: &JointPositions) -> f32 {
    resolve_with_catalogue(joint, positions, None, &mut Vec::new(), LimitClamp::Every)[0]
}

/// Resolve every joint in a catalogue to its final per-DOF coordinates, returning a
/// `name → [q0, q1, q2]` map.
///
/// This is the chain-aware resolver [`articulate`] uses: a mimic of a mimic follows its driver's
/// resolved motion (the per-joint [`resolve_q`] only sees the raw command map, so it cannot chain
/// through a mimic source whose name is never written into [`JointPositions`]). Each joint is resolved
/// by recursing into its mimic source by name, guarding against cycles.
pub fn resolve_all(
    joints: &[ArticulatedJoint],
    positions: &JointPositions,
) -> HashMap<String, [f32; MAX_JOINT_DOF]> {
    resolve_all_impl(joints, positions, LimitClamp::Every)
}

/// [`resolve_all`] with the per-DOF limit clamp applied ONLY to mimic followers: mimic chains are
/// still followed identically, but directly-commanded coordinates ride through raw while every
/// follower's coupled value is clamped to its own limit box, exactly what [`articulate`] will
/// render for it.
///
/// Exists solely for the loop-closure solver's residual / finite-difference evaluation
/// ([`crate::loop_solver`]). The asymmetry is deliberate: one clamp rule per ROLE:
///  * FREE VARIABLES stay unclamped so the numerical gradient keeps responding exactly AT a joint
///    bound (clamping there would zero the finite-difference column and freeze the solve); the
///    solver projects its own iterates into the limit box AFTER each step instead.
///  * MIMIC FOLLOWERS are never solver variables: their value is whatever the display shows, and
///    the display clamps them. Evaluating a follower unclamped would let the solver "close" a loop
///    through a follower pose [`articulate`] refuses to render: the status would report a closed
///    mechanism while the screen shows it torn apart at the closure. A follower saturated at its
///    bound genuinely stops responding to its source: that zero gradient is the mechanism's
///    truth, not a finite-difference artifact.
///
/// Sharing the resolver (rather than duplicating it) is what guarantees the solver's view of
/// mimic semantics can never drift from [`articulate`]'s.
pub fn resolve_all_for_solver(
    joints: &[ArticulatedJoint],
    positions: &JointPositions,
) -> HashMap<String, [f32; MAX_JOINT_DOF]> {
    resolve_all_impl(joints, positions, LimitClamp::MimicFollowers)
}

/// Shared body of [`resolve_all`] / [`resolve_all_for_solver`]: catalogue-aware resolution with the
/// limit-clamp rule switchable (the [`LimitClamp`] mode threads through the whole mimic recursion,
/// so a chain resolves under one consistent rule end to end).
fn resolve_all_impl(
    joints: &[ArticulatedJoint],
    positions: &JointPositions,
    clamp: LimitClamp,
) -> HashMap<String, [f32; MAX_JOINT_DOF]> {
    let by_name: HashMap<&str, &ArticulatedJoint> =
        joints.iter().map(|j| (j.name.as_str(), j)).collect();
    joints
        .iter()
        .map(|j| {
            (
                j.name.clone(),
                resolve_with_catalogue(j, positions, Some(&by_name), &mut Vec::new(), clamp),
            )
        })
        .collect()
}

/// Shared resolver for one joint's full per-DOF coordinate array, with optional catalogue lookup for
/// chaining through mimic sources.
///
/// `catalogue` maps joint name → joint; when present, a mimic resolves its source by recursing into the
/// source joint's full resolution (mimic + limits), giving correct mimic-of-mimic behaviour. When
/// `None` (the single-joint [`resolve_q`] path), a mimic falls back to the source's raw commanded array
/// from `positions`. A mimic applies `multiplier * q_source[i] + offset` element-wise across the DOFs.
/// `visiting` holds the names currently being resolved to break cycles: a source already in that set
/// contributes all-zero, so A↔B (or longer loops) terminate with a finite pose instead of recursing
/// forever. `clamp` selects which joints get the per-DOF [`clamp_dof`]: every joint (the display
/// semantics) or mimic followers only (the solver's evaluation semantics; see
/// [`resolve_all_for_solver`]).
fn resolve_with_catalogue(
    joint: &ArticulatedJoint,
    positions: &JointPositions,
    catalogue: Option<&HashMap<&str, &ArticulatedJoint>>,
    visiting: &mut Vec<String>,
    clamp: LimitClamp,
) -> [f32; MAX_JOINT_DOF] {
    let raw = match &joint.mimic {
        Some(m) => {
            let source_q = match catalogue.and_then(|c| c.get(m.source.as_str())) {
                // Chain through the source's own resolution unless that source is already being
                // resolved (a mimic cycle), in which case its contribution is treated as all-zero.
                Some(src) if !visiting.contains(&m.source) => {
                    visiting.push(m.source.clone());
                    let r = resolve_with_catalogue(src, positions, catalogue, visiting, clamp);
                    visiting.pop();
                    r
                }
                // No catalogue (single-joint path) or a cycle: fall back to the source's raw command.
                _ => positions.dofs(&m.source),
            };
            std::array::from_fn(|i| m.multiplier * source_q[i] + m.offset)
        }
        None => positions.dofs(&joint.name),
    };
    let clamp_this = match clamp {
        LimitClamp::Every => true,
        LimitClamp::MimicFollowers => joint.mimic.is_some(),
    };
    if !clamp_this {
        return raw;
    }
    std::array::from_fn(|i| clamp_dof(joint.kind, i, &joint.lower, &joint.upper, raw[i]))
}

/// Apply [`JointPositions`] to the scene: set each articulated child's local `Transform` from its
/// resolved per-DOF coordinates. Runs in `Update` (after the scene rebuild, before `PostUpdate`
/// propagation), gated to fire only when the commanded positions or the joint catalogue actually
/// changed, so a static pose costs nothing. Transform propagation downstream carries the change to
/// every `GlobalTransform`.
pub fn articulate(
    joints: Res<ArticulatedJoints>,
    positions: Res<JointPositions>,
    mut transforms: Query<&mut Transform>,
) {
    // Resolve the whole catalogue up front so mimic-of-mimic chains follow their driver's RESOLVED q
    // (a mimic source's name is never written into JointPositions, so per-joint resolution alone could
    // not chain through it).
    let resolved = resolve_all(&joints.0, &positions);
    for j in &joints.0 {
        let q = resolved
            .get(&j.name)
            .copied()
            .unwrap_or([0.0; MAX_JOINT_DOF]);
        let target = joint_local_transform(j.origin, j.axis, j.axis2, j.kind, &q);
        // The child entity may have been despawned mid-reload on a stale catalogue; skip if so.
        if let Ok(mut t) = transforms.get_mut(j.child) {
            // Only write through change detection when the value actually moves, to avoid needlessly
            // re-propagating an unchanged transform.
            if *t != target {
                *t = target;
            }
        }
    }
}

/// Whether a `<state>` is the document default (`@default` present and truthy). Matches the hcdformat
/// validator's case-insensitive `"true"` test (`validate::check_states`) so the viewer and the
/// validator agree on which state (if any) a document loads into.
fn state_is_default(s: &KinematicState) -> bool {
    s.default
        .as_deref()
        .map(|d| d.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// The document's default `<state default="true">`, if one is declared. At most one is valid
/// (`E_MULTI_DEFAULT_STATE`); the FIRST wins defensively on a malformed multi-default doc, so a load
/// still lands in a well-defined pose rather than none.
pub fn default_state(h: &Hcdf) -> Option<&KinematicState> {
    h.state.iter().find(|s| state_is_default(s))
}

/// Seed [`JointPositions`] from a `<state>`'s joint positions, REPLACING whatever was commanded: a
/// named state is a WHOLE pose, so start from the zero pose and set only the joints it lists. Each
/// `<joint-position>` carries a single authored value, so it drives DOF 0; a multi-DOF joint keeps its
/// higher DOFs at zero (the schema has no per-DOF state value). Only the RAW command map is seeded
/// here: mimic/limit resolution rides downstream in [`articulate`]; a listed mimic follower is
/// overwritten by its source there, and every value is clamped into its limit box on apply.
pub fn apply_state(positions: &mut JointPositions, state: &KinematicState) {
    positions.0.clear();
    for jp in &state.joint_position {
        if let (Some(joint), Some(v)) = (jp.joint.as_deref(), parse_f32(jp.value.as_deref())) {
            positions.set_dof(joint, 0, v);
        }
    }
}

/// On document reload, seed the commanded positions for the fresh robot, and forget the last-driven
/// joint so the loop solver treats the fresh doc's loop joints as ALL free (a doc whose start pose is
/// slightly open then self-assembles at load, held by nothing). [`ArticulatedJoints`] is rebuilt by
/// the scene's `rebuild_on_change`; this only resets the commands.
///
/// A document that authors a `<state default="true">` loads INTO that pose (the viewer used to always
/// zero-pose, silently wrong for such docs) so a reload RE-APPLIES the default state rather than
/// clearing to zero. Absent a default state, clear to the zero pose as before. Either way this seeds
/// the raw command map; mimic/limit resolution rides downstream in [`articulate`].
pub fn reset_on_reload(
    doc: Res<HcdfDoc>,
    mut positions: ResMut<JointPositions>,
    mut driven: ResMut<crate::loop_solver::DrivenJoint>,
) {
    if !doc.is_changed() {
        return;
    }
    match doc.0.as_deref().and_then(default_state) {
        Some(state) => apply_state(&mut positions, state),
        None if !positions.0.is_empty() => positions.0.clear(),
        None => {}
    }
    if driven.0.is_some() {
        driven.0 = None;
    }
}

/// Wires articulation into an [`App`]: the source-of-truth resources plus the
/// `reset_on_reload → solve_loops → articulate` chain. Pure-core (no egui); the slider panel lives in
/// the UI plugin. The loop-closure solver is registered HERE (the core plugin path) deliberately:
/// embedders (dendrite_build) compose the core plugins, so anything registered only in an app-level
/// plugin would be unreachable in the editor.
pub struct JointsPlugin;
impl Plugin for JointsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<JointPositions>()
            .init_resource::<ArticulatedJoints>()
            .init_resource::<crate::loop_solver::LoopSolveEnabled>()
            .init_resource::<crate::loop_solver::DrivenJoint>()
            .init_resource::<crate::loop_solver::LoopClosureStatus>()
            .add_systems(
                Update,
                (
                    reset_on_reload,
                    // Solve loop closures BETWEEN the reset and the pose apply, with the same change
                    // gating as articulate (plus the enable toggle so flipping the checkbox
                    // re-evaluates immediately): on a reload frame the solver sees the cleared
                    // commands and the fresh catalogue; on a slider frame it sees the new command
                    // and writes the solved passive coordinates back into JointPositions before
                    // articulate applies them: mechanism and display move in the same frame.
                    crate::loop_solver::solve_loops
                        .after(crate::scene::SceneSet::Rebuild)
                        .after(reset_on_reload)
                        .run_if(
                            resource_changed::<JointPositions>
                                .or_else(resource_changed::<ArticulatedJoints>)
                                .or_else(resource_changed::<crate::loop_solver::LoopSolveEnabled>),
                        ),
                    // Re-pose only when commands or the joint set changed. Runs after the scene rebuild
                    // so a freshly spawned catalogue + a reset pose both apply the same frame; runs in
                    // Update so PostUpdate propagation then refreshes GlobalTransforms. Also ordered
                    // after reset_on_reload so on a reload frame the cleared (zero) commands are applied
                    // to the freshly spawned entities (never the stale previous pose) and after
                    // solve_loops so the applied pose is the SOLVED one, never a one-frame-open flash.
                    articulate
                        .after(crate::scene::SceneSet::Rebuild)
                        .after(reset_on_reload)
                        .after(crate::loop_solver::solve_loops)
                        .run_if(
                            resource_changed::<JointPositions>
                                .or_else(resource_changed::<ArticulatedJoints>),
                        ),
                ),
            );
    }
}
