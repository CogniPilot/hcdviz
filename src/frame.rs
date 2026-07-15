//! Coordinate-frame reconciliation: the #1 correctness concern.
//!
//! HCDF declares two independent conventions on the root `<hcdf>` element:
//!   - `@world-frame`  ∈ {ENU, NED}: the world/inertial axes.
//!   - `@body-frame`   ∈ {FLU, FRD}: the per-body axes every comp/joint pose is authored in.
//!
//! Bevy renders in a fixed Y-up, right-handed basis (X right, Y up, Z toward viewer / "back").
//! ENU/NED and FLU/FRD are all *right-handed* frames, so each maps to Bevy by a proper rotation
//! (det +1); they differ only in which physical axis carries which label, never in handedness. We
//! still reconcile at two levels, because world-frame content and body-frame content are authored in
//! different conventions:
//!
//!   * [`WorldConvention::to_bevy_mat3`]: a [`WorldRoot`] basis that maps WORLD coordinates → Bevy.
//!     Carried by the single root entity; every world-frame thing (grid, world axes, top-level comp
//!     placement) lives under it.
//!   * [`BodyConvention::to_world_mat3`]: a per-root-comp basis that maps BODY coordinates → the world
//!     frame, so a body's authored +X/+Y/+Z land on the intended directions regardless of FLU vs FRD;
//!     the [`WorldRoot`] basis above then carries the result to Bevy.
//!
//! The maps are pure orthonormal [`Mat3`] rotations, so they unit-test cleanly headless. A point `p`
//! authored in a body frame, under the document's world frame, renders at
//! `world.to_bevy_mat3() * (body.to_world_mat3() * p)` once the comp sits at the world origin
//! (see [`FrameConvention::body_point_to_bevy`]).
use bevy::math::Mat3;
use bevy::prelude::*;

/// HCDF world/inertial frame convention (`@world-frame`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorldConvention {
    /// East-North-Up: X=East, Y=North, Z=Up. Right-handed, Z-up. The default.
    #[default]
    Enu,
    /// North-East-Down: X=North, Y=East, Z=Down. Z-down but still right-handed (N×E=Down), like ENU.
    Ned,
}

/// HCDF body frame convention (`@body-frame`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BodyConvention {
    /// Forward-Left-Up: X=Forward, Y=Left, Z=Up. Right-handed. The default.
    #[default]
    Flu,
    /// Forward-Right-Down: X=Forward, Y=Right, Z=Down. Flips Y and Z vs FLU.
    Frd,
}

impl WorldConvention {
    /// Map the typed `@world-frame` attribute; absent ⇒ ENU (the schema default).
    pub fn from_schema(v: Option<crate::schema::model::enums::WorldFrame>) -> Self {
        match v {
            Some(crate::schema::model::enums::WorldFrame::NED) => Self::Ned,
            Some(crate::schema::model::enums::WorldFrame::ENU) | None => Self::Enu,
        }
    }

    /// Column-major basis mapping a WORLD-frame vector to the Bevy rendering basis.
    ///
    /// The internal Bevy basis is chosen so the world ground plane lies on Bevy's X/−Z plane with up =
    /// Bevy +Y, the canonical Z-up→Y-up turn that Z-up engines use, and so a viewer looking down −Z
    /// sees "North/forward into the screen". Concretely:
    ///   ENU  X(East)→+X,   Y(North)→−Z, Z(Up)→+Y
    ///   NED  X(North)→−Z,  Y(East)→+X,  Z(Down)→−Y
    /// Both are *proper* rotations (det = +1): NED is itself right-handed, so mapping it to Bevy is a
    /// pure rotation (axis reassignment); no reflection or scale flip is needed at the world level.
    pub fn to_bevy_mat3(self) -> Mat3 {
        match self {
            // columns are images of world X, Y, Z respectively.
            Self::Enu => Mat3::from_cols(
                Vec3::X,     // East  -> +X
                Vec3::NEG_Z, // North -> -Z
                Vec3::Y,     // Up    -> +Y
            ),
            Self::Ned => Mat3::from_cols(
                Vec3::NEG_Z, // North -> -Z
                Vec3::X,     // East  -> +X
                Vec3::NEG_Y, // Down  -> -Y
            ),
        }
    }

    /// The world basis as a [`Transform`] for the [`WorldRoot`] entity.
    pub fn to_bevy_transform(self) -> Transform {
        mat3_to_transform(self.to_bevy_mat3())
    }
}

impl BodyConvention {
    /// Map the typed `@body-frame` attribute; absent ⇒ FLU (the schema default).
    pub fn from_schema(v: Option<crate::schema::model::enums::BodyFrame>) -> Self {
        match v {
            Some(crate::schema::model::enums::BodyFrame::FRD) => Self::Frd,
            Some(crate::schema::model::enums::BodyFrame::FLU) | None => Self::Flu,
        }
    }

    /// Column-major basis mapping a BODY-frame vector into the WORLD frame's axes (a function of the
    /// body convention alone, independent of ENU/NED).
    ///
    /// Body poses are authored in the body triple; we express them in the *world* triple so the single
    /// [`WorldRoot`] basis then carries everything to Bevy. The robotics convention is that a body at
    /// rest faces "North", so:
    ///   FLU  Forward(X)→North(world Y), Left(Y)→West(−world X), Up(Z)→Up(world Z).
    ///   FRD  Forward(X)→North(world X), Right(Y)→East(world Y), Down(Z)→Down(world Z)  (identity, as
    ///        the FRD body triple already aligns with the NED world triple).
    /// Composed with the matching world map this yields the rviz screen directions: Forward→Bevy −Z,
    /// Up→Bevy +Y for FLU/ENU; Forward→−Z, Down→−Y, Right→+X for FRD/NED (see the frame unit tests).
    pub fn to_world_mat3(self) -> Mat3 {
        match self {
            Self::Flu => Mat3::from_cols(Vec3::Y, Vec3::NEG_X, Vec3::Z),
            Self::Frd => Mat3::IDENTITY,
        }
    }

    /// The body→world basis as a [`Transform`] applied at each root comp.
    pub fn to_world_transform(self) -> Transform {
        mat3_to_transform(self.to_world_mat3())
    }
}

/// The full frame convention pair read from `<hcdf>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameConvention {
    pub world: WorldConvention,
    pub body: BodyConvention,
}

impl FrameConvention {
    pub fn from_hcdf(h: &crate::schema::Hcdf) -> Self {
        Self {
            world: WorldConvention::from_schema(h.world_frame),
            body: BodyConvention::from_schema(h.body_frame),
        }
    }

    /// Map a point authored in the BODY frame (at a comp sitting at the world origin) all the way to
    /// Bevy world coordinates: `world.to_bevy ∘ body.to_world`. Used by the frame unit tests.
    pub fn body_point_to_bevy(self, p: Vec3) -> Vec3 {
        self.world.to_bevy_mat3() * (self.body.to_world_mat3() * p)
    }

    /// Map a point authored directly in the WORLD frame to Bevy world coordinates.
    pub fn world_point_to_bevy(self, p: Vec3) -> Vec3 {
        self.world.to_bevy_mat3() * p
    }
}

/// Root entity carrying the world-frame → Bevy basis. All HCDF content is spawned beneath it so the
/// reconciliation happens exactly once.
#[derive(Component, Debug, Clone, Copy)]
pub struct WorldRoot {
    pub convention: FrameConvention,
}

/// Build a [`Transform`] from an orthonormal, right-handed column basis (a proper rotation, det +1).
///
/// All four HCDF frame conventions (ENU/NED × FLU/FRD) are proper rotations (they differ only by
/// axis assignment, never handedness) so this is always a pure quaternion. We deliberately do NOT
/// emit a negative scale to absorb a reflection: a left-handed `Transform` flips triangle winding and
/// normals in Bevy (the same class of bug as a mirrored mesh), so an improper basis is a programming
/// error here rather than something to paper over.
pub fn mat3_to_transform(m: Mat3) -> Transform {
    debug_assert!(
        (m.determinant() - 1.0).abs() < 1e-4,
        "frame basis must be a proper rotation (det +1); got det {}",
        m.determinant()
    );
    Transform::from_rotation(Quat::from_mat3(&m))
}

/// Convert an official [`schema::Pose`](crate::schema::Pose) to a Bevy [`Transform`].
///
/// Quaternion wins over rpy when present (schema rule). `rpy` is the HCDF/URDF/SDF convention:
/// **XYZ-extrinsic (fixed-axis) Euler**, i.e. the rotation `R = Rz(yaw)·Ry(pitch)·Rx(roll)` (roll/
/// pitch/yaw about the FIXED x/y/z axes), exactly matching `hcdformat`'s `pose_math::rpy_to_matrix`
/// and the Python `frames.rpy_to_matrix`. NOTE this is NOT glam's `EulerRot::XYZ` (which is INTRINSIC
/// XYZ = `Rx·Ry·Rz`); the two agree only when rpy is all-zero or single-axis, so the intrinsic form
/// silently mis-rotated any pose with multiple non-zero rpy components (e.g. SO-ARM's 90°/180° joints
/// and visuals). Built explicitly as `qz * qy * qx` to be convention-unambiguous. The quat is stored
/// `[x,y,z,w]` in HCDF; Bevy's [`Quat::from_xyzw`] takes the same order. Translation/rotation
/// are expressed in whatever frame the pose was authored in; the [`WorldRoot`]/body-basis wrappers
/// carry them to Bevy, so this function performs NO axis conversion itself.
pub fn pose_to_transform(p: &crate::schema::Pose) -> Transform {
    // An absent xyz/rpy means the identity component (origin / no rotation), so fold through the
    // `*_or_zero()` accessors (the typed `Pose` carries `Option<[f64;3]>` post-presence-aware-pose).
    let xyz = p.xyz_or_zero();
    let t = Vec3::new(xyz[0] as f32, xyz[1] as f32, xyz[2] as f32);
    let rot = match p.quat {
        Some(q) => {
            let quat = Quat::from_xyzw(q[0] as f32, q[1] as f32, q[2] as f32, q[3] as f32);
            // A zero/degenerate quat normalizes to NaN; guard to identity.
            if quat.length_squared() > 1e-9 {
                quat.normalize()
            } else {
                Quat::IDENTITY
            }
        }
        None => {
            // XYZ-extrinsic (fixed-axis): R = Rz(yaw)·Ry(pitch)·Rx(roll). Built explicitly so the
            // convention is unambiguous (NOT glam's intrinsic `EulerRot::XYZ`). qz*qy*qx composes the
            // same order as the matrix product, matching hcdformat::pose_math::rpy_to_matrix.
            let rpy = p.rpy_or_zero();
            let (roll, pitch, yaw) = (rpy[0] as f32, rpy[1] as f32, rpy[2] as f32);
            Quat::from_rotation_z(yaw) * Quat::from_rotation_y(pitch) * Quat::from_rotation_x(roll)
        }
    };
    Transform::from_translation(t).with_rotation(rot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Pose;
    use std::f64::consts::FRAC_PI_2;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    /// rpy must be XYZ-EXTRINSIC (fixed-axis): R = Rz(yaw)·Ry(pitch)·Rx(roll). For
    /// (roll=90°, pitch=0, yaw=90°) that sends +X→+Y and +Y→+Z. glam's INTRINSIC `EulerRot::XYZ`
    /// (the old bug) would send +X→+Z instead: the exact mis-rotation that exploded SO-ARM (whose
    /// joints/visuals are full of 90°/180° rpy), while OpenArm (all-zero rpy) hid it.
    #[test]
    fn rpy_is_xyz_extrinsic_not_glam_intrinsic() {
        let p = Pose {
            xyz: None,
            rpy: Some([FRAC_PI_2, 0.0, FRAC_PI_2]),
            quat: None,
        };
        let r = pose_to_transform(&p).rotation;
        assert!(approx(r * Vec3::X, Vec3::Y), "+X→+Y, got {:?}", r * Vec3::X);
        assert!(approx(r * Vec3::Y, Vec3::Z), "+Y→+Z, got {:?}", r * Vec3::Y);
        assert!(
            !approx(r * Vec3::X, Vec3::Z),
            "must NOT match glam intrinsic XYZ (+X→+Z)"
        );
    }

    /// quat wins over rpy and is taken verbatim (xyzw order).
    #[test]
    fn quat_wins_over_rpy() {
        let p = Pose {
            xyz: None,
            rpy: Some([FRAC_PI_2, 0.0, 0.0]),
            quat: Some([0.0, 0.0, 0.0, 1.0]),
        };
        assert!(
            approx(pose_to_transform(&p).rotation * Vec3::Y, Vec3::Y),
            "identity quat beats rpy"
        );
    }
}
