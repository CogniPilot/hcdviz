//! Loop-closure SOLVING: make the constraint joints the spanning tree demotes
//! ([`crate::kinematics::KinematicTree::constraints`]) actually bind, so a parallel mechanism moves
//! as a mechanism (drive a four-bar's crank and the passive joints follow) instead of visibly
//! pulling apart (constraint links were previously drawn, never enforced).
//!
//! FORMULATION: constraint projection over the commanded joint map, NOT end-effector IK. After a
//! slider change the driven joint is HELD and the other joint coordinates on the affected loop
//! paths are adjusted until every closure constraint is satisfied: "the mechanism assembles", never
//! "drag a target". Per closure joint c (parent comp P, child comp C, fixed origin `O_c`):
//!
//! ```text
//!   A = T_world(P)·O_c      B = T_world(C)     (tree convention: child frame ≡ joint frame,
//!   E = A⁻¹·B                                   the same convention closures use)
//!   e = [translation(E); rotvec(rotation(E))]  (a 6-error in the closure JOINT frame)
//! ```
//!
//! Each closure PROJECTS `e` (and therefore its Jacobian rows): an explicit `<constraint-axes>`
//! gives the documented componentwise 0/1 mask; otherwise the projector derives from the joint type
//! (`I − aaᵀ` for free-axis subspaces: PROJECTORS, not masks, because HCDF axes are arbitrary unit
//! vectors in the joint frame; screw adds a helical-coupling row; universal swaps its rotation rows
//! for the EXACT manifold invariant `a·(R·a2) − a·a2`: a linearized row along `a×a2` would flag an
//! exactly-assembled U-joint open by ~αβ/2). See [`Projector`].
//!
//! SOLVER: damped least squares `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e` with LM-lite λ adaptation, a per-step
//! norm clamp, and limit projection applied AFTER each step (clamping a VARIABLE inside the
//! evaluation would zero finite-difference gradients exactly at a bound and freeze the solve
//! there). The Jacobian is NUMERICAL forward differences over the pure FK path: the problems are
//! tiny (≲12 variables, a few closures), FD is consistent with the residual by construction, and it
//! inherits the mimic chain rule and every multi-DOF joint composition (universal's rotated second
//! axis, ball RPY, screw coupling, planar basis) with zero bespoke twist algebra: the classic
//! source of sign/frame bugs. Rotation errors are differenced on a CONSISTENT shortest-arc branch
//! ([`align_rotation_branch`]), so a closure sitting at the rotation-vector π boundary still gets a
//! finite, correctly-signed column. FK evaluates through the SAME mimic resolution
//! [`crate::joints::articulate`] uses ([`resolve_all_for_solver`] shares the resolver: variables
//! raw, mimic FOLLOWERS clamped to their limit boxes exactly as the display clamps them), so the
//! residual is the error of the pose the user actually SEES: a follower with a binding limit inside
//! a loop makes the loop honestly report open instead of "closing" through a follower pose the
//! renderer refuses to reach. The dense solve is a hand-rolled small Cholesky on the damped normal
//! matrix: no new dependency, wasm-clean.
//! Non-convergence keeps the best iterate and reports the per-closure residual (the mechanism
//! genuinely cannot close under its limits/geometry); never a NaN, never a panic.
//!
//! WRITE-BACK: solved passive coordinates land in [`JointPositions`], so the single source of
//! truth stays single and the sliders SHOW the solved values (the panel always tells the
//! mechanism's truth). A per-DOF epsilon gate keeps resource change detection terminating: the
//! solver's own write re-triggers exactly one converged (write-free) run, then the chain is quiet:
//! a static mechanism costs nothing per frame. The core here is PURE (build a [`LoopProblem`] from
//! `&Hcdf` + [`KinematicTree`], no ECS, no entities); [`solve_loops`] is the thin system face,
//! registered inside [`crate::joints::JointsPlugin`], the CORE plugin path, so embedders composing
//! core plugins (dendrite_build) reach it without any extra registration.
use crate::doc::HcdfDoc;
use crate::frame::pose_to_transform;
use crate::joints::{
    clamp_dof, joint_local_transform, make_articulated_joint, resolve_all_for_solver,
    ArticulatedJoint, JointKind, JointPositions, MAX_JOINT_DOF,
};
use crate::kinematics::{build_kinematic_tree, KinematicTree};
use crate::schema::Hcdf;
use bevy::prelude::*;
use std::collections::{HashMap, HashSet};

/// Iteration stop: the solve is converged when the stacked projected residual norm drops below this
/// (metres/radians mixed: the standard DLS convergence measure for a pose error).
pub const SOLVE_TOL: f64 = 1e-6;

/// Display tolerance for "this closure is OPEN": comfortably above [`SOLVE_TOL`] plus the f32 FK
/// noise floor (world poses compose in `f32`, so a perfectly solved metre-scale loop still measures
/// ~1e-6), yet far below any separation a human could see (0.1 mm / ~0.006°). Drives the
/// warning-red constraint gizmo and the panel's "open by" text via [`ClosureError::open`].
pub const OPEN_TOL: f32 = 1e-4;

/// Write-back epsilon: a solved coordinate is written into [`JointPositions`] only when it moves by
/// more than this, so the solver's own writes cannot re-trigger it forever (guaranteeing termination).
pub const WRITEBACK_EPS: f32 = 1e-7;

/// Forward-difference step for the numerical Jacobian (radians / metres).
const FD_H: f32 = 1e-4;

/// Iteration budget per solve. Problems this small converge in a handful of Gauss-Newton steps;
/// hitting the cap means the mechanism cannot close (limits/geometry) and we keep the best iterate.
const MAX_ITERS: usize = 64;

/// Trust-region bound on one DLS step's norm (radians/metres mixed): DLS can propose huge steps
/// near singular configurations; the clamp keeps every iterate inside the linearization's
/// neighborhood.
const STEP_CLAMP: f64 = 0.5;

/// LM-lite damping: start at λ≈0.1, shrink on an accepted step, grow on a rejected one (bounded
/// both ways so the adaptation can always recover in a few iterations).
const LAMBDA_INIT: f64 = 0.1;
const LAMBDA_MIN: f64 = 1e-3;
const LAMBDA_MAX: f64 = 1e3;
const LAMBDA_SHRINK: f64 = 0.5;
const LAMBDA_GROW: f64 = 4.0;

/// A step is accepted only when it improves the residual norm by a REAL margin: 0.1% relative, or
/// [`ACCEPT_MIN_DECREASE`] absolute. Plain `norm_try < norm` would accept the sub-noise creep the
/// f32 FK floor produces near convergence: creep large enough to trip the write-back epsilon gate
/// frame after frame, churning forever. The relative rule alone is not enough either: when one
/// UNFIXABLE row dominates the norm (a blocked/over-constrained closure holding it at, say, 0.5),
/// 0.1% of the total is a huge hurdle for the still-solvable rows and the best-effort iterate would
/// stall visibly short of its constrained optimum, so the absolute rule keeps those rows grinding
/// down to the [`ACCEPT_MIN_DECREASE`] floor.
const ACCEPT_FACTOR: f64 = 0.999;

/// Absolute residual-norm decrease that always justifies accepting a step: comfortably above the
/// per-row f32 FK quantization (~1e-7 per component at metre scale) so noise alone still cannot
/// sustain acceptance, yet small enough that a blocked closure's solvable rows converge to within
/// ~1e-4 of their constrained optimum before steps start being rejected.
const ACCEPT_MIN_DECREASE: f64 = 1e-7;

/// Early exit after this many CONSECUTIVE rejected steps: each rejection grows λ by
/// [`LAMBDA_GROW`], so six in a row means the damping grew ~4000× without finding any improving
/// direction: the local model is exhausted and further iterations only burn time at the residual
/// floor (either converged-to-noise or genuinely blocked; both keep the best iterate).
const MAX_REJECTS: usize = 6;

/// Cholesky pivot floor: the damped normal matrix `JJᵀ + λ²I` is positive-definite by construction
/// (λ ≥ [`LAMBDA_MIN`]), so a pivot at/below this means a non-finite residual leaked in; the solve
/// bails rather than emitting NaN steps.
const MIN_PIVOT: f64 = 1e-12;

// ───────────────────────────── ECS resources (the solver's thin state) ─────────────────────────────

/// Master toggle for the solver: the "Solve loop closures" checkbox in
/// [`crate::ui::joints_panel`]. ON by default; OFF restores exactly the historical behavior:
/// constraint links draw, nothing is enforced, nothing is written.
#[derive(Resource)]
pub struct LoopSolveEnabled(pub bool);

impl Default for LoopSolveEnabled {
    fn default() -> Self {
        Self(true)
    }
}

/// The joint the user is CURRENTLY driving: the most recently slider-touched joint, held fixed by
/// the solver while every other loop-member joint adjusts around it.
///
/// Written by [`crate::ui::joints_panel`] only when a slider actually changes a value (the solver's
/// own write-backs go straight into [`JointPositions`] and can never set this). `None` after a
/// document load/reload (cleared by [`crate::joints::reset_on_reload`]) and after the panel's Reset
/// button: with no driver, EVERY loop joint is free, so a doc that loads slightly open
/// self-assembles at its zero pose.
#[derive(Resource, Default)]
pub struct DrivenJoint(pub Option<String>);

/// Per-closure solve outcome, rebuilt on every solver run (and on every reload). One entry per
/// closure in the CURRENT doc, kept even while the solver is disabled (with `error: None`) so the
/// panel can keep offering the enable checkbox; empty when the doc has no closures at all.
#[derive(Resource, Default)]
pub struct LoopClosureStatus(pub Vec<ClosureStatus>);

/// One closure's identity + latest measured state for the UI and the constraint-link gizmo.
#[derive(Debug, Clone)]
pub struct ClosureStatus {
    /// Index of the closure joint into `hcdf.joint`: the stable key
    /// [`crate::scene::ConstraintLinkMarker`] carries so each gizmo finds ITS closure.
    pub doc_joint: usize,
    /// Closure joint name (an unnamed joint falls back to `#<index>`) for the panel's open-by line.
    pub name: String,
    /// The measured closure error at the last solve, `None` while the solver is disabled (not
    /// evaluated; gizmos and panel text fall back to exactly the pre-solver look).
    pub error: Option<ClosureError>,
}

/// How far a closure is from closed, measured at the solver's final (best) iterate.
#[derive(Debug, Clone, Copy)]
pub struct ClosureError {
    /// Projected translation error norm (metres): the "X mm" half of the panel text.
    pub trans: f32,
    /// Projected rotation error norm (radians): the "Y °" half of the panel text. A universal
    /// closure's manifold-invariant row counts here (it is radian-scale for crossed axes).
    pub rot: f32,
    /// Full projected residual norm (translation + rotation + coupling rows; mixed units: the
    /// solver's own convergence measure for this closure).
    pub residual: f32,
}

impl ClosureError {
    /// Whether this closure is visibly open (beyond [`OPEN_TOL`]): the warning-red gizmo /
    /// open-by-text criterion.
    pub fn open(&self) -> bool {
        self.residual > OPEN_TOL
    }
}

// ──────────────────────────────── the pure loop problem + solver ────────────────────────────────

/// The per-closure projection of the 6-error `e = [t; r]`, kept as two 3×3 blocks (translation and
/// rotation) plus an optional EXTRA row (screw's helical coupling / universal's manifold
/// invariant). The block shape is deliberate: the status split into "mm open" vs "degrees open"
/// falls straight out of the same matrices the residual uses. Rank-deficient blocks (a projector's
/// whole point) are harmless downstream: the λ²-damped normal matrix stays positive-definite.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Projector {
    /// Applied to the translation error (rows are constrained directions).
    t: Mat3,
    /// Applied to the rotation-vector error.
    r: Mat3,
    /// Optional extra row appended after the six projected components.
    extra: Option<ExtraRow>,
}

/// The optional extra residual row a closure appends after its six projected `[t; rv]` components.
/// Kept beside the 3×3 blocks in [`Projector`] because it is evaluated (and finite-differenced)
/// exactly like them: one raw-error function feeds values and Jacobian alike.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ExtraRow {
    /// Screw's helical coupling, a LINEAR row over the raw 6-error: `a·t − (pitch/2π)(a·r) = 0`.
    Coupling([f32; 6]),
    /// Universal's EXACT manifold-membership test `a·(R_E·a2) − a·a2 = 0`. Nonlinear in the
    /// rotation error `R_E`, which is why the row keeps the axes and evaluates on the error
    /// quaternion instead of projecting the rotation vector; see [`Projector::from_kind`] for the
    /// derivation and why the linearization is not good enough.
    Universal { a: Vec3, a2: Vec3 },
}

impl ExtraRow {
    /// Evaluate the row on one closure's raw error. (The quaternion's double-cover sign is
    /// irrelevant: the universal invariant acts through the rotation, `q` and `−q` agree.)
    fn value(self, e: &RawError) -> f32 {
        match self {
            Self::Coupling(row) => coupling_dot(row, e.t, e.rv),
            Self::Universal { a, a2 } => a.dot(e.rot * a2) - a.dot(a2),
        }
    }
}

impl Projector {
    /// The documented `<constraint-axes>` semantics: a componentwise 0/1 mask over
    /// `tx ty tz rx ry rz` (1 = constrained), applied as diagonal projectors.
    fn from_mask(m: [bool; 6]) -> Self {
        let d = |a: bool, b: bool, c: bool| {
            Mat3::from_diagonal(Vec3::new(f32::from(a), f32::from(b), f32::from(c)))
        };
        Self {
            t: d(m[0], m[1], m[2]),
            r: d(m[3], m[4], m[5]),
            extra: None,
        }
    }

    /// Type-derived projector for a closure without (usable) `<constraint-axes>`: constrain every
    /// direction the joint type does NOT articulate. `I − aaᵀ` projects onto the subspace
    /// orthogonal to a free axis `a`; `aaᵀ` onto the axis itself. `axis`/`axis2` are already
    /// normalized (parse-time invariant of [`make_articulated_joint`]).
    fn from_kind(kind: JointKind, axis: Vec3, axis2: Vec3) -> Self {
        let free_about = |a: Vec3| Mat3::IDENTITY - outer(a, a);
        match kind {
            // Rotation free about the axis; translation welded.
            JointKind::Revolute | JointKind::Continuous => Self {
                t: Mat3::IDENTITY,
                r: free_about(axis),
                extra: None,
            },
            // Translation free along the axis; rotation welded.
            JointKind::Prismatic => Self {
                t: free_about(axis),
                r: Mat3::IDENTITY,
                extra: None,
            },
            // Independent coaxial rotation + slide: both free about/along the one axis.
            JointKind::Cylindrical => Self {
                t: free_about(axis),
                r: free_about(axis),
                extra: None,
            },
            // Like cylindrical, but the axial advance is TIED to the axial rotation by the thread:
            // the coupling row enforces `a·t = (pitch/2π)(a·r)`.
            JointKind::Screw { pitch } => {
                let k = pitch / std::f32::consts::TAU;
                Self {
                    t: free_about(axis),
                    r: free_about(axis),
                    extra: Some(ExtraRow::Coupling([
                        axis.x,
                        axis.y,
                        axis.z,
                        -k * axis.x,
                        -k * axis.y,
                        -k * axis.z,
                    ])),
                }
            }
            // The rotation must lie ON the U-joint manifold {Rot(a,α)·Rot(a2,β)}: a curved
            // 2-surface in SO(3), so the one scalar constrained direction CANNOT be captured by a
            // linear projector: the row `(a×a2)ᵀ·rv` is exact only at first order, and an
            // EXACTLY-assembled universal measures a phantom ~αβ/2 against it (50× [`OPEN_TOL`] at
            // α = β = 0.1: a warning-red gizmo on a perfect mechanism, and free variables dragged
            // off to cancel an error that does not exist). The EXACT scalar invariant is
            // `a·(R·a2) = a·a2`: Rot(a,α) fixes a and Rot(a2,β) fixes a2, so every manifold point
            // preserves the crossed-axes inner product; conversely any R preserving it maps a2
            // onto its circle about a (R·a2 = Rot(a,α)·a2 for some α) whence Rot(a,−α)·R fixes
            // a2 and R factors as Rot(a,α)·Rot(a2,β). Near assembly the row scales like ‖a×a2‖ ×
            // (rotation error along a×a2): radian-scale for crossed axes, agreeing with the other
            // rotation rows' units. Degenerate (parallel) axes leave no circle to preserve:
            // constrain nothing rotational rather than fabricate a direction (the lenient-viewer
            // discipline; such a universal is malformed anyway).
            JointKind::Universal => Self {
                t: Mat3::IDENTITY,
                r: Mat3::ZERO,
                extra: (axis.cross(axis2).length_squared() > 1e-12)
                    .then_some(ExtraRow::Universal { a: axis, a2: axis2 }),
            },
            // In-plane translation free; only the component along the plane NORMAL (= `axis`) is
            // constrained. Rotation welded.
            JointKind::Planar => Self {
                t: outer(axis, axis),
                r: Mat3::IDENTITY,
                extra: None,
            },
            // Spherical closure: position welded, orientation entirely free.
            JointKind::Ball => Self {
                t: Mat3::IDENTITY,
                r: Mat3::ZERO,
                extra: None,
            },
            // Weld closure.
            JointKind::Fixed => Self {
                t: Mat3::IDENTITY,
                r: Mat3::IDENTITY,
                extra: None,
            },
            // `free`/unknown: a closure that constrains nothing (trivially closed).
            JointKind::Other => Self {
                t: Mat3::ZERO,
                r: Mat3::ZERO,
                extra: None,
            },
        }
    }
}

/// `a·bᵀ` (outer product) as a [`Mat3`]: column `j` is `a · b[j]`, so `outer(a,b) · v = a (b·v)`.
fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Parse `<constraint-axes>` (EXACTLY six whitespace-separated `0`/`1` tokens) into a mask.
/// Anything else returns `None` and the closure falls back to its type-derived projector (lenient
/// viewer; the hcdformat validator flags malformed axes upstream as `E_LOOP_AXES_MALFORMED`).
fn parse_constraint_axes(s: &str) -> Option<[bool; 6]> {
    let mut out = [false; 6];
    let mut n = 0;
    for tok in s.split_whitespace() {
        if n >= 6 {
            return None;
        }
        out[n] = match tok {
            "0" => false,
            "1" => true,
            _ => return None,
        };
        n += 1;
    }
    (n == 6).then_some(out)
}

/// One closure constraint extracted from a demoted joint.
struct ClosureSpec {
    /// Index into `hcdf.joint` (the status/gizmo key).
    doc_joint: usize,
    /// Display name (`#<index>` fallback for unnamed joints).
    name: String,
    /// Parent comp index (the side carrying the joint origin).
    parent: usize,
    /// Child comp index (child frame ≡ joint frame).
    child: usize,
    /// The closure joint's fixed origin `O_c` in the parent comp frame.
    origin: Transform,
    /// The error projection for this closure (mask or type-derived).
    proj: Projector,
}

/// One closure's raw (UNPROJECTED) pose error at an evaluation point: the shared input every
/// residual stack, Jacobian column, and status report is computed from, so all three views can
/// never measure different mechanisms.
#[derive(Clone, Copy)]
struct RawError {
    /// Translation of `E = A⁻¹·B` in the closure joint frame (metres).
    t: Vec3,
    /// `E`'s rotation as the exact group element: what nonlinear rows like
    /// [`ExtraRow::Universal`] evaluate on.
    rot: Quat,
    /// Shortest-arc rotation vector of `rot` (‖rv‖ ≤ π): the linear rows' coordinates.
    rv: Vec3,
}

/// One free scalar variable: a DOF of a movable, non-mimic tree joint on some loop path.
struct FreeVar {
    /// Index into [`LoopProblem::joints`].
    joint: usize,
    /// DOF index within that joint (see [`joint_local_transform`] for per-kind meaning).
    dof: usize,
}

/// A document's loop-closure problem, PURE (built from the parsed doc + spanning tree; no ECS, no
/// entities), so the whole solver unit-tests headless and runs identically on wasm.
pub struct LoopProblem {
    /// Pure mirror of the scene's `ArticulatedJoints` catalogue: one entry per spanning-tree edge
    /// (in tree order), built by the SAME [`make_articulated_joint`] the scene rebuild uses so
    /// kind/axis/limit/mimic semantics are byte-identical. The child `Entity` is a placeholder:
    /// FK here walks comp indices, never the ECS.
    joints: Vec<ArticulatedJoint>,
    /// child comp index → (parent comp index, index into `joints`): the upward FK chain. Roots are
    /// absent (their common world/body basis cancels exactly in `A⁻¹B`, so FK starts at identity).
    parent_edge: HashMap<usize, (usize, usize)>,
    /// The closure constraints, in document constraint order.
    closures: Vec<ClosureSpec>,
    /// Free-variable CANDIDATES: every DOF of every movable, non-mimic, NAMED tree joint lying on
    /// any closure's P↔C path (union over closures: closures sharing joints solve as ONE stacked
    /// system). The driven joint is excluded per solve, not here (it changes with every touch).
    candidates: Vec<FreeVar>,
}

/// A solved free-variable coordinate ready for write-back.
#[derive(Debug, Clone)]
pub struct SolvedDof {
    /// Joint name (the [`JointPositions`] key).
    pub joint: String,
    /// DOF index within the joint.
    pub dof: usize,
    /// The coordinate at the solver's best iterate (already projected into the joint's limits).
    pub value: f32,
}

/// The outcome of one [`LoopProblem::solve`].
#[derive(Debug, Clone)]
pub struct LoopSolution {
    /// Every free variable's coordinate at the best iterate (unchanged ones included: the caller's
    /// epsilon gate decides what constitutes a real write).
    pub values: Vec<SolvedDof>,
    /// Per-closure error at the best iterate, parallel to the problem's closures (document
    /// constraint order).
    pub errors: Vec<ClosureError>,
    /// Whether the stacked residual dropped below [`SOLVE_TOL`]. Note the f32 FK noise floor: a
    /// metre-scale mechanism can be visually perfect yet measure ~1e-6, so per-closure OPENNESS
    /// (the display question) uses [`ClosureError::open`] against [`OPEN_TOL`] instead.
    pub converged: bool,
    /// Iterations spent (0 when already converged at the seed).
    pub iterations: usize,
}

impl LoopSolution {
    /// The solved coordinate for `joint` DOF `dof`, `None` when that DOF was not a free variable
    /// (the driven joint, a mimic follower, or a joint off every loop path).
    pub fn value(&self, joint: &str, dof: usize) -> Option<f32> {
        self.values
            .iter()
            .find(|v| v.joint == joint && v.dof == dof)
            .map(|v| v.value)
    }
}

impl LoopProblem {
    /// Extract the loop problem from a parsed doc + its spanning tree. `None` when the doc has no
    /// demoted constraint joints (the overwhelmingly common case: pure tree robots cost nothing).
    pub fn build(h: &Hcdf, tree: &KinematicTree) -> Option<Self> {
        if tree.constraints.is_empty() {
            return None;
        }
        // The pure joint catalogue: same construction as the scene rebuild's edge loop.
        let mut joints = Vec::with_capacity(tree.edges.len());
        let mut parent_edge = HashMap::new();
        for e in &tree.edges {
            let joint = &h.joint[e.joint];
            let origin = joint
                .origin
                .as_ref()
                .map(pose_to_transform)
                .unwrap_or(Transform::IDENTITY);
            parent_edge.insert(e.child, (e.parent, joints.len()));
            joints.push(make_articulated_joint(joint, Entity::PLACEHOLDER, origin));
        }
        // Every demoted joint is a closure. Parsing it through make_articulated_joint keeps its
        // kind/axis/origin semantics identical to a tree joint's: no second parser to drift.
        let closures: Vec<ClosureSpec> = tree
            .constraints
            .iter()
            .map(|c| {
                let j = &h.joint[c.joint];
                let origin = j
                    .origin
                    .as_ref()
                    .map(pose_to_transform)
                    .unwrap_or(Transform::IDENTITY);
                let cj = make_articulated_joint(j, Entity::PLACEHOLDER, origin);
                // Explicit <constraint-axes> wins when well-formed; malformed axes fall back to the
                // type-derived projector (lenient viewer; the validator flags them upstream).
                let proj = j
                    .loop_
                    .as_ref()
                    .and_then(|l| l.constraint_axes.as_deref())
                    .and_then(parse_constraint_axes)
                    .map(Projector::from_mask)
                    .unwrap_or_else(|| Projector::from_kind(cj.kind, cj.axis, cj.axis2));
                ClosureSpec {
                    doc_joint: c.joint,
                    name: if cj.name.is_empty() {
                        format!("#{}", c.joint)
                    } else {
                        cj.name.clone()
                    },
                    parent: c.parent,
                    child: c.child,
                    origin: cj.origin,
                    proj,
                }
            })
            .collect();
        // Free-variable candidates: the union of loop-path joints over all closures. Mimic
        // followers are never variables (their coupled motion enters through the FD evaluation);
        // unnamed joints can't be addressed in JointPositions, so they stay passive too.
        let mut on_path: HashSet<usize> = HashSet::new();
        for c in &closures {
            on_path.extend(closure_path_edges(&parent_edge, c.parent, c.child));
        }
        let candidates = joints
            .iter()
            .enumerate()
            .filter(|(i, j)| {
                on_path.contains(i)
                    && j.kind.is_movable()
                    && j.mimic.is_none()
                    && !j.name.is_empty()
            })
            .flat_map(|(i, j)| (0..j.kind.dof_count()).map(move |d| FreeVar { joint: i, dof: d }))
            .collect();
        Some(Self {
            joints,
            parent_edge,
            closures,
            candidates,
        })
    }

    /// Solve the closure constraints, holding `driven` fixed (usually the last slider-touched
    /// joint; `None` (the fresh-load state) frees every loop joint so an open zero pose
    /// self-assembles). Returns the best iterate's free-variable coordinates + per-closure errors;
    /// it never panics and never emits NaN for finite inputs.
    pub fn solve(&self, positions: &JointPositions, driven: Option<&str>) -> LoopSolution {
        // Seed = the commanded map projected into every movable joint's limit box (i.e. exactly
        // the coordinates `articulate` displays) so the (unclamped) evaluation starts from, and
        // the projected steps stay within, the feasible box.
        let mut w = JointPositions(positions.0.clone());
        for j in &self.joints {
            if j.kind.is_movable() && j.mimic.is_none() && !j.name.is_empty() {
                let q = w.dofs(&j.name);
                for (d, &raw) in q.iter().enumerate().take(j.kind.dof_count()) {
                    let c = clamp_dof(j.kind, d, &j.lower, &j.upper, raw);
                    if c != raw {
                        w.set_dof(&j.name, d, c);
                    }
                }
            }
        }
        let vars: Vec<&FreeVar> = self
            .candidates
            .iter()
            .filter(|v| driven != Some(self.joints[v.joint].name.as_str()))
            .collect();
        let mut raw = self.raw_errors(&w);
        let mut r = self.project(&raw);
        let mut norm = norm2(&r);
        let mut lambda = LAMBDA_INIT;
        let mut iterations = 0;
        let mut rejects = 0;
        while iterations < MAX_ITERS
            && rejects < MAX_REJECTS
            && norm >= SOLVE_TOL
            && !vars.is_empty()
        {
            iterations += 1;
            let jac = self.jacobian(&mut w, &vars, &raw, &r);
            let Some(mut dq) = dls_step(&jac, &r, vars.len(), lambda) else {
                // The damped normal matrix is PD by construction, so a failed factorization means a
                // non-finite residual leaked in, so grow the damping and retry rather than panic.
                lambda = (lambda * LAMBDA_GROW).min(LAMBDA_MAX);
                rejects += 1;
                continue;
            };
            clamp_step_norm(&mut dq);
            let mut w_try = JointPositions(w.0.clone());
            for (k, v) in vars.iter().enumerate() {
                let j = &self.joints[v.joint];
                let q = w.dof(&j.name, v.dof) - dq[k] as f32;
                // Project AFTER the step (never inside the FD evaluation, which would zero the
                // gradient exactly at a bound). A blocked mechanism therefore ends AT its limit
                // with the residual honestly reporting how far open it is.
                w_try.set_dof(
                    &j.name,
                    v.dof,
                    clamp_dof(j.kind, v.dof, &j.lower, &j.upper, q),
                );
            }
            let raw_try = self.raw_errors(&w_try);
            let r_try = self.project(&raw_try);
            let norm_try = norm2(&r_try);
            // LM-lite: accept only a REAL improvement, relative OR absolute (see the constants'
            // rationale); shrink damping on success, grow it on failure and retry from the same
            // (still-best) iterate.
            if norm_try < norm * ACCEPT_FACTOR || norm - norm_try > ACCEPT_MIN_DECREASE {
                w = w_try;
                raw = raw_try;
                r = r_try;
                norm = norm_try;
                lambda = (lambda * LAMBDA_SHRINK).max(LAMBDA_MIN);
                rejects = 0;
            } else {
                lambda = (lambda * LAMBDA_GROW).min(LAMBDA_MAX);
                rejects += 1;
            }
        }
        // Only improving steps were ever accepted, so `w` IS the best iterate.
        let values = vars
            .iter()
            .map(|v| {
                let j = &self.joints[v.joint];
                SolvedDof {
                    joint: j.name.clone(),
                    dof: v.dof,
                    value: w.dof(&j.name, v.dof),
                }
            })
            .collect();
        let errors = self.closure_errors(&raw);
        LoopSolution {
            values,
            errors,
            converged: norm < SOLVE_TOL,
            iterations,
        }
    }

    /// Every closure's raw pose error at `w` (document constraint order): the projection-free
    /// evaluation the residual stack, the Jacobian, and the status report all share. Evaluation
    /// resolves mimics through the display's own resolver: followers clamped, variables raw (see
    /// module docs and [`resolve_all_for_solver`]).
    fn raw_errors(&self, w: &JointPositions) -> Vec<RawError> {
        let resolved = resolve_all_for_solver(&self.joints, w);
        let mut memo = HashMap::new();
        self.closures
            .iter()
            .map(|c| {
                let a = self
                    .comp_world(c.parent, &resolved, &mut memo)
                    .mul_transform(c.origin);
                let b = self.comp_world(c.child, &resolved, &mut memo);
                let (t, rot) = pose_error(a, b);
                RawError {
                    t,
                    rot,
                    rv: rotation_vector(rot),
                }
            })
            .collect()
    }

    /// The stacked projected residual over one evaluation's raw errors: per closure, 6 rows
    /// `[P_t·t; P_r·rv]` plus the optional extra row.
    fn project(&self, raw: &[RawError]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.closures.len() * 7);
        for (c, e) in self.closures.iter().zip(raw) {
            let pt = c.proj.t * e.t;
            let pr = c.proj.r * e.rv;
            out.extend([pt.x, pt.y, pt.z, pr.x, pr.y, pr.z].map(f64::from));
            if let Some(row) = c.proj.extra {
                out.push(f64::from(row.value(e)));
            }
        }
        out
    }

    /// Per-closure error report (the status/UX view of the same raw errors [`Self::project`]
    /// stacks for the solver).
    fn closure_errors(&self, raw: &[RawError]) -> Vec<ClosureError> {
        self.closures
            .iter()
            .zip(raw)
            .map(|(c, e)| {
                let pt = c.proj.t * e.t;
                let pr = c.proj.r * e.rv;
                let trans_sq = pt.length_squared();
                let mut rot_sq = pr.length_squared();
                let mut mixed_sq = 0.0;
                if let Some(row) = c.proj.extra {
                    let v = row.value(e);
                    match row {
                        // The universal invariant IS the closure's rotation error (radian-scale
                        // for crossed axes): it belongs in the "degrees open" half of the report.
                        ExtraRow::Universal { .. } => rot_sq += v * v,
                        // The screw coupling mixes metres and radians: counted in the overall
                        // residual only, never mislabeled as pure translation or rotation.
                        ExtraRow::Coupling(_) => mixed_sq = v * v,
                    }
                }
                ClosureError {
                    trans: trans_sq.sqrt(),
                    rot: rot_sq.sqrt(),
                    residual: (trans_sq + rot_sq + mixed_sq).sqrt(),
                }
            })
            .collect()
    }

    /// FK: a comp's world pose under the resolved coordinates, memoized per evaluation. Roots are
    /// identity: the shared WorldRoot/body-basis prefix cancels exactly in every `A⁻¹B`, so world
    /// conventions never enter the residual. Recursion terminates because the spanning tree is
    /// acyclic by construction (kinematics.rs demotes cycle-creating joints).
    fn comp_world(
        &self,
        comp: usize,
        resolved: &HashMap<String, [f32; MAX_JOINT_DOF]>,
        memo: &mut HashMap<usize, Transform>,
    ) -> Transform {
        if let Some(&t) = memo.get(&comp) {
            return t;
        }
        let world = match self.parent_edge.get(&comp) {
            None => Transform::IDENTITY,
            Some(&(parent, jidx)) => {
                let pw = self.comp_world(parent, resolved, memo);
                let j = &self.joints[jidx];
                let q = resolved
                    .get(&j.name)
                    .copied()
                    .unwrap_or([0.0; MAX_JOINT_DOF]);
                pw.mul_transform(joint_local_transform(j.origin, j.axis, j.axis2, j.kind, &q))
            }
        };
        memo.insert(comp, world);
        world
    }

    /// Forward-difference Jacobian (row-major, rows = residual entries, cols = `vars`): perturb one
    /// variable by [`FD_H`], re-evaluate, diff against `r0`. The ACTUAL perturbation `(q+h)−q` is
    /// measured in f64 (f32 rounding shrinks it for large coordinates) and a fully-absorbed bump
    /// leaves a zero column instead of dividing by zero.
    ///
    /// Every bumped evaluation's rotation vectors are re-expressed on the SAME π-hemisphere branch
    /// as the base evaluation `raw0` before projecting ([`align_rotation_branch`]): a closure whose
    /// rotation-error angle sits within [`FD_H`] of π would otherwise flip shortest-arc
    /// representation between the two evaluations, turning an O(h) physical motion into a ~2π/h
    /// garbage column that points every DLS step uphill: a fully closable mechanism would then
    /// burn [`MAX_REJECTS`] and be falsely reported stuck open at residual ≈ π.
    fn jacobian(
        &self,
        w: &mut JointPositions,
        vars: &[&FreeVar],
        raw0: &[RawError],
        r0: &[f64],
    ) -> Vec<f64> {
        let n = vars.len();
        let mut jac = vec![0.0; r0.len() * n];
        for (k, v) in vars.iter().enumerate() {
            let j = &self.joints[v.joint];
            let saved = w.dof(&j.name, v.dof);
            let bumped = saved + FD_H;
            let dh = f64::from(bumped) - f64::from(saved);
            if dh <= 0.0 {
                continue;
            }
            w.set_dof(&j.name, v.dof, bumped);
            let raw1: Vec<RawError> = self
                .raw_errors(w)
                .into_iter()
                .zip(raw0)
                .map(|(e1, e0)| RawError {
                    rv: align_rotation_branch(e1.rv, e0.rv),
                    ..e1
                })
                .collect();
            let r1 = self.project(&raw1);
            w.set_dof(&j.name, v.dof, saved);
            for (row, (a, b)) in r1.iter().zip(r0).enumerate() {
                jac[row * n + k] = (a - b) / dh;
            }
        }
        jac
    }
}

/// The tree-path edge set between two comps (through their common ancestor): the SYMMETRIC
/// DIFFERENCE of the two root paths. Edges shared by both root paths (above the common ancestor)
/// move A and B rigidly together and cancel exactly in `E = A⁻¹B`, so they must NOT become
/// variables: their FD columns would be numerically zero noise.
fn closure_path_edges(
    parent_edge: &HashMap<usize, (usize, usize)>,
    a: usize,
    b: usize,
) -> Vec<usize> {
    let pa = root_path(parent_edge, a);
    let pb = root_path(parent_edge, b);
    let sa: HashSet<usize> = pa.iter().copied().collect();
    let sb: HashSet<usize> = pb.iter().copied().collect();
    pa.iter()
        .filter(|j| !sb.contains(j))
        .chain(pb.iter().filter(|j| !sa.contains(j)))
        .copied()
        .collect()
}

/// Edge indices (into [`LoopProblem::joints`]) from `comp` up to its root, child→root order. The
/// walk is bounded purely defensively: the spanning tree is acyclic by construction.
fn root_path(parent_edge: &HashMap<usize, (usize, usize)>, mut comp: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for _ in 0..=parent_edge.len() {
        match parent_edge.get(&comp) {
            Some(&(parent, jidx)) => {
                out.push(jidx);
                comp = parent;
            }
            None => break,
        }
    }
    out
}

/// The pose error between two frames as `(translation, rotation)` of `E = A⁻¹·B`, expressed in A's
/// (the closure joint's) frame. The rotation stays a quaternion here; [`RawError`] carries both it
/// and its [`rotation_vector`], each feeding the row kinds that need that form.
fn pose_error(a: Transform, b: Transform) -> (Vec3, Quat) {
    let e = a.compute_affine().inverse() * b.compute_affine();
    let (_, rot, t) = e.to_scale_rotation_translation();
    (t, rot)
}

/// The rotation-vector (axis·angle) form of a quaternion on the SHORTEST arc (angle ≤ π; the
/// double cover is collapsed by flipping to the positive-w hemisphere first). Small angles use the
/// first-order `2·(x,y,z)` form, exact to O(θ³), avoiding the 0/0 axis normalization at identity.
fn rotation_vector(q: Quat) -> Vec3 {
    let q = if q.w < 0.0 { -q } else { q };
    let v = Vec3::new(q.x, q.y, q.z);
    let s = v.length();
    if s < 1e-6 {
        return v * 2.0;
    }
    v * (2.0 * s.atan2(q.w) / s)
}

/// Re-express a shortest-arc rotation vector on the same π-hemisphere BRANCH as `reference`: of
/// `rv` and its antipodal representation `rv − 2π·r̂` (the IDENTICAL rotation, written as
/// angle − 2π about the same axis), keep whichever lies closer to the reference.
///
/// Shortest-arc vectors are unique only below ‖rv‖ = π: a rotation error whose angle crosses π
/// between two nearby evaluations flips representation to (2π − θ) about the NEGATED axis: a ~2π
/// coordinate jump for an O(h) physical motion. Finite differencing must therefore difference both
/// evaluations on ONE branch (the caller aligns the bumped evaluation to the base one). Away from
/// the boundary the antipode (‖·‖ ≥ π by construction) is never the closer choice, so this is a
/// no-op everywhere differencing is already well posed. The residual VALUE deliberately stays on
/// the shortest arc: ‖rv‖ is the geodesic distance on SO(3), the honest "how far open" measure.
fn align_rotation_branch(rv: Vec3, reference: Vec3) -> Vec3 {
    let n = rv.length();
    if n < 1e-6 {
        return rv;
    }
    let alt = rv * (1.0 - std::f32::consts::TAU / n);
    if (alt - reference).length_squared() < (rv - reference).length_squared() {
        alt
    } else {
        rv
    }
}

/// `row · [t; rv]`: the screw coupling row applied to a raw 6-error.
fn coupling_dot(row: [f32; 6], t: Vec3, rv: Vec3) -> f32 {
    row[0] * t.x + row[1] * t.y + row[2] * t.z + row[3] * rv.x + row[4] * rv.y + row[5] * rv.z
}

/// `‖v‖₂` of a stacked residual / step.
fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Scale a step down to the trust-region bound [`STEP_CLAMP`] when it exceeds it.
fn clamp_step_norm(dq: &mut [f64]) {
    let n = norm2(dq);
    if n > STEP_CLAMP {
        let s = STEP_CLAMP / n;
        for d in dq {
            *d *= s;
        }
    }
}

/// One damped-least-squares step `dq = Jᵀ(JJᵀ + λ²I)⁻¹ r` for the row-major `m×n` Jacobian `jac`
/// (`m = r.len()`). `None` only when the Cholesky factorization collapses (non-finite input).
fn dls_step(jac: &[f64], r: &[f64], n: usize, lambda: f64) -> Option<Vec<f64>> {
    let m = r.len();
    // A = JJᵀ + λ²I (symmetric m×m; m stays tiny: 6..7 rows per closure).
    let mut a = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut s = 0.0;
            for k in 0..n {
                s += jac[i * n + k] * jac[j * n + k];
            }
            a[i * m + j] = s;
            a[j * m + i] = s;
        }
        a[i * m + i] += lambda * lambda;
    }
    let y = cholesky_solve(&a, r, m)?;
    // dq = Jᵀ y.
    let mut dq = vec![0.0; n];
    for (k, d) in dq.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..m {
            s += jac[i * n + k] * y[i];
        }
        *d = s;
    }
    Some(dq)
}

/// Solve `A·x = b` for a symmetric positive-definite row-major `n×n` matrix via an LLᵀ Cholesky
/// factorization + two triangular substitutions. Returns `None` when a pivot collapses (A not PD;
/// for the λ²-damped normal matrix that only happens if the residual went non-finite). Hand-rolled
/// because the systems are tiny (n ≲ 20) and a linear-algebra crate would be a new dependency on
/// the wasm bundle for nothing.
fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= MIN_PIVOT || !s.is_finite() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    // Forward substitution L·y = b, then back substitution Lᵀ·x = y (in place).
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

// ─────────────────────────────────────── the ECS face ───────────────────────────────────────

/// The solver system, ordered `reset_on_reload → solve_loops → articulate` inside
/// [`crate::joints::JointsPlugin`] and change-gated exactly like `articulate` (plus the enable
/// toggle, so flipping the checkbox re-evaluates immediately). Its own write-back re-triggers
/// exactly one more (converged, write-free) run, then the whole chain is quiet until the next real
/// change.
///
/// The tree + problem are rebuilt per run ON PURPOSE: it is trivially cheap next to the FD solve
/// (hash walks over the joint list; a no-loop doc bails right after the tree walk), needs no
/// cache-invalidation coupling to the scene rebuild, and the run condition already limits runs to
/// frames where something actually changed.
pub fn solve_loops(
    doc: Res<HcdfDoc>,
    enabled: Res<LoopSolveEnabled>,
    driven: Res<DrivenJoint>,
    mut positions: ResMut<JointPositions>,
    mut status: ResMut<LoopClosureStatus>,
) {
    let problem = doc
        .0
        .as_ref()
        .and_then(|h| LoopProblem::build(h, &build_kinematic_tree(h)));
    let Some(problem) = problem else {
        // No doc / no closures: empty status hides every loop-UX affordance.
        if !status.0.is_empty() {
            status.0.clear();
        }
        return;
    };
    if !enabled.0 {
        // Solver OFF = exactly the historical behavior (no writes, orange gizmos, no open-by
        // text), but keep one UNEVALUATED entry per closure so the panel still offers the
        // checkbox that turns solving back on.
        status.0 = problem
            .closures
            .iter()
            .map(|c| ClosureStatus {
                doc_joint: c.doc_joint,
                name: c.name.clone(),
                error: None,
            })
            .collect();
        return;
    }
    let solution = problem.solve(&positions, driven.0.as_deref());
    for v in &solution.values {
        // The per-DOF epsilon gate: only real movement is written, so change detection settles
        // (reads go through Deref and never dirty the resource).
        if (positions.dof(&v.joint, v.dof) - v.value).abs() > WRITEBACK_EPS {
            positions.set_dof(&v.joint, v.dof, v.value);
        }
    }
    status.0 = problem
        .closures
        .iter()
        .zip(&solution.errors)
        .map(|(c, e)| ClosureStatus {
            doc_joint: c.doc_joint,
            name: c.name.clone(),
            error: Some(*e),
        })
        .collect();
}

// ─────────────────────────────────────────── tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn mat3_eq(a: Mat3, b: Mat3) -> bool {
        a.abs_diff_eq(b, EPS)
    }

    // ── Cholesky ──────────────────────────────────────────────────────────────────────────────

    #[test]
    fn cholesky_solves_a_known_spd_system() {
        // A = [[4,2],[2,3]], b = [10,13] → x = A⁻¹b = (0.5, 4.0) (det 8; verified by hand).
        let a = [4.0, 2.0, 2.0, 3.0];
        let b = [10.0, 13.0];
        let x = cholesky_solve(&a, &b, 2).expect("SPD system must factor");
        assert!((x[0] - 0.5).abs() < 1e-12, "x0 = {}", x[0]);
        assert!((x[1] - 4.0).abs() < 1e-12, "x1 = {}", x[1]);
    }

    #[test]
    fn cholesky_solves_a_3x3_and_roundtrips() {
        // SPD by construction (diagonally dominant). Verify by multiplying back.
        let a = [6.0, 1.0, 2.0, 1.0, 5.0, 0.5, 2.0, 0.5, 4.0];
        let b = [1.0, -2.0, 3.0];
        let x = cholesky_solve(&a, &b, 3).expect("SPD");
        for i in 0..3 {
            let ax: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!((ax - b[i]).abs() < 1e-12, "row {i}: {ax} vs {}", b[i]);
        }
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        // Indefinite (eigenvalues 3, −1): the second pivot goes non-positive.
        assert!(cholesky_solve(&[1.0, 2.0, 2.0, 1.0], &[1.0, 1.0], 2).is_none());
        // All-zero matrix: first pivot is zero.
        assert!(cholesky_solve(&[0.0; 4], &[1.0, 1.0], 2).is_none());
        // Non-finite input must bail, never NaN-propagate.
        assert!(cholesky_solve(&[f64::NAN, 0.0, 0.0, 1.0], &[1.0, 1.0], 2).is_none());
    }

    // ── constraint-axes parsing ───────────────────────────────────────────────────────────────

    #[test]
    fn constraint_axes_parse_accepts_exactly_six_binary_tokens() {
        assert_eq!(
            parse_constraint_axes("1 1 0 0 0 1"),
            Some([true, true, false, false, false, true])
        );
        // Whitespace-resilient (any run of whitespace separates tokens).
        assert_eq!(
            parse_constraint_axes("  1\t1 0  0 0 1 "),
            Some([true, true, false, false, false, true])
        );
    }

    #[test]
    fn constraint_axes_parse_rejects_malformed() {
        assert_eq!(parse_constraint_axes(""), None);
        assert_eq!(parse_constraint_axes("1 1 0 0 0"), None); // five tokens
        assert_eq!(parse_constraint_axes("1 1 0 0 0 1 1"), None); // seven tokens
        assert_eq!(parse_constraint_axes("1 1 0 0 0 2"), None); // non-binary token
        assert_eq!(parse_constraint_axes("true 1 0 0 0 1"), None); // junk token
    }

    // ── projector math ────────────────────────────────────────────────────────────────────────

    #[test]
    fn mask_projector_is_componentwise_diagonal() {
        let p = Projector::from_mask([true, false, true, false, true, false]);
        assert!(mat3_eq(p.t, Mat3::from_diagonal(Vec3::new(1.0, 0.0, 1.0))));
        assert!(mat3_eq(p.r, Mat3::from_diagonal(Vec3::new(0.0, 1.0, 0.0))));
        assert!(p.extra.is_none());
    }

    #[test]
    fn revolute_projector_frees_rotation_about_axis_only() {
        // An arbitrary (non-cardinal) unit axis: the reason these are projectors, not masks.
        let a = Vec3::new(1.0, 1.0, 0.0).normalize();
        let p = Projector::from_kind(JointKind::Revolute, a, Vec3::Y);
        assert!(mat3_eq(p.t, Mat3::IDENTITY), "translation fully welded");
        // Rotation error ALONG the axis is annihilated; orthogonal error passes through.
        assert!((p.r * a).length() < EPS);
        let perp = Vec3::new(1.0, -1.0, 0.0).normalize();
        assert!((p.r * perp - perp).length() < EPS);
    }

    #[test]
    fn prismatic_and_cylindrical_projectors_free_the_axis_subspaces() {
        let a = Vec3::Z;
        let pri = Projector::from_kind(JointKind::Prismatic, a, Vec3::Y);
        assert!((pri.t * a).length() < EPS, "translation free along axis");
        assert!((pri.t * Vec3::X - Vec3::X).length() < EPS);
        assert!(mat3_eq(pri.r, Mat3::IDENTITY), "rotation welded");
        let cyl = Projector::from_kind(JointKind::Cylindrical, a, Vec3::Y);
        assert!((cyl.t * a).length() < EPS);
        assert!((cyl.r * a).length() < EPS);
    }

    #[test]
    fn planar_projector_constrains_normal_translation_only() {
        let n = Vec3::Z; // plane normal
        let p = Projector::from_kind(JointKind::Planar, n, Vec3::Y);
        assert!((p.t * n - n).length() < EPS, "normal component constrained");
        assert!((p.t * Vec3::X).length() < EPS, "in-plane translation free");
        assert!(mat3_eq(p.r, Mat3::IDENTITY), "rotation welded");
    }

    #[test]
    fn ball_fixed_other_projectors() {
        let ball = Projector::from_kind(JointKind::Ball, Vec3::X, Vec3::Y);
        assert!(mat3_eq(ball.t, Mat3::IDENTITY));
        assert!(mat3_eq(ball.r, Mat3::ZERO), "spherical: rotation all free");
        let fixed = Projector::from_kind(JointKind::Fixed, Vec3::X, Vec3::Y);
        assert!(mat3_eq(fixed.t, Mat3::IDENTITY));
        assert!(mat3_eq(fixed.r, Mat3::IDENTITY), "weld: everything welded");
        let other = Projector::from_kind(JointKind::Other, Vec3::X, Vec3::Y);
        assert!(mat3_eq(other.t, Mat3::ZERO));
        assert!(
            mat3_eq(other.r, Mat3::ZERO),
            "free joint constrains nothing"
        );
    }

    /// A pure-rotation [`RawError`] for exercising the nonlinear rows.
    fn rot_error(rot: Quat) -> RawError {
        RawError {
            t: Vec3::ZERO,
            rot,
            rv: rotation_vector(rot),
        }
    }

    #[test]
    fn universal_extra_row_is_the_exact_manifold_invariant() {
        let (a, a2) = (Vec3::X, Vec3::Y);
        let p = Projector::from_kind(JointKind::Universal, a, a2);
        assert!(mat3_eq(p.t, Mat3::IDENTITY), "translation fully welded");
        assert!(mat3_eq(p.r, Mat3::ZERO), "no linearized rotation rows");
        let Some(row @ ExtraRow::Universal { .. }) = p.extra else {
            panic!("universal must carry the manifold-invariant row");
        };
        // EXACTLY-assembled U-joint poses (Rot(a,α)·Rot(a2,β) for ANY two angles) measure zero.
        // The first-order a×a2 projector this row replaced measured a phantom ~αβ/2 here (0.0050
        // at α = β = 0.1: fifty times OPEN_TOL on a perfect mechanism).
        for (alpha, beta) in [(0.1, 0.1), (0.6, 0.6), (-0.4, 0.9), (1.2, -0.3)] {
            let e = rot_error(Quat::from_axis_angle(a, alpha) * Quat::from_axis_angle(a2, beta));
            assert!(
                row.value(&e).abs() < EPS,
                "phantom residual {} at ({alpha}, {beta})",
                row.value(&e)
            );
        }
        // A rotation OFF the manifold (about a×a2 = Z) measures its exact geometry:
        // X·(Rot(Z,ε)·Y) − X·Y = −sin ε.
        let e = rot_error(Quat::from_rotation_z(0.2));
        assert!(
            (row.value(&e) + 0.2f32.sin()).abs() < EPS,
            "off-manifold value {}",
            row.value(&e)
        );
        // Degenerate (parallel axes): constrain nothing rotational rather than invent a direction.
        let d = Projector::from_kind(JointKind::Universal, Vec3::X, Vec3::X);
        assert!(d.extra.is_none());
        assert!(mat3_eq(d.r, Mat3::ZERO));
    }

    #[test]
    fn screw_projector_couples_advance_to_rotation() {
        let pitch = 0.5;
        let a = Vec3::Z;
        let p = Projector::from_kind(JointKind::Screw { pitch }, a, Vec3::Y);
        let Some(ExtraRow::Coupling(row)) = p.extra else {
            panic!("screw carries the helical coupling row");
        };
        let k = pitch / std::f32::consts::TAU;
        // A consistent helical move (t = k·θ·a paired with rv = θ·a) must zero the coupling row.
        let theta = 0.7;
        let v = coupling_dot(row, a * (k * theta), a * theta);
        assert!(v.abs() < EPS, "consistent screw motion residual = {v}");
        // An advance WITHOUT rotation trips it by exactly the axial advance.
        let v = coupling_dot(row, a * 0.2, Vec3::ZERO);
        assert!((v - 0.2).abs() < EPS);
    }

    // ── rotation vector ───────────────────────────────────────────────────────────────────────

    #[test]
    fn rotation_vector_identity_and_known_axis_angle() {
        assert!(rotation_vector(Quat::IDENTITY).length() < 1e-9);
        let rv = rotation_vector(Quat::from_axis_angle(Vec3::Z, 0.3));
        assert!((rv - Vec3::new(0.0, 0.0, 0.3)).length() < EPS, "{rv:?}");
    }

    #[test]
    fn rotation_vector_takes_the_shortest_arc() {
        // 3π/2 about +Z ≡ π/2 about −Z: the double cover collapses to the ≤π representative.
        let rv = rotation_vector(Quat::from_axis_angle(Vec3::Z, 1.5 * std::f32::consts::PI));
        assert!(
            (rv - Vec3::new(0.0, 0.0, -std::f32::consts::FRAC_PI_2)).length() < EPS,
            "{rv:?}"
        );
        assert!(rv.length() <= std::f32::consts::PI + EPS);
    }

    #[test]
    fn align_rotation_branch_unfolds_the_pi_crossing() {
        use std::f32::consts::PI;
        let z = Vec3::Z;
        // Base evaluation: θ = π + 1e-4 measured on the shortest arc → (π − 1e-4) about −Z.
        let reference = -z * (PI - 1e-4);
        // Bumped evaluation: θ moved back through π → hemisphere flipped to +Z.
        let bumped = z * (PI - 2e-4);
        let aligned = align_rotation_branch(bumped, reference);
        // Same branch as the reference: the continuous coordinate just past −π.
        assert!(
            (aligned - (-z * (PI + 2e-4))).length() < 1e-5,
            "{aligned:?}"
        );
        // Far from the boundary the antipode is never closer: a strict no-op.
        let rv = Vec3::new(0.1, -0.2, 0.05);
        assert_eq!(align_rotation_branch(rv, Vec3::ZERO), rv);
        assert_eq!(align_rotation_branch(rv, rv + Vec3::splat(1e-4)), rv);
        // The identity rotation has no meaningful axis to fold around; left untouched.
        assert_eq!(align_rotation_branch(Vec3::ZERO, reference), Vec3::ZERO);
    }

    #[test]
    fn pose_error_is_expressed_in_the_a_frame() {
        // A at (1,0,0) rotated +90° about Z; B at (1,1,0) with the same rotation: the offset is +Y
        // in world = +X in A's rotated frame, with zero rotation error.
        let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let a = Transform::from_translation(Vec3::X).with_rotation(rot);
        let b = Transform::from_translation(Vec3::new(1.0, 1.0, 0.0)).with_rotation(rot);
        let (t, e_rot) = pose_error(a, b);
        assert!((t - Vec3::X).length() < EPS, "{t:?}");
        assert!(rotation_vector(e_rot).length() < EPS, "{e_rot:?}");
    }
}
