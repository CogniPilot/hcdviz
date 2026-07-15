//! Headless tests for LOOP-CLOSURE SOLVING (no GPU, no window).
//!
//! Two layers, mirroring the feature's own split:
//!  * PURE: [`LoopProblem`] built straight from parsed HCDF + the spanning tree, driven through
//!    slider-like incremental sweeps (each step solves from the previous step's solution, exactly
//!    like a slider drag): parallelogram closed form, Grashof closure, constraint-axes
//!    masking, limit-blocked best-effort, mimic-in-loop, multi-loop stacking.
//!  * ECS: the `reset_on_reload → solve_loops → articulate` chain in a real (minimal) app: solved
//!    write-back reaching both [`JointPositions`] and the entity transforms, the 30-frame no-churn
//!    discipline at converged rest (the connector_stability.rs discipline), reload reset,
//!    zero-driver self-assembly of a slightly-open doc, and solver-OFF passthrough.
//!
//! The four-bar fixtures use a `rocker_tip` helper comp (fixed-jointed at the rocker's free end):
//! HCDF's closure convention is "child frame ≡ joint frame", so the closure's CHILD frame must SIT
//! at the physical attachment point: a revolute rocker's own frame is at its pivot, hence the tip
//! comp carries the attachment frame. Parallelogram closed form (crank length = rocker length,
//! ground spacing = coupler length): θ_rocker = θ_crank and θ_coupler = −θ_crank for all crank
//! angles: the coupler stays parallel to ground.
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::doc::HcdfDoc;
use hcdviz::joints::{JointPositions, JointsPlugin};
use hcdviz::kinematics::build_kinematic_tree;
use hcdviz::loop_solver::{
    DrivenJoint, LoopClosureStatus, LoopProblem, LoopSolution, LoopSolveEnabled,
};
use hcdviz::pick::Selected;
use hcdviz::scene::{CompEntity, ScenePlugin};
use hcdviz::schema::Hcdf;
use std::collections::HashMap;
use std::sync::Arc;

/// Closed-form tolerance for solved joint angles.
const ANGLE_EPS: f32 = 1e-4;
/// Residual tolerance a closed loop must beat over a sweep.
const RESIDUAL_EPS: f32 = 1e-5;
/// The no-churn observation window (the connector_stability.rs discipline).
const N_FRAMES: usize = 30;

// ────────────────────────────────────────── fixtures ──────────────────────────────────────────

// PARALLELOGRAM four-bar, authored CLOSED at the zero pose: crank length = rocker length = 1,
// ground spacing = coupler length = 2. Tree = ground→crank→coupler, ground→rocker→rocker_tip
// (fixed); the closure joint ties the coupler's far end (2,0,0 in coupler frame) to rocker_tip,
// with the documented explicit mask "1 1 1 1 1 0" (weld all but rotation about the joint Z).
const PARALLELOGRAM: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="fourbar" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/>
  <comp name="coupler"/>
  <comp name="rocker"/>
  <comp name="rocker_tip"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler" type="revolute"><parent comp="crank"/><child comp="coupler"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rocker" type="revolute"><parent comp="ground"/><child comp="rocker"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip" type="fixed"><parent comp="rocker"/><child comp="rocker_tip"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="coupler"/><child comp="rocker_tip"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler</predecessor><successor>rocker_tip</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// The same parallelogram with the ROCKER limited to ±0.2 rad: driving the crank past 0.2 makes the
// closure geometrically unsatisfiable.
const PARALLELOGRAM_BLOCKED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="fourbarblocked" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/>
  <comp name="coupler"/>
  <comp name="rocker"/>
  <comp name="rocker_tip"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler" type="revolute"><parent comp="crank"/><child comp="coupler"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rocker" type="revolute"><parent comp="ground"/><child comp="rocker"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-0.2" upper="0.2"/></joint>
  <joint name="j_tip" type="fixed"><parent comp="rocker"/><child comp="rocker_tip"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="coupler"/><child comp="rocker_tip"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler</predecessor><successor>rocker_tip</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// GRASHOF crank-rocker (no closed form asserted): crank 1, rocker 2 (pivot at x=2), coupler √5,
// authored closed at zero (coupler runs from crank tip (0,1) to rocker tip (2,2), so the closure
// origin in the coupler frame is (2,1,0)). No <constraint-axes>: the revolute closure exercises the
// TYPE-DERIVED projector path. Grashof: 1 + √5 < 2 + 2 → the crank can drive continuously.
const GRASHOF: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="crankrocker" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/>
  <comp name="coupler"/>
  <comp name="rocker"/>
  <comp name="rocker_tip"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler" type="revolute"><parent comp="crank"/><child comp="coupler"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rocker" type="revolute"><parent comp="ground"/><child comp="rocker"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip" type="fixed"><parent comp="rocker"/><child comp="rocker_tip"/><origin xyz="0 2 0"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="coupler"/><child comp="rocker_tip"/><origin xyz="2 1 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler</predecessor><successor>rocker_tip</successor></loop>
  </joint>
</hcdf>"#;

// constraint-axes MASK honored: a peg welded to base at (cos 0.3, sin 0.3, 0.5) and
// an arm whose point (1,0,0) must meet it, but only tx/ty are constrained ("1 1 0 0 0 0"), so the
// permanent 0.5 m tz offset and all rotation never count. The solver need only spin the arm to 0.3.
const MASKED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="masked" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="arm"/>
  <comp name="peg"/>
  <joint name="j_arm" type="revolute"><parent comp="base"/><child comp="arm"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_peg" type="fixed"><parent comp="base"/><child comp="peg"/><origin xyz="0.955336 0.295520 0.5"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="arm"/><child comp="peg"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>arm</predecessor><successor>peg</successor><constraint-axes>1 1 0 0 0 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// The CONTROL for the mask: identical geometry but tz IS constrained ("1 1 1 0 0 0"): the 0.5 m
// offset is unfixable by the arm's Z rotation, so the best effort zeroes tx/ty and honestly reports
// 0.5 m open.
const MASKED_FULL: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="maskedfull" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="arm"/>
  <comp name="peg"/>
  <joint name="j_arm" type="revolute"><parent comp="base"/><child comp="arm"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_peg" type="fixed"><parent comp="base"/><child comp="peg"/><origin xyz="0.955336 0.295520 0.5"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="arm"/><child comp="peg"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>arm</predecessor><successor>peg</successor><constraint-axes>1 1 1 0 0 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// Mimic-in-loop, source = the DRIVEN joint: the coupler mimics the crank ×−1,
// exactly the parallelogram closed form, leaving the rocker as the only free variable.
const PARA_MIMIC_DRIVEN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="fourbarmimicdriven" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/>
  <comp name="coupler"/>
  <comp name="rocker"/>
  <comp name="rocker_tip"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler" type="revolute"><parent comp="crank"/><child comp="coupler"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><mimic joint="j_crank" multiplier="-1" offset="0"/></joint>
  <joint name="j_rocker" type="revolute"><parent comp="ground"/><child comp="rocker"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip" type="fixed"><parent comp="rocker"/><child comp="rocker_tip"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="coupler"/><child comp="rocker_tip"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler</predecessor><successor>rocker_tip</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// Mimic-in-loop, source = a FREE VARIABLE (the FD chain-rule case): the coupler
// mimics the ROCKER ×−1 (consistent with the parallelogram: θ_coupler = −θ_crank = −θ_rocker), so
// perturbing the lone variable moves TWO loop joints through the shared resolver.
const PARA_MIMIC_VAR: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="fourbarmimicvar" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/>
  <comp name="coupler"/>
  <comp name="rocker"/>
  <comp name="rocker_tip"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler" type="revolute"><parent comp="crank"/><child comp="coupler"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><mimic joint="j_rocker" multiplier="-1" offset="0"/></joint>
  <joint name="j_rocker" type="revolute"><parent comp="ground"/><child comp="rocker"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip" type="fixed"><parent comp="rocker"/><child comp="rocker_tip"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="coupler"/><child comp="rocker_tip"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler</predecessor><successor>rocker_tip</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// Mimic-in-loop with a BINDING FOLLOWER LIMIT: the coupler mimics the crank ×−1 but is itself
// limited to ±0.1 rad. Past crank = 0.1 the DISPLAY clamps the follower at its bound while the
// mimic commands more: the displayed mechanism cannot close, and the solver must measure THAT
// mechanism (followers resolve clamped in its evaluation), not a phantom one where the follower
// tracks its source through the forbidden range.
const PARA_MIMIC_LIMITED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="fourbarmimiclimited" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/>
  <comp name="coupler"/>
  <comp name="rocker"/>
  <comp name="rocker_tip"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler" type="revolute"><parent comp="crank"/><child comp="coupler"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-0.1" upper="0.1"/><mimic joint="j_crank" multiplier="-1" offset="0"/></joint>
  <joint name="j_rocker" type="revolute"><parent comp="ground"/><child comp="rocker"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip" type="fixed"><parent comp="rocker"/><child comp="rocker_tip"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="coupler"/><child comp="rocker_tip"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler</predecessor><successor>rocker_tip</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// TWO INDEPENDENT loops: two disjoint parallelograms on one ground, the second
// shifted +10 in X. Their variable sets never interact; the stacked system is block-diagonal.
const TWO_LOOPS: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="twoloops" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank_a"/><comp name="coupler_a"/><comp name="rocker_a"/><comp name="tip_a"/>
  <comp name="crank_b"/><comp name="coupler_b"/><comp name="rocker_b"/><comp name="tip_b"/>
  <joint name="j_crank_a" type="revolute"><parent comp="ground"/><child comp="crank_a"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler_a" type="revolute"><parent comp="crank_a"/><child comp="coupler_a"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rocker_a" type="revolute"><parent comp="ground"/><child comp="rocker_a"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip_a" type="fixed"><parent comp="rocker_a"/><child comp="tip_a"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop_a" type="revolute"><parent comp="coupler_a"/><child comp="tip_a"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler_a</predecessor><successor>tip_a</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
  <joint name="j_crank_b" type="revolute"><parent comp="ground"/><child comp="crank_b"/><origin xyz="10 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler_b" type="revolute"><parent comp="crank_b"/><child comp="coupler_b"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rocker_b" type="revolute"><parent comp="ground"/><child comp="rocker_b"/><origin xyz="12 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip_b" type="fixed"><parent comp="rocker_b"/><child comp="tip_b"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop_b" type="revolute"><parent comp="coupler_b"/><child comp="tip_b"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler_b</predecessor><successor>tip_b</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// Two loops SHARING a joint: a double parallelogram, the middle rocker belongs
// to BOTH loop paths (loop 1: crank→coupler1→rocker_m; loop 2: rocker_m→coupler2→rocker_f), so the
// two closures must solve as ONE stacked system. Closed form: every link parallels the crank:
// q_rm = q_rf = θ, q_c1 = q_c2 = −θ.
const FIVEBAR_SHARED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="doublepara" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/><comp name="coupler1"/><comp name="rocker_m"/><comp name="tip_m"/>
  <comp name="coupler2"/><comp name="rocker_f"/><comp name="tip_f"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_c1" type="revolute"><parent comp="crank"/><child comp="coupler1"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rm" type="revolute"><parent comp="ground"/><child comp="rocker_m"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tipm" type="fixed"><parent comp="rocker_m"/><child comp="tip_m"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop1" type="revolute"><parent comp="coupler1"/><child comp="tip_m"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler1</predecessor><successor>tip_m</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
  <joint name="j_c2" type="revolute"><parent comp="rocker_m"/><child comp="coupler2"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rf" type="revolute"><parent comp="ground"/><child comp="rocker_f"/><origin xyz="4 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tipf" type="fixed"><parent comp="rocker_f"/><child comp="tip_f"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop2" type="revolute"><parent comp="coupler2"/><child comp="tip_f"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler2</predecessor><successor>tip_f</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// A parallelogram whose ZERO pose is slightly OPEN (test-only fixture): the closure
// origin sits 5 cm off the rocker tip at zero, so the doc must self-assemble at load with NO driver
// (DrivenJoint = None ⇒ every loop joint free).
const PARALLELOGRAM_OPEN: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="fourbaropen" body-frame="FLU" world-frame="ENU">
  <comp name="ground"/>
  <comp name="crank"/>
  <comp name="coupler"/>
  <comp name="rocker"/>
  <comp name="rocker_tip"/>
  <joint name="j_crank" type="revolute"><parent comp="ground"/><child comp="crank"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_coupler" type="revolute"><parent comp="crank"/><child comp="coupler"/><origin xyz="0 1 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_rocker" type="revolute"><parent comp="ground"/><child comp="rocker"/><origin xyz="2 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_tip" type="fixed"><parent comp="rocker"/><child comp="rocker_tip"/><origin xyz="0 1 0"/></joint>
  <joint name="j_loop" type="revolute"><parent comp="coupler"/><child comp="rocker_tip"/><origin xyz="2 0.05 0"/><axis xyz="0 0 1"/>
    <loop><predecessor>coupler</predecessor><successor>rocker_tip</successor><constraint-axes>1 1 1 1 1 0</constraint-axes></loop>
  </joint>
</hcdf>"#;

// A weld (fixed-type) closure whose rotation error at the ZERO SEED sits at/just past the
// rotation-vector π boundary: base→arm revolute Z (limits ±4 rad, so BOTH 2π-representatives of
// the target angle are reachable), base→peg statically yawed by θ, closure arm↔peg with the
// type-derived full weld. Closing simply means driving the arm to θ (mod 2π).
fn weld_near_pi(rot: f64) -> String {
    format!(
        r#"<?xml version="1.0"?>
<hcdf version="1.0" name="weldpi" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="arm"/>
  <comp name="peg"/>
  <joint name="j_arm" type="revolute"><parent comp="base"/><child comp="arm"/><axis xyz="0 0 1"/><limit lower="-4" upper="4"/></joint>
  <joint name="j_peg" type="fixed"><parent comp="base"/><child comp="peg"/><origin rpy="0 0 {rot}"/></joint>
  <joint name="j_loop" type="fixed"><parent comp="arm"/><child comp="peg"/>
    <loop><predecessor>arm</predecessor><successor>peg</successor></loop>
  </joint>
</hcdf>"#
    )
}

// A UNIVERSAL closure that is EXACTLY assembled for every driver angle: base→gimbal revolute X
// (the driver), gimbal→cap revolute Y mimicking the gimbal ×1, and a type-derived universal closure
// base↔cap with axis X / axis2 Y. The cap's world rotation is Rot(X,q)·Rot(Y,q), a point ON the
// U-joint manifold {Rot(a,α)·Rot(a2,β)}, so a CORRECT universal constraint measures zero at every
// q. j_cap is a mimic (never a variable) and j_gimbal is the driver, so solve() has no free
// variable: it purely MEASURES the closure, isolating the projector.
const UNIVERSAL_ASSEMBLED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="ujoint" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="gimbal"/>
  <comp name="cap"/>
  <joint name="j_gimbal" type="revolute"><parent comp="base"/><child comp="gimbal"/><axis xyz="1 0 0"/><limit lower="-3" upper="3"/></joint>
  <joint name="j_cap" type="revolute"><parent comp="gimbal"/><child comp="cap"/><axis xyz="0 1 0"/><mimic joint="j_gimbal" multiplier="1" offset="0"/></joint>
  <joint name="j_loop" type="universal"><parent comp="base"/><child comp="cap"/><axis xyz="1 0 0"/><axis2 xyz="0 1 0"/>
    <loop><predecessor>base</predecessor><successor>cap</successor></loop>
  </joint>
</hcdf>"#;

// A plain loop-free chain (a reload target).
const NOLOOP: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="chain" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="link1"/>
  <joint name="shoulder" type="revolute"><parent comp="base"/><child comp="link1"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/><limit lower="-1.5708" upper="1.5708"/></joint>
</hcdf>"#;

// ─────────────────────────────────────── pure-core helpers ───────────────────────────────────────

fn parse(xml: &str) -> Hcdf {
    Hcdf::from_xml_str(xml).unwrap()
}

fn problem(h: &Hcdf) -> LoopProblem {
    LoopProblem::build(h, &build_kinematic_tree(h)).expect("fixture declares loop closures")
}

/// Drive `driven`'s DOF 0 through `angles` sequentially, exactly like a slider drag: each
/// step solves from the PREVIOUS step's written-back solution (continuation keeps the solver on the
/// mechanism's assembly branch), calling `check` on every step's solution.
fn sweep(
    p: &LoopProblem,
    pos: &mut JointPositions,
    driven: &str,
    angles: impl Iterator<Item = f32>,
    check: &mut impl FnMut(f32, &LoopSolution),
) {
    for theta in angles {
        pos.set_dof(driven, 0, theta);
        let s = p.solve(pos, Some(driven));
        for v in &s.values {
            pos.set_dof(&v.joint, v.dof, v.value);
        }
        check(theta, &s);
    }
}

/// ±60° crank sweep in 5° steps, out and back through zero (25 solves).
fn crank_sweep() -> impl Iterator<Item = f32> {
    let up = (0..=12).map(|i| (i as f32) * 5f32.to_radians());
    let down = (-12..=11).rev().map(|i| (i as f32) * 5f32.to_radians());
    up.chain(down)
}

// ─────────────────────────────────────── pure-core tests ───────────────────────────────────────

#[test]
fn parallelogram_closed_form_over_crank_sweep() {
    let h = parse(PARALLELOGRAM);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    let mut steps = 0;
    sweep(&p, &mut pos, "j_crank", crank_sweep(), &mut |theta, s| {
        steps += 1;
        assert!(
            s.errors[0].residual < RESIDUAL_EPS,
            "closure residual {} at crank {theta}",
            s.errors[0].residual
        );
        let rocker = s.value("j_rocker", 0).expect("rocker is a free variable");
        let coupler = s.value("j_coupler", 0).expect("coupler is a free variable");
        assert!(
            (rocker - theta).abs() < ANGLE_EPS,
            "θ_rocker = θ_crank: {rocker} vs {theta}"
        );
        assert!(
            (coupler + theta).abs() < ANGLE_EPS,
            "θ_coupler = −θ_crank: {coupler} vs {}",
            -theta
        );
        // The driven joint is never a variable: the solver must not touch the crank.
        assert!(s.value("j_crank", 0).is_none(), "crank must stay driven");
    });
    assert_eq!(steps, 37, "the whole sweep ran");
}

#[test]
fn grashof_crank_rocker_closes_over_sweep() {
    let h = parse(GRASHOF);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    let coupler_len = 5f32.sqrt();
    let angles = (-10..=10).map(|i| (i as f32) * 0.05);
    sweep(&p, &mut pos, "j_crank", angles, &mut |theta, s| {
        assert!(
            s.errors[0].residual < RESIDUAL_EPS,
            "closure residual {} at crank {theta}",
            s.errors[0].residual
        );
        // The rocker angle must be a root of the loop-closure equation: the (rigid) coupler spans
        // crank tip R(θ)·(0,1) to rocker tip (2,0) + R(θ₄)·(0,2), so their distance is √5 exactly.
        let rocker = s.value("j_rocker", 0).expect("rocker is a free variable");
        let crank_tip = Vec2::new(-theta.sin(), theta.cos());
        let rocker_tip = Vec2::new(2.0 - 2.0 * rocker.sin(), 2.0 * rocker.cos());
        let dist = crank_tip.distance(rocker_tip);
        assert!(
            (dist - coupler_len).abs() < ANGLE_EPS,
            "loop-closure equation violated at crank {theta}: coupler span {dist} ≠ {coupler_len}"
        );
    });
}

#[test]
fn constraint_axes_mask_leaves_unmasked_rows_free() {
    // "1 1 0 0 0 0": only tx/ty count. The peg is 0.5 m away in tz FOREVER, but the mask hides it:
    // the solver just spins the arm to the peg's bearing (0.3 rad) and the closure reads CLOSED.
    let h = parse(MASKED);
    let p = problem(&h);
    let s = p.solve(&JointPositions::default(), None);
    let arm = s.value("j_arm", 0).expect("arm is a free variable");
    assert!(
        (arm - 0.3).abs() < ANGLE_EPS,
        "arm must rotate to 0.3, got {arm}"
    );
    assert!(
        s.errors[0].residual < RESIDUAL_EPS,
        "masked closure must read closed, residual {}",
        s.errors[0].residual
    );
    assert!(!s.errors[0].open());
}

#[test]
fn unmasked_tz_reports_honestly_open_without_nan() {
    // The control: same geometry, tz constrained. Best effort still zeroes tx/ty (arm → 0.3) and
    // the report shows exactly the unfixable 0.5 m: finite, no panic, converged = false.
    let h = parse(MASKED_FULL);
    let p = problem(&h);
    let s = p.solve(&JointPositions::default(), None);
    assert!(!s.converged, "0.5 m tz cannot close");
    assert!(s.errors[0].open());
    assert!(
        (s.errors[0].residual - 0.5).abs() < 1e-3,
        "residual must be the tz offset, got {}",
        s.errors[0].residual
    );
    assert!(
        (s.errors[0].trans - 0.5).abs() < 1e-3,
        "the openness is pure translation, got {}",
        s.errors[0].trans
    );
    let arm = s.value("j_arm", 0).expect("arm is a free variable");
    assert!(arm.is_finite(), "best-effort iterate must stay finite");
    assert!((arm - 0.3).abs() < 1e-3, "tx/ty still solved, got {arm}");
}

#[test]
fn limit_blocked_closure_reports_open_then_recloses() {
    let h = parse(PARALLELOGRAM_BLOCKED);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    // Within the rocker's ±0.2 limit the mechanism follows normally…
    let reachable = (0..=4).map(|i| (i as f32) * 0.05);
    sweep(&p, &mut pos, "j_crank", reachable, &mut |theta, s| {
        assert!(
            s.errors[0].residual < RESIDUAL_EPS,
            "closed at crank {theta}"
        );
        let rocker = s.value("j_rocker", 0).unwrap();
        assert!((rocker - theta).abs() < ANGLE_EPS);
    });
    // …past it the closure cannot hold: best effort pins the rocker AT its limit, everything stays
    // finite, and the report says OPEN (converged = false, residual way over tolerance).
    let blocked = (5..=12).map(|i| (i as f32) * 0.05);
    sweep(&p, &mut pos, "j_crank", blocked, &mut |theta, s| {
        assert!(!s.converged, "crank {theta} is unreachable");
        assert!(
            s.errors[0].open(),
            "closure must report open at crank {theta}"
        );
        let rocker = s.value("j_rocker", 0).unwrap();
        assert!(rocker.is_finite() && s.value("j_coupler", 0).unwrap().is_finite());
        assert!(
            (0.19..=0.2 + 1e-5).contains(&rocker),
            "best effort ends at the rocker limit, got {rocker}"
        );
    });
    // Releasing the crank back inside the reachable range re-closes the loop.
    pos.set_dof("j_crank", 0, 0.1);
    let s = p.solve(&pos, Some("j_crank"));
    assert!(
        s.errors[0].residual < RESIDUAL_EPS,
        "re-closed after release, residual {}",
        s.errors[0].residual
    );
    assert!((s.value("j_rocker", 0).unwrap() - 0.1).abs() < ANGLE_EPS);
}

#[test]
fn mimic_in_loop_with_driven_source_closes() {
    // Coupler mimics the DRIVEN crank ×−1 (the parallelogram's own coupling): the rocker is the
    // only variable and the mimic's contribution rides through the shared resolver.
    let h = parse(PARA_MIMIC_DRIVEN);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    sweep(&p, &mut pos, "j_crank", crank_sweep(), &mut |theta, s| {
        assert!(
            s.errors[0].residual < RESIDUAL_EPS,
            "closed at crank {theta}"
        );
        let rocker = s.value("j_rocker", 0).expect("rocker is a free variable");
        assert!((rocker - theta).abs() < ANGLE_EPS, "{rocker} vs {theta}");
        // The mimic follower must never be a solver variable.
        assert!(
            s.value("j_coupler", 0).is_none(),
            "mimics are never variables"
        );
    });
}

#[test]
fn mimic_in_loop_with_variable_source_closes_via_chain_rule() {
    // Coupler mimics the ROCKER ×−1: perturbing the single variable moves TWO loop joints, so the
    // finite-difference Jacobian exercises the mimic chain rule end to end.
    let h = parse(PARA_MIMIC_VAR);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    sweep(&p, &mut pos, "j_crank", crank_sweep(), &mut |theta, s| {
        assert!(
            s.errors[0].residual < RESIDUAL_EPS,
            "closed at crank {theta}"
        );
        let rocker = s.value("j_rocker", 0).expect("rocker is a free variable");
        assert!((rocker - theta).abs() < ANGLE_EPS, "{rocker} vs {theta}");
        assert!(
            s.value("j_coupler", 0).is_none(),
            "mimics are never variables"
        );
    });
}

#[test]
fn limited_mimic_follower_reports_the_displayed_mechanism() {
    let h = parse(PARA_MIMIC_LIMITED);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    // Inside the follower's ±0.1 window the parallelogram behaves normally.
    pos.set_dof("j_crank", 0, 0.05);
    let s = p.solve(&pos, Some("j_crank"));
    assert!(
        s.errors[0].residual < RESIDUAL_EPS,
        "closed inside the follower limit, residual {}",
        s.errors[0].residual
    );
    assert!((s.value("j_rocker", 0).unwrap() - 0.05).abs() < ANGLE_EPS);
    // Past it, articulate renders the coupler CLAMPED at −0.1 while the mimic commands −0.5: the
    // displayed mechanism cannot close. The regression this pins: an unclamped evaluation "closed"
    // a phantom mechanism (residual 0, converged, calm gizmo) while the rendered one gaped ~0.79 m
    // at the closure: the status must tell the mechanism's truth instead.
    pos.set_dof("j_crank", 0, 0.5);
    let s = p.solve(&pos, Some("j_crank"));
    assert!(!s.converged, "the displayed mechanism cannot close");
    assert!(s.errors[0].open(), "the gizmo/panel must warn");
    // The reported gap is the DISPLAYED one, best-effort minimized: with crank 0.5 and the coupler
    // pinned at −0.1, the coupler's far end sits 1.7748 from the rocker pivot, so the best the
    // unit rocker can leave is a 0.7748 m translation gap, reached at θ_rocker ≈ 0.3672 (NOT the
    // phantom solution's 0.5).
    assert!(
        (s.errors[0].trans - 0.7748).abs() < 0.01,
        "translation gap {}",
        s.errors[0].trans
    );
    let rocker = s.value("j_rocker", 0).expect("rocker is a free variable");
    assert!(
        (rocker - 0.3672).abs() < 5e-3,
        "best-effort rocker under DISPLAY semantics, got {rocker}"
    );
    assert!(
        s.value("j_coupler", 0).is_none(),
        "the follower is never a variable"
    );
}

#[test]
fn two_independent_loops_both_close() {
    let h = parse(TWO_LOOPS);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    // Drive loop A's crank, then loop B's (the driver hand-off frees crank_a, but loop A is
    // already closed, so its zero residual holds it in place).
    pos.set_dof("j_crank_a", 0, 0.4);
    let s = p.solve(&pos, Some("j_crank_a"));
    for v in &s.values {
        pos.set_dof(&v.joint, v.dof, v.value);
    }
    pos.set_dof("j_crank_b", 0, -0.3);
    let s = p.solve(&pos, Some("j_crank_b"));
    for v in &s.values {
        pos.set_dof(&v.joint, v.dof, v.value);
    }
    assert!(
        s.errors.iter().all(|e| e.residual < RESIDUAL_EPS),
        "both loops closed"
    );
    assert!(
        (pos.dof("j_crank_a", 0) - 0.4).abs() < 1e-3,
        "closed loop A undisturbed"
    );
    assert!((pos.dof("j_rocker_a", 0) - 0.4).abs() < 1e-3);
    assert!((pos.dof("j_coupler_a", 0) + 0.4).abs() < 1e-3);
    assert!((pos.dof("j_rocker_b", 0) + 0.3).abs() < ANGLE_EPS);
    assert!((pos.dof("j_coupler_b", 0) - 0.3).abs() < ANGLE_EPS);
}

#[test]
fn five_bar_sharing_a_joint_solves_as_one_stacked_system() {
    // Double parallelogram: rocker_m sits on BOTH loop paths, so the 12-row stacked system couples
    // the two closures through it. Closed form: every link parallels the crank.
    let h = parse(FIVEBAR_SHARED);
    let p = problem(&h);
    let mut pos = JointPositions::default();
    let angles = (-8..=8).map(|i| (i as f32) * 0.05);
    sweep(&p, &mut pos, "j_crank", angles, &mut |theta, s| {
        assert!(
            s.errors.iter().all(|e| e.residual < RESIDUAL_EPS),
            "both closures closed at crank {theta}"
        );
        assert!(
            (s.value("j_rm", 0).unwrap() - theta).abs() < ANGLE_EPS,
            "middle rocker"
        );
        assert!(
            (s.value("j_rf", 0).unwrap() - theta).abs() < ANGLE_EPS,
            "far rocker"
        );
        assert!(
            (s.value("j_c1", 0).unwrap() + theta).abs() < ANGLE_EPS,
            "coupler 1"
        );
        assert!(
            (s.value("j_c2", 0).unwrap() + theta).abs() < ANGLE_EPS,
            "coupler 2"
        );
    });
}

#[test]
fn weld_closure_across_the_rotation_vector_pi_branch_converges() {
    // θ = π + 1e-4 (one FD step past π) is the regression case: the seed's error measures on the
    // shortest arc just past −π while the bumped FD evaluation lands just under +π, without
    // branch-aligned differencing that representation flip produced a ~2π/1e-4 garbage Jacobian
    // column, every DLS step increased the norm, and this fully closable weld was reported stuck
    // open at residual ≈ π after exhausting the reject budget. θ = π − 1e-4 and π pin the rest of
    // the boundary's neighborhood.
    for rot in [
        std::f64::consts::PI - 1e-4,
        std::f64::consts::PI,
        std::f64::consts::PI + 1e-4,
    ] {
        let h = parse(&weld_near_pi(rot));
        let p = problem(&h);
        let s = p.solve(&JointPositions::default(), None);
        assert!(
            s.converged,
            "θ = {rot}: must converge, residual {}",
            s.errors[0].residual
        );
        assert!(
            s.errors[0].residual < RESIDUAL_EPS,
            "θ = {rot}: residual {}",
            s.errors[0].residual
        );
        // The arm lands on the weld angle modulo 2π (either representative is inside ±4 rad).
        let arm = s.value("j_arm", 0).expect("arm is a free variable");
        let wrapped = (arm - rot as f32).rem_euclid(std::f32::consts::TAU);
        assert!(
            wrapped < 1e-3 || std::f32::consts::TAU - wrapped < 1e-3,
            "θ = {rot}: arm settled at {arm}"
        );
    }
}

#[test]
fn exactly_assembled_universal_closure_reads_closed() {
    // The regression: the first-order `a×a2` rotation projector this replaced measured a phantom
    // ≈q²/2 on a PERFECTLY assembled U-joint: 0.0050 rad at q = 0.1 (50× OPEN_TOL, a warning-red
    // gizmo on a mechanism that is exactly closed), growing to ~0.10 rad at q = 0.6. The exact
    // manifold-invariant row measures zero at every angle.
    let h = parse(UNIVERSAL_ASSEMBLED);
    let p = problem(&h);
    for i in 1..=12 {
        let q = (i as f32) * 0.05; // 0.05 … 0.6 rad, spanning q = 0.1 and q = 0.6
        let mut pos = JointPositions::default();
        pos.set_dof("j_gimbal", 0, q);
        // Drive the gimbal (and hence, through the mimic, the cap): no free variable remains, so the
        // solve is a pure measurement of the closure the projector produces.
        let s = p.solve(&pos, Some("j_gimbal"));
        assert!(
            s.value("j_gimbal", 0).is_none() && s.value("j_cap", 0).is_none(),
            "no free variables: the driver is held and the cap is a mimic"
        );
        assert!(
            s.errors[0].residual < RESIDUAL_EPS,
            "exactly-assembled universal must read closed at q = {q}, residual {}",
            s.errors[0].residual
        );
        assert!(
            !s.errors[0].open(),
            "no phantom open flag at q = {q} (residual {})",
            s.errors[0].residual
        );
    }
}

#[test]
fn zero_driver_solve_assembles_an_open_pose() {
    // The pure-core self-assembly case: no driver at all (the fresh-load state), every loop joint is
    // free and the slightly-open zero pose closes.
    let h = parse(PARALLELOGRAM_OPEN);
    let p = problem(&h);
    let s = p.solve(&JointPositions::default(), None);
    assert!(
        s.errors[0].residual < RESIDUAL_EPS,
        "self-assembly residual {}",
        s.errors[0].residual
    );
    assert!(!s.values.is_empty() && s.values.iter().all(|v| v.value.is_finite()));
}

// ──────────────────────────────────────── ECS helpers ────────────────────────────────────────

/// Counts frames on which [`JointPositions`] was dirtied: the no-churn probe.
#[derive(Resource, Default)]
struct WriteCount(usize);

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<HcdfDoc>()
        .init_resource::<WriteCount>()
        .add_plugins(ScenePlugin)
        .add_plugins(JointsPlugin)
        .add_systems(
            Update,
            (|mut c: ResMut<WriteCount>| c.0 += 1).run_if(resource_changed::<JointPositions>),
        );
    app
}

fn load(app: &mut App, xml: &str) {
    app.world_mut().resource_mut::<HcdfDoc>().0 = Some(Arc::new(Hcdf::from_xml_str(xml).unwrap()));
    app.update(); // rebuild_on_change + reset + solve + articulate
    app.update(); // flush commands / settle the solver's own write-back
}

/// Write one slider-like edit (position + driven) and settle: the solve happens on the first
/// update, its write-back settles on the second, the rest is margin.
fn drive(app: &mut App, joint: &str, q: f32) {
    app.world_mut()
        .resource_mut::<JointPositions>()
        .set_dof(joint, 0, q);
    app.world_mut().resource_mut::<DrivenJoint>().0 = Some(joint.to_string());
    for _ in 0..6 {
        app.update();
    }
}

/// Map comp name → its spawned entity.
fn comp_entities(app: &mut App) -> HashMap<String, Entity> {
    let world = app.world_mut();
    let mut q = world.query::<(Entity, &CompEntity)>();
    q.iter(world).map(|(e, c)| (c.name.clone(), e)).collect()
}

/// Every entity's local transform, in a stable order: the churn-free-rest comparison snapshot.
fn transform_snapshot(app: &mut App) -> Vec<(Entity, Transform)> {
    let world = app.world_mut();
    let mut q = world.query::<(Entity, &Transform)>();
    let mut v: Vec<(Entity, Transform)> = q.iter(world).map(|(e, t)| (e, *t)).collect();
    v.sort_by_key(|(e, _)| *e);
    v
}

fn status_entries(app: &App) -> &LoopClosureStatus {
    app.world().resource::<LoopClosureStatus>()
}

// ───────────────────────────────────────── ECS tests ─────────────────────────────────────────

#[test]
fn ecs_drive_writes_solved_coordinates_and_poses_entities() {
    let mut app = build_app();
    load(&mut app, PARALLELOGRAM);
    drive(&mut app, "j_crank", 0.5);

    // Solved passive coordinates land in the single source of truth (the sliders SHOW them)…
    let (rocker_q, coupler_q) = {
        let p = app.world().resource::<JointPositions>();
        (p.dof("j_rocker", 0), p.dof("j_coupler", 0))
    };
    assert!((rocker_q - 0.5).abs() < 1e-3, "rocker solved to {rocker_q}");
    assert!(
        (coupler_q + 0.5).abs() < 1e-3,
        "coupler solved to {coupler_q}"
    );

    // …and articulate applied them to the entities the same frame chain (the rocker's local pose
    // is its joint origin rotated by the SOLVED angle).
    let names = comp_entities(&mut app);
    let t = *app
        .world()
        .entity(names["rocker"])
        .get::<Transform>()
        .unwrap();
    assert!(
        t.rotation
            .abs_diff_eq(Quat::from_rotation_z(rocker_q), 1e-4),
        "rocker entity rotation {:?}",
        t.rotation
    );

    // The status reports the closure closed.
    let s = status_entries(&app);
    assert_eq!(s.0.len(), 1);
    assert_eq!(s.0[0].name, "j_loop");
    assert!(s.0[0].error.is_some_and(|e| !e.open()), "closure closed");
}

#[test]
fn ecs_blocked_closure_flags_open_in_status() {
    let mut app = build_app();
    load(&mut app, PARALLELOGRAM_BLOCKED);
    drive(&mut app, "j_crank", 0.6);

    let s = status_entries(&app);
    assert!(
        s.0[0].error.is_some_and(|e| e.open()),
        "blocked closure must report open"
    );
    let p = app.world().resource::<JointPositions>();
    let rocker = p.dof("j_rocker", 0);
    assert!(
        rocker.is_finite() && rocker <= 0.2 + 1e-5,
        "best effort at the limit: {rocker}"
    );
}

#[test]
fn converged_rest_thirty_frames_no_churn() {
    // The connector_stability.rs discipline: once the mechanism converged and nothing changes,
    // JointPositions must never be dirtied again and no transform may be rewritten.
    let mut app = build_app();
    load(&mut app, PARALLELOGRAM);
    drive(&mut app, "j_crank", 0.3);
    for _ in 0..4 {
        app.update(); // extra settle margin beyond drive()'s own
    }

    let writes_before = app.world().resource::<WriteCount>().0;
    let snap_before = transform_snapshot(&mut app);
    for _ in 0..N_FRAMES {
        app.update();
    }
    assert_eq!(
        app.world().resource::<WriteCount>().0,
        writes_before,
        "JointPositions dirtied during converged rest"
    );
    assert_eq!(
        transform_snapshot(&mut app),
        snap_before,
        "a transform was rewritten during converged rest"
    );
}

#[test]
fn reload_resets_solver_state_with_no_stale_writes() {
    let mut app = build_app();
    load(&mut app, PARALLELOGRAM);
    drive(&mut app, "j_crank", 0.5);
    assert!(
        !app.world().resource::<JointPositions>().0.is_empty(),
        "precondition: posed"
    );
    assert!(
        app.world().resource::<DrivenJoint>().0.is_some(),
        "precondition: driven"
    );

    // Reload onto a LOOP-FREE doc: commands cleared, driver forgotten, closure status emptied,
    // and nothing writes stale loop coordinates for joints that no longer exist.
    load(&mut app, NOLOOP);
    assert!(
        app.world().resource::<JointPositions>().0.is_empty(),
        "reload must clear the commanded positions"
    );
    assert!(
        app.world().resource::<DrivenJoint>().0.is_none(),
        "driver forgotten"
    );
    assert!(
        status_entries(&app).0.is_empty(),
        "no closures ⇒ empty status"
    );

    // Reload the four-bar again: its zero pose is authored CLOSED, so the solver has nothing to
    // write: the fresh doc rests at zero with a clean command map and a closed status entry.
    load(&mut app, PARALLELOGRAM);
    assert!(
        app.world().resource::<JointPositions>().0.is_empty(),
        "closed-at-zero doc needs no solver writes at load"
    );
    let s = status_entries(&app);
    assert_eq!(s.0.len(), 1);
    assert!(s.0[0].error.is_some_and(|e| !e.open()));
}

#[test]
fn slightly_open_doc_self_assembles_at_load() {
    // The self-assembly case in the real app: a doc whose zero pose is open loads with DrivenJoint = None
    // (all loop joints free) and assembles itself, no slider ever touched.
    let mut app = build_app();
    load(&mut app, PARALLELOGRAM_OPEN);
    for _ in 0..4 {
        app.update();
    }
    let s = status_entries(&app);
    assert!(
        s.0[0].error.is_some_and(|e| !e.open()),
        "open zero pose must self-assemble at load"
    );
    assert!(
        !app.world().resource::<JointPositions>().0.is_empty(),
        "self-assembly writes the solved coordinates"
    );
    assert!(
        app.world().resource::<DrivenJoint>().0.is_none(),
        "nothing drove it"
    );
}

#[test]
fn solver_off_is_passthrough_and_toggle_on_closes() {
    // Solver OFF = exactly today's behavior: the crank command is the ONLY entry
    // in JointPositions, the passive links rest at their local origins (the mechanism pulls apart
    // downstream), and the status entries exist but are unevaluated (checkbox stays offered).
    let mut app = build_app();
    app.world_mut().resource_mut::<LoopSolveEnabled>().0 = false;
    load(&mut app, PARALLELOGRAM);
    drive(&mut app, "j_crank", 0.4);

    {
        let p = app.world().resource::<JointPositions>();
        assert_eq!(p.0.len(), 1, "no solver writes while off");
        assert!(p.0.contains_key("j_crank"));
    }
    let names = comp_entities(&mut app);
    let coupler = *app
        .world()
        .entity(names["coupler"])
        .get::<Transform>()
        .unwrap();
    assert!(
        (coupler.translation - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6
            && coupler.rotation.abs_diff_eq(Quat::IDENTITY, 1e-6),
        "passive coupler rests at its origin while the solver is off"
    );
    let s = status_entries(&app);
    assert_eq!(
        s.0.len(),
        1,
        "closures still listed (the checkbox needs them)"
    );
    assert!(s.0[0].error.is_none(), "…but unevaluated");

    // Flipping the toggle back ON re-solves the held pose immediately (the enable change alone
    // re-triggers the solver, no slider touch needed).
    app.world_mut().resource_mut::<LoopSolveEnabled>().0 = true;
    for _ in 0..6 {
        app.update();
    }
    let p = app.world().resource::<JointPositions>();
    assert!(
        (p.dof("j_rocker", 0) - 0.4).abs() < 1e-3,
        "toggle-on closes the mechanism around the held crank"
    );
    assert!(status_entries(&app).0[0].error.is_some_and(|e| !e.open()));
}
