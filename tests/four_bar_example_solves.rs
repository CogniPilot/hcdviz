//! Regression: the SHIPPED `examples/four-bar.hcdf` (in the sibling hcdformat repo) must actually
//! SOLVE through the loop-closure solver, not merely validate.
//!
//! This closes a real coverage gap. The example and the solver live in different repos, so the
//! solver's own fixtures (inline XML) exercise the *shape* of a four-bar but never the *file we ship*.
//! An early cut of the example closed the loop onto `rocker` directly, whose frame sits at its fixed
//! ground pivot, not the moving pin, so it validated 0/0 yet, when driven, left the rocker limp and
//! the closure gaping open. The fix (a zero-DOF `rocker_tip` frame fixed at the rocker's far end, so
//! the closure child frame IS the pin) is invisible to the validator; only driving the mechanism
//! catches it. Hence this test drives the real file and asserts the parallelogram closed form.
//!
//! The example is located relative to this crate: the same sibling-repo layout `.cargo/config.toml`
//! already assumes for the in-tree `hcdformat` path dependency. A packaged build without the sibling
//! checkout (e.g. a crates.io `hcdformat`) simply has no file to test, so the test skips loudly rather
//! than failing on an environment it cannot control.
use hcdviz::joints::JointPositions;
use hcdviz::kinematics::build_kinematic_tree;
use hcdviz::loop_solver::LoopProblem;
use hcdviz::schema::Hcdf;

/// Path to the shipped example, resolved from this crate's manifest via the sibling-repo layout that
/// `.cargo/config.toml` already pins for the `hcdformat` path dependency.
const EXAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../git/cps_describe/hcdformat/examples/four-bar.hcdf"
);

#[test]
fn shipped_four_bar_example_solves_as_a_parallelogram() {
    let Ok(xml) = std::fs::read_to_string(EXAMPLE) else {
        eprintln!("SKIP: sibling example not present at {EXAMPLE} (packaged build without the sibling repo)");
        return;
    };

    let h = Hcdf::from_xml_str(&xml).expect("shipped four-bar.hcdf parses");
    let tree = build_kinematic_tree(&h);
    let p = LoopProblem::build(&h, &tree).expect("shipped four-bar.hcdf declares a loop closure");

    // Drive the crank through its full ±60° range in 5° steps, out and back through zero, continuing
    // from each solved pose exactly as a slider drag would (keeps the solver on branch).
    let mut pos = JointPositions::default();
    let up = (0..=12).map(|i| (i as f32) * 5f32.to_radians());
    let down = (-12..=11).rev().map(|i| (i as f32) * 5f32.to_radians());
    for theta in up.chain(down) {
        pos.set_dof("j_crank", 0, theta);
        let s = p.solve(&pos, Some("j_crank"));
        for v in &s.values {
            pos.set_dof(&v.joint, v.dof, v.value);
        }

        // The closure must actually close (nothing gaping open) at every step.
        assert!(
            !s.errors.iter().any(|e| e.open()),
            "closure went open at crank={theta:.4}: {:?}",
            s.errors
        );
        // Parallelogram closed form: rocker tracks the crank, coupler is its negative.
        let rocker = s.value("j_rocker", 0).expect("j_rocker is a free variable");
        let coupler = s
            .value("j_coupler", 0)
            .expect("j_coupler is a free variable");
        assert!(
            (rocker - theta).abs() < 1e-3,
            "rocker should track the crank: crank={theta:.4} rocker={rocker:.4}"
        );
        assert!(
            (coupler + theta).abs() < 1e-3,
            "coupler should be the crank's negative: crank={theta:.4} coupler={coupler:.4}"
        );
    }
}
