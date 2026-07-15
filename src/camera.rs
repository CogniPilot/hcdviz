//! rviz-style hand-rolled orbit camera (modernized from dendrite-viewer `update_camera`, Bevy 0.17 →
//! 0.19). Left-drag orbits, right-drag pans, scroll zooms; the camera fits the scene on each load.
//!
//! Modernizations vs the 0.17 source: input via `MessageReader<MouseMotion>`/`MouseWheel` (events are
//! Messages in 0.19), `Camera3d` + `Projection::Perspective` component spawn, and a `SceneFitted`
//! message-driven fit instead of polling registry positions. egui pointer-capture is respected so
//! dragging on a panel doesn't move the camera.
use bevy::input::mouse::{MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_egui::EguiContexts;

/// Orbit-camera state (spherical coords around a focus point), smoothed toward targets.
#[derive(Resource)]
pub struct OrbitCamera {
    pub focus: Vec3,
    pub target_focus: Vec3,
    pub distance: f32,
    pub target_distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub orbit_sensitivity: f32,
    pub zoom_speed: f32,
    pub smooth: f32,
    pub min_distance: f32,
    pub max_distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            focus: Vec3::ZERO,
            target_focus: Vec3::ZERO,
            distance: 3.0,
            target_distance: 3.0,
            yaw: -std::f32::consts::FRAC_PI_4,
            pitch: 0.5,
            orbit_sensitivity: 0.005,
            // Per-line zoom rate: one wheel notch multiplies the distance by e^-zoom_speed ≈ 0.86
            // (a ~14 % step). See [`zoom_factor`] for why this is exponential, not linear.
            zoom_speed: 0.15,
            smooth: 0.20,
            min_distance: 0.02,
            max_distance: 200.0,
        }
    }
}

/// Marks the single main camera entity.
#[derive(Component)]
pub struct MainCamera;

/// Sent by the scene builder once a load is rendered so the camera can frame the bounds.
#[derive(Message)]
pub struct SceneFitted {
    pub center: Vec3,
    pub radius: f32,
}

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 0.001,
            far: 2000.0,
            ..default()
        }),
        Transform::from_xyz(2.0, 2.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));
}

/// Frame the camera to a freshly built scene's bounds.
pub fn fit_to_scene(mut fitted: MessageReader<SceneFitted>, mut cam: ResMut<OrbitCamera>) {
    for ev in fitted.read() {
        cam.target_focus = ev.center;
        cam.focus = ev.center;
        // Pull back to comfortably contain a sphere of the given radius at the default FOV.
        let dist = (ev.radius.max(0.05) * 2.5).clamp(cam.min_distance, cam.max_distance);
        cam.target_distance = dist;
        cam.distance = dist;
    }
}

/// Orbit / pan / zoom driven by the mouse, modernized to 0.19 Message readers.
pub fn orbit_camera(
    mut cam: ResMut<OrbitCamera>,
    mut q: Query<&mut Transform, With<MainCamera>>,
    mut motion: MessageReader<MouseMotion>,
    mut wheel: MessageReader<MouseWheel>,
    buttons: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    contexts: Option<EguiContexts>,
) {
    // Respect egui pointer-capture WHEN egui is present, but don't require it: HcdvizCorePlugin must run
    // without an egui plugin (e.g. an embedder that supplies its own UI), so `contexts` is optional and
    // a missing egui simply means "no capture" rather than a panic.
    let egui_capture = contexts
        .and_then(|mut c| c.ctx_mut().ok().map(|ctx| ctx.egui_wants_pointer_input()))
        .unwrap_or(false);
    let delta: Vec2 = motion.read().map(|m| m.delta).sum();

    if !egui_capture {
        if buttons.pressed(MouseButton::Left) {
            cam.yaw -= delta.x * cam.orbit_sensitivity;
            cam.pitch = (cam.pitch - delta.y * cam.orbit_sensitivity).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.05,
                std::f32::consts::FRAC_PI_2 - 0.05,
            );
        }
        if buttons.pressed(MouseButton::Right) {
            // Pan in the camera's right/up plane.
            let (right, up) = camera_right_up(cam.yaw, cam.pitch);
            let pan = cam.distance * 0.0015;
            let shift = -right * delta.x * pan + up * delta.y * pan;
            cam.target_focus += shift;
        }
        // Normalize each wheel message to line-equivalent units BEFORE summing: pixel-unit deltas
        // (trackpads, and every browser/wasm wheel event) arrive ~100/notch, so treating them as
        // line units (as the old zoom did) saturated the factor on a single tick and teleported the
        // camera. Normalizing + the exponential [`zoom_factor`] is the "giant jump" fix.
        let scroll: f32 = wheel.read().map(|w| wheel_lines(w.unit, w.y)).sum();
        if scroll != 0.0 {
            cam.target_distance = (cam.target_distance * zoom_factor(scroll, cam.zoom_speed))
                .clamp(cam.min_distance, cam.max_distance);
        }
    } else {
        wheel.read().for_each(drop);
    }

    // Smooth toward targets (frame-rate independent).
    let dt = time.delta_secs();
    let lerp = 1.0 - (-cam.smooth * 60.0 * dt).exp();
    let lerp = lerp.clamp(0.0, 1.0);
    cam.distance += (cam.target_distance - cam.distance) * lerp;
    let df = (cam.target_focus - cam.focus) * lerp;
    cam.focus += df;

    if let Ok(mut t) = q.single_mut() {
        let offset = orbit_offset(cam.yaw, cam.pitch, cam.distance);
        t.translation = cam.focus + offset;
        t.look_at(cam.focus, Vec3::Y);
    }
}

/// Bevy Y-up spherical offset from yaw/pitch/distance.
fn orbit_offset(yaw: f32, pitch: f32, distance: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(distance * cp * sy, distance * sp, distance * cp * cy)
}

fn camera_right_up(yaw: f32, pitch: f32) -> (Vec3, Vec3) {
    let forward = -orbit_offset(yaw, pitch, 1.0).normalize();
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    (right, up)
}

/// Pixel-unit wheel deltas (trackpads and browser/wasm wheel events) arrive at roughly this many
/// pixels per physical notch; dividing by it puts them on the same ~1-per-notch scale as a classic
/// mouse wheel's [`MouseScrollUnit::Line`] events, so one gesture zooms the same amount on any device.
const PIXELS_PER_LINE: f32 = 100.0;

/// Upper bound on the line-equivalent scroll consumed in a single frame. Momentum scrolling and coarse
/// browser wheels can deliver a large burst at once; clamping keeps even the worst burst to a smooth
/// fraction of the distance instead of a jump to a clamp rail.
const MAX_ZOOM_LINES_PER_FRAME: f32 = 4.0;

/// One wheel message's vertical delta in line-equivalent units (see [`PIXELS_PER_LINE`]). This is the
/// crux of the zoom fix: the old code read `w.y` directly, so a ~100 px browser notch was treated as
/// 100 line-notches at once.
fn wheel_lines(unit: MouseScrollUnit, y: f32) -> f32 {
    match unit {
        MouseScrollUnit::Line => y,
        MouseScrollUnit::Pixel => y / PIXELS_PER_LINE,
    }
}

/// Multiplicative zoom factor for a line-equivalent scroll amount at the given per-line `rate`.
///
/// Exponential, so it is always positive, symmetric (scrolling in by N then out by N returns to the
/// exact starting distance), and monotonic in the delta, unlike the old `1 - rate·scroll` factor,
/// which went NEGATIVE for a large delta and had to be floored, producing the giant single-tick jump.
/// The scroll is clamped to [`MAX_ZOOM_LINES_PER_FRAME`] first so a burst can't slam a clamp rail.
/// Positive scroll shrinks the distance (zoom in), matching the previous sign convention.
fn zoom_factor(scroll_lines: f32, rate: f32) -> f32 {
    let s = scroll_lines.clamp(-MAX_ZOOM_LINES_PER_FRAME, MAX_ZOOM_LINES_PER_FRAME);
    (-s * rate).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    // The default per-line zoom rate ([`OrbitCamera::default`]); the tests assert the feel at it.
    const RATE: f32 = 0.15;

    #[test]
    fn pixel_wheel_no_longer_saturates() {
        // A browser/trackpad notch is ~100 px. The OLD linear factor `1 - 100*0.075` = -6.5 (floored
        // to 0.05) teleported the camera to 5 % of its distance. Normalized + exponential, one notch
        // is a gentle step.
        let f = zoom_factor(wheel_lines(MouseScrollUnit::Pixel, 100.0), RATE);
        assert!(
            (0.80..0.95).contains(&f),
            "one pixel-notch should be a gentle step, got {f}"
        );
    }

    #[test]
    fn line_and_pixel_notch_feel_the_same() {
        let line = zoom_factor(wheel_lines(MouseScrollUnit::Line, 1.0), RATE);
        let pixel = zoom_factor(wheel_lines(MouseScrollUnit::Pixel, 100.0), RATE);
        assert!(
            (line - pixel).abs() < 1e-6,
            "a wheel notch should feel identical across devices: line={line} pixel={pixel}"
        );
    }

    #[test]
    fn zoom_is_symmetric_in_then_out() {
        // Scroll in by N then out by N must return to the exact starting distance (no drift), which
        // the old `max(0.05)`-floored factor could not guarantee once it saturated.
        let d0 = 5.0f32;
        let d2 = d0 * zoom_factor(3.0, RATE) * zoom_factor(-3.0, RATE);
        assert!((d2 - d0).abs() < 1e-5, "in-then-out drifted: {d0} -> {d2}");
    }

    #[test]
    fn direction_is_preserved() {
        assert!(
            zoom_factor(1.0, RATE) < 1.0,
            "positive scroll zooms in (distance shrinks)"
        );
        assert!(
            zoom_factor(-1.0, RATE) > 1.0,
            "negative scroll zooms out (distance grows)"
        );
    }

    #[test]
    fn a_giant_burst_is_bounded_not_a_teleport() {
        // A momentum flick of thousands of pixels must clamp to the per-frame cap, not collapse the
        // distance onto the rail.
        let f = zoom_factor(wheel_lines(MouseScrollUnit::Pixel, 5000.0), RATE);
        assert_eq!(
            f,
            zoom_factor(MAX_ZOOM_LINES_PER_FRAME, RATE),
            "burst should clamp to the per-frame cap"
        );
        assert!(
            f > 0.5,
            "even the worst single frame stays a smooth fraction, got {f}"
        );
    }
}
