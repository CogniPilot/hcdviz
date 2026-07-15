//! # hcdviz: Hardware Configuration Descriptive Visualizer
//!
//! A clean, read-only static HCDF viewer on Bevy 0.19 (rviz for HCDF). Parsing + the sha-pinned
//! official schema come from the [`hcdformat`] crate (re-exported as [`schema`]). Rendering reacts to
//! the read-only [`doc::HcdfDoc`] resource; hcdviz never mutates it.
//!
//! Embedders (`dendrite_build`) add their own overlays/panels via [`display::AddDisplayExt`] without
//! touching hcdviz internals: the rviz plugin-host model.
pub use hcdformat as schema;

pub mod camera;
pub mod connectivity;
pub mod connector;
pub mod datasheet;
pub mod display;
pub mod doc;
pub mod frame;
pub mod geometry;
pub mod hierarchy;
pub mod joints;
pub mod kinematics;
pub mod loop_solver;
pub mod mem_assets;
pub mod network;
pub mod open;
pub mod physical;
pub mod pick;
pub mod scene;
pub mod standalone_connectivity;
pub mod stl;
pub mod ui;
pub mod web_clipboard;

#[cfg(test)]
mod tests;

use bevy::prelude::*;
use display::AddDisplayExt;

/// Renderer core: schema verify, the read-only doc resource + loader, scene build, orbit camera,
/// picking/selection. No egui, so an embedder can use core alone.
pub struct HcdvizCorePlugin;
impl Plugin for HcdvizCorePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<doc::HcdfDoc>()
            .init_resource::<doc::SchemaStatus>()
            .init_resource::<display::DisplayRegistry>()
            .init_resource::<camera::OrbitCamera>()
            .add_message::<doc::LoadHcdf>()
            .add_systems(Startup, (doc::verify_schema, camera::setup_camera))
            .add_systems(
                Update,
                (
                    doc::load_hcdf_system.before(scene::SceneSet::Rebuild),
                    camera::orbit_camera,
                    camera::fit_to_scene,
                ),
            )
            .add_plugins(stl::StlPlugin)
            .add_plugins(scene::ScenePlugin)
            .add_plugins(connectivity::ConnectivityPlugin)
            .add_plugins(joints::JointsPlugin)
            .add_plugins(scene::HighlightPlugin)
            .add_plugins(pick::PickPlugin);
    }
}

/// egui shell (status + display toggle tree + inspector) PLUS the runtime file-open control. Separate
/// from core so an embedder can use core alone, and so the [`open`] feature (an extra writer of the doc
/// via [`doc::LoadHcdf`]) is scoped to the default viewer, NOT to embedders like `dendrite_build` that
/// compose [`HcdvizCorePlugin`] directly and own their own open flow.
///
/// The [`open::drain_open_channel`] system stages each pick's asset set for the
/// [`doc::load_hcdf_system`] acceptance swap of the [`mem_assets::MemAssetStore`] resource (an accepted
/// `.hcdfz` open populates it, and the meshes then render from RAM on both targets), so the bin must
/// register it (via [`mem_assets::register_mem_asset_source`]) before `AssetPlugin`; the hcdviz bin does
/// exactly that.
pub struct HcdvizUiPlugin;
impl Plugin for HcdvizUiPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<bevy_egui::EguiPlugin>() {
            app.add_plugins(bevy_egui::EguiPlugin::default());
        }
        app.init_resource::<open::OpenChannel>()
            .init_resource::<connector::ActiveConnector>()
            .add_plugins(web_clipboard::WebClipboardPlugin)
            .add_systems(Update, open::drain_open_channel)
            .add_systems(
                bevy_egui::EguiPrimaryContextPass,
                (
                    // Style tuning first so even the first frame's popups get the taller combo
                    // height + solid scrollbars (see ui::tune_egui_style).
                    ui::apply_style_tuning,
                    ui::panels,
                    ui::joints_panel,
                    hierarchy::hierarchy_panel,
                )
                    .chain(),
            );
    }
}

/// The full default viewer: core + ui + all built-in displays. One-line embed for a bin or
/// dendrite_build.
pub struct HcdvizAppPlugin;
impl Plugin for HcdvizAppPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(HcdvizCorePlugin)
            .add_plugins(HcdvizUiPlugin);
        app.add_display(scene::VisualDisplay)
            .add_display(scene::KinematicsDisplay)
            .add_display(scene::SensorsDisplay)
            .add_display(scene::SensorAxisAlignDisplay)
            .add_display(scene::FramesDisplay)
            .add_display(scene::ConnectivityDisplay)
            .add_display(network::NetworksDisplay)
            .add_display(scene::CollisionDisplay)
            .add_display(scene::InertialDisplay)
            .add_display(scene::GridDisplay);
    }
}
