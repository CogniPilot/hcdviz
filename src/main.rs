//! Native hcdviz binary: `hcdviz [path/to/file.hcdf]`.
//!
//! HCDF mesh URIs are document-relative (e.g. `assets/foo.glb` next to the .hcdf), so the default asset
//! source registered below (see [`hcdviz::mem_assets`]) resolves them: a runtime-opened `.hcdfz`'s
//! meshes from RAM on both targets, everything else (native) from the startup file's directory on disk.
//! (The .hcdf text itself is read via `std::fs`, independent of the asset source.) This wires the
//! startup-arg file; a runtime-opened `.hcdf` in a different directory still needs bundling to carry
//! its meshes (the on-disk root is fixed at launch).
use bevy::prelude::*;
use hcdviz::doc::LoadHcdf;
use hcdviz::standalone_connectivity::StandaloneConnectivityProducerPlugin;
use hcdviz::HcdvizAppPlugin;
use std::path::PathBuf;

#[derive(Resource)]
struct StartupArg(Option<PathBuf>);

fn main() {
    // Browser: surface Rust panics as real console.error messages (a wasm panic otherwise shows only
    // "unreachable executed"). Native is unaffected. Must run before anything that can panic.
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    // Resolve to an absolute path so the fs read and the asset root agree.
    let path = std::env::args().nth(1).map(|a| {
        let p = PathBuf::from(a);
        std::fs::canonicalize(&p).unwrap_or(p)
    });
    // NATIVE asset root = the .hcdf's directory; a URI like "assets/x.glb" then resolves under it via
    // the default source's filesystem fallback. (The browser has no disk root.)
    #[cfg(not(target_arch = "wasm32"))]
    let asset_root = path
        .as_ref()
        .and_then(|p| p.parent())
        .map(|d| d.to_string_lossy().into_owned())
        .unwrap_or_else(|| "assets".to_string());

    let mut app = App::new();
    // Register the in-memory asset source as the DEFAULT, BEFORE AssetPlugin (it consumes the default
    // source when it builds, so this registration supersedes `AssetPlugin.file_path`). A runtime-opened
    // `.hcdfz` feeds its meshes into this store so `asset_server.load("assets/<name>")` reads from RAM
    // on BOTH targets. Native chains a filesystem fallback at the startup asset root, keeping the CLI
    // flow (sibling meshes on disk) unchanged; the sandboxed browser serves from RAM alone.
    #[cfg(not(target_arch = "wasm32"))]
    hcdviz::mem_assets::register_mem_asset_source(&mut app, &asset_root);
    #[cfg(target_arch = "wasm32")]
    hcdviz::mem_assets::register_mem_asset_source(&mut app);
    app.add_plugins(
        DefaultPlugins
            // Web: fit the canvas to its parent so it fills the browser viewport. Bevy defaults this
            // off → a fixed-resolution canvas leaving empty space (the black gap). No-op on native.
            // The viewer plays no audio; bevy's `audio` feature is compiled out (Cargo.toml),
            // so no AudioPlugin exists to open an output device (rodio→cpal→WebAudio), so the
            // browser's "AudioContext was not allowed to start" noise never happens.
            .set(bevy::window::WindowPlugin {
                primary_window: Some(bevy::window::Window {
                    fit_canvas_to_parent: true,
                    ..default()
                }),
                ..default()
            }),
    )
    .add_plugins(HcdvizAppPlugin)
    .add_plugins(StandaloneConnectivityProducerPlugin)
    .insert_resource(StartupArg(path))
    .add_systems(Startup, kick_load)
    .run();
}

fn kick_load(
    arg: Res<StartupArg>,
    open: Res<hcdviz::open::OpenChannel>,
    mut ev: MessageWriter<LoadHcdf>,
) {
    if let Some(p) = &arg.0 {
        // A `.hcdfz` bundle's bytes are a ZIP; the LoadHcdf::Path flow reads the file as text and fails
        // on them, so route a bundle startup arg through the same byte pipeline the picker uses (bundle
        // extraction + meshes served from RAM). A plain `.hcdf` stays on the Path flow, which resolves
        // native <include>s and falls back to sibling meshes on disk.
        if hcdviz::open::enqueue_if_bundle(&open, p) {
            return;
        }
        ev.write(LoadHcdf::Path(p.clone()));
    }
}
