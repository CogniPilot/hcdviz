//! A `.hcdfz` bundle passed as the native startup argument loads (no GPU / no window).
//!
//! The `LoadHcdf::Path` flow reads the startup file to a `String`, which fails on a bundle's ZIP bytes, so
//! a `.hcdfz` CLI argument used to error at startup. The fix routes a bundle startup arg through the SAME
//! byte pipeline as a user pick ([`hcdviz::open::enqueue_if_bundle`] -> `drain_open_channel` -> bundle
//! extraction -> a staged `LoadHcdf::Open` whose asset set swaps in on acceptance). This drives that path
//! headlessly: pack a real bundle, point the startup routing at it, and assert the doc loads with its
//! meshes served from RAM.
use bevy::prelude::*;
use hcdviz::doc::{load_hcdf_system, HcdfDoc, LoadHcdf, SchemaStatus};
use hcdviz::mem_assets::MemAssetStore;
use hcdviz::open::{drain_open_channel, enqueue_if_bundle, OpenChannel};
use hcdviz::schema::{open_bundle_bytes, pack_to_bytes, Hcdf, MemBundle};
use std::path::Path;

// One comp with a collision mesh so the packed bundle carries an asset (the packer content-addresses the
// `wheel.stl` site into `assets/wheel_<sha>.stl`), letting the test prove asset extraction, not just parse.
const BUNDLE_DOC: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="startup-bundle" body-frame="FLU" world-frame="ENU">
  <comp name="c">
    <collision name="col"><geometry><mesh uri="wheel.stl"/></geometry></collision>
  </comp>
</hcdf>"#;

/// Pack `BUNDLE_DOC` + one mesh into `.hcdfz` bytes.
fn bundle_bytes() -> Vec<u8> {
    let doc = Hcdf::from_xml_str(BUNDLE_DOC).expect("doc parses");
    let mut assets = MemBundle::new();
    assets.insert(
        "wheel.stl".to_string(),
        b"solid wheel\nendsolid wheel\n".to_vec(),
    );
    let (bytes, _report) = pack_to_bytes(&doc, &assets, false).expect("bundle packs");
    bytes
}

/// A headless app mirroring the native bin's load wiring: the open channel + in-memory asset store, and
/// the `drain_open_channel -> load_hcdf_system` chain (drain stages the `LoadHcdf::Open`, load parses it
/// and swaps the staged assets in on acceptance).
fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<OpenChannel>()
        .init_resource::<MemAssetStore>()
        .init_resource::<HcdfDoc>()
        .init_resource::<SchemaStatus>()
        .add_message::<LoadHcdf>()
        .add_systems(Update, (drain_open_channel, load_hcdf_system).chain());
    app
}

#[test]
fn startup_arg_bundle_loads_with_assets() {
    let bytes = bundle_bytes();
    // The rewritten asset path the store must end up serving (learned by opening the bundle directly).
    let opened = open_bundle_bytes(&bytes).expect("bundle opens");
    let (asset_path, _) = opened
        .assets
        .first()
        .expect("bundle carries a mesh asset")
        .clone();

    // Write the bundle where the startup routing will read it (native reads the CLI path off disk).
    let path = std::env::temp_dir().join("hcdviz_startup_bundle_test.hcdfz");
    std::fs::write(&path, &bytes).expect("write temp bundle");

    let mut app = build_app();
    let channel = app.world().resource::<OpenChannel>().clone();
    // The startup routing must recognize the ZIP as a bundle and enqueue it (a plain `.hcdf` returns
    // false and the bin keeps the LoadHcdf::Path flow instead).
    assert!(
        enqueue_if_bundle(&channel, &path),
        "a .hcdfz startup arg must route through the byte pipeline"
    );
    app.update();

    let doc = app.world().resource::<HcdfDoc>();
    let loaded = doc.0.as_ref().expect("the bundle's root doc must load");
    assert_eq!(
        loaded.name, "startup-bundle",
        "the packed doc must be shown"
    );

    let store = app.world().resource::<MemAssetStore>();
    assert!(
        store.0.get_asset(Path::new(&asset_path)).is_some(),
        "the bundle's mesh must be served from RAM under {asset_path}"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn plain_hcdf_startup_arg_stays_on_the_path_flow() {
    // A plain `.hcdf` is NOT a ZIP, so the startup routing declines it (returns false) and the bin keeps
    // the LoadHcdf::Path flow, which resolves native includes and the sibling-mesh disk fallback.
    let path = std::env::temp_dir().join("hcdviz_startup_plain_test.hcdf");
    std::fs::write(&path, BUNDLE_DOC).expect("write temp hcdf");
    let channel = OpenChannel::default();
    assert!(
        !enqueue_if_bundle(&channel, &path),
        "a plain .hcdf must not be routed through the bundle byte pipeline"
    );
    let _ = std::fs::remove_file(&path);
}
