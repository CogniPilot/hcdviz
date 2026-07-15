//! The asset-store swap must be transactional with document ACCEPTANCE (no GPU / no window).
//!
//! The open flow used to replace the in-memory asset store BEFORE the picked file was known to parse,
//! so a failed open stranded the still-rendered previous document without its backing assets. The swap
//! is now staged through `LoadHcdf::Open` and applied by `load_hcdf_system` only after a successful
//! parse. This drives the REAL pipeline headlessly (the picker channel -> `drain_open_channel` ->
//! `load_hcdf_system`, over MinimalPlugins + AssetPlugin + the registered mem source + the STL loader)
//! and asserts a failed open leaves the old document rendering with its assets serving, while a later
//! ACCEPTED open still swaps them out. The variant contracts ride the same seam and are pinned here
//! too: an accepted `LoadHcdf::Path` is a CLEAN open that clears the store like an `Open`, while
//! `LoadHcdf::Xml` (the embedder republish seam) retains it by contract.
use bevy::asset::{AssetPlugin, LoadState};
use bevy::mesh::{Mesh, VertexAttributeValues};
use bevy::prelude::*;
use hcdviz::doc::{load_hcdf_system, HcdfDoc, LoadHcdf, SchemaStatus};
use hcdviz::mem_assets::MemAssetStore;
use hcdviz::open::{drain_open_channel, OpenChannel, PickedDoc};
use hcdviz::schema::{open_bundle_bytes, pack_to_bytes, Hcdf, MemBundle};
use std::path::Path;

/// ASCII STL with `tri_count` triangles (three unshared vertices each), so the loaded mesh's vertex
/// count reflects which document's bytes the pipeline actually served.
fn ascii_stl(tri_count: usize) -> Vec<u8> {
    let mut s = String::from("solid t\n");
    for i in 0..tri_count {
        let z = i as f32;
        s.push_str(&format!(
            "facet normal 0 0 1\nouter loop\nvertex 0 0 {z}\nvertex 1 0 {z}\nvertex 0 1 {z}\nendloop\nendfacet\n"
        ));
    }
    s.push_str("endsolid t\n");
    s.into_bytes()
}

/// Pack a one-comp doc named `doc_name` whose collision references `mesh_name`, carrying `mesh_bytes`,
/// into `.hcdfz` bytes. Returns the bundle bytes plus the content-addressed `assets/<name>` uri the
/// packed root doc references (learned by opening the bundle back).
fn bundle(doc_name: &str, mesh_name: &str, mesh_bytes: Vec<u8>) -> (Vec<u8>, String) {
    let xml = format!(
        r#"<?xml version="1.0"?>
<hcdf version="1.0" name="{doc_name}" body-frame="FLU" world-frame="ENU">
  <comp name="c">
    <collision name="col"><geometry><mesh uri="{mesh_name}"/></geometry></collision>
  </comp>
</hcdf>"#
    );
    let doc = Hcdf::from_xml_str(&xml).expect("doc parses");
    let mut assets = MemBundle::new();
    assets.insert(mesh_name.to_string(), mesh_bytes);
    let (bytes, _report) = pack_to_bytes(&doc, &assets, false).expect("bundle packs");
    let opened = open_bundle_bytes(&bytes).expect("bundle opens");
    let (uri, _) = opened
        .assets
        .first()
        .expect("bundle carries a mesh")
        .clone();
    (bytes, uri)
}

// Unterminated element: NOT a ZIP, so `open_to_xml` stages it as plain text with an empty asset set
// and the parse failure happens in `load_hcdf_system`, after the staging.
const MALFORMED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="bad" body-frame="FLU" world-frame="ENU"><comp"#;

/// Headless app over the REAL open pipeline: the picker channel and the drain -> load chain, with the
/// mem source registered as the default asset source (exactly like the bins, BEFORE `AssetPlugin`) and
/// an EMPTY on-disk fallback root so only the store can serve meshes.
fn build_app() -> App {
    let fs_root = std::env::temp_dir().join("hcdviz_open_transactional_empty_root");
    std::fs::create_dir_all(&fs_root).expect("create empty fallback root");
    let mut app = App::new();
    hcdviz::mem_assets::register_mem_asset_source(&mut app, fs_root.to_str().unwrap());
    app.add_plugins((MinimalPlugins, AssetPlugin::default()))
        .init_asset::<Mesh>()
        .add_plugins(hcdviz::stl::StlPlugin)
        .init_resource::<OpenChannel>()
        .init_resource::<HcdfDoc>()
        .init_resource::<SchemaStatus>()
        .add_message::<LoadHcdf>()
        .add_systems(Update, (drain_open_channel, load_hcdf_system).chain());
    app
}

/// Deliver a pick exactly as the platform picker does: push onto the shared channel queue.
fn push_pick(app: &App, name: &str, bytes: Vec<u8>) {
    let channel = app.world().resource::<OpenChannel>();
    match channel.0.lock() {
        Ok(mut q) => q.push(PickedDoc {
            name: name.to_string(),
            bytes,
        }),
        Err(poisoned) => poisoned.into_inner().push(PickedDoc {
            name: name.to_string(),
            bytes,
        }),
    }
}

fn doc_name(app: &App) -> String {
    app.world()
        .resource::<HcdfDoc>()
        .0
        .as_ref()
        .expect("a document is loaded")
        .name
        .clone()
}

/// Pump the app until `done` holds (asset IO + loading run on the task pool, so poll with updates).
fn drive_until(app: &mut App, what: &str, mut done: impl FnMut(&mut App) -> bool) {
    for _ in 0..5000 {
        app.update();
        if done(app) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    panic!("timed out waiting for {what}");
}

/// Vertex count of the loaded mesh behind `handle` (`None` while not loaded).
fn vertex_count(app: &App, handle: &Handle<Mesh>) -> Option<usize> {
    let mesh = app.world().resource::<Assets<Mesh>>().get(handle)?;
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(v)) => Some(v.len()),
        _ => None,
    }
}

fn load_state(app: &App, handle: &Handle<Mesh>) -> LoadState {
    app.world()
        .resource::<AssetServer>()
        .load_state(handle.id())
}

/// Open bundle A through the real byte pipeline and return its tracked mesh uri, asserting the open
/// was accepted and the asset set is tracked (the shared preamble of the variant-contract tests).
fn open_bundle_a(app: &mut App) -> String {
    let (bytes_a, uri_a) = bundle("doc-a", "wheel.stl", ascii_stl(1));
    push_pick(app, "a.hcdfz", bytes_a);
    app.update();
    assert_eq!(doc_name(app), "doc-a");
    let store = app.world().resource::<MemAssetStore>().clone();
    assert_ne!(
        store.asset_path(&uri_a),
        uri_a,
        "an accepted open must track + stamp its assets"
    );
    uri_a
}

/// A failed open over a loaded document must be a pure no-op for that document AND its assets: the doc
/// stays shown, the store still tracks and serves its mesh bytes under the SAME stamped path (no
/// generation bump), and a later ACCEPTED open still swaps the set out.
#[test]
fn failed_open_keeps_previous_doc_and_its_assets_serving() {
    let mut app = build_app();
    let (bytes_a, uri_a) = bundle("doc-a", "wheel.stl", ascii_stl(1));

    // Accepted open of bundle A through the real byte pipeline.
    push_pick(&app, "a.hcdfz", bytes_a);
    app.update();
    assert_eq!(doc_name(&app), "doc-a");
    let store = app.world().resource::<MemAssetStore>().clone();
    let path_a = store.asset_path(&uri_a);
    assert_ne!(
        path_a, uri_a,
        "an accepted open must track + stamp its assets"
    );
    let handle_a: Handle<Mesh> = app.world().resource::<AssetServer>().load(path_a.clone());
    drive_until(&mut app, "document A's mesh to load", |app| {
        vertex_count(app, &handle_a).is_some()
    });
    assert_eq!(vertex_count(&app, &handle_a), Some(3), "A is one triangle");

    // A malformed pick reaches the parser and FAILS there. The staged (empty) asset set is dropped;
    // nothing about A changes.
    push_pick(&app, "bad.hcdf", MALFORMED.as_bytes().to_vec());
    app.update();
    assert_eq!(doc_name(&app), "doc-a", "the failed open must keep A shown");
    let status = app.world().resource::<SchemaStatus>();
    assert!(
        status.message.contains("parse error"),
        "the failure must be reported: {}",
        status.message
    );
    assert_eq!(
        store.asset_path(&uri_a),
        path_a,
        "A's serving asset set (and its generation) must be untouched"
    );
    assert!(
        store.0.get_asset(Path::new(&uri_a)).is_some(),
        "A's backing bytes must still be in the store"
    );
    assert_eq!(
        vertex_count(&app, &handle_a),
        Some(3),
        "A's already-loaded mesh must keep serving"
    );

    // A later ACCEPTED open still swaps: bundle B replaces A's asset set.
    let (bytes_b, uri_b) = bundle("doc-b", "fin.stl", ascii_stl(2));
    push_pick(&app, "b.hcdfz", bytes_b);
    app.update();
    assert_eq!(doc_name(&app), "doc-b");
    assert_eq!(
        store.asset_path(&uri_a),
        uri_a,
        "A's uri must be untracked after B is accepted"
    );
    assert!(
        store.0.get_asset(Path::new(&uri_a)).is_none(),
        "A's bytes must be gone after B is accepted"
    );
    let path_b = store.asset_path(&uri_b);
    assert_ne!(path_b, uri_b, "B's assets are tracked + stamped");
    let handle_b: Handle<Mesh> = app.world().resource::<AssetServer>().load(path_b);
    drive_until(&mut app, "document B's mesh to load", |app| {
        vertex_count(app, &handle_b).is_some()
    });
    assert_eq!(vertex_count(&app, &handle_b), Some(6), "B is two triangles");
}

/// An accepted `LoadHcdf::Path` is a CLEAN open and must clear the store like an `Open`: after a
/// bundle open, a path-loaded plain `.hcdf` leaves NO stale tracked asset resolvable (the bundle's
/// uri untracks, its bytes leave the store, and a fresh load of it genuinely fails).
#[test]
fn accepted_path_load_clears_the_prior_bundles_assets() {
    let mut app = build_app();
    let uri_a = open_bundle_a(&mut app);
    let store = app.world().resource::<MemAssetStore>().clone();

    // A plain `.hcdf` opened from disk through the startup Path flow.
    let dir = std::env::temp_dir().join("hcdviz_open_transactional_path_doc");
    std::fs::create_dir_all(&dir).expect("create doc dir");
    let path = dir.join("plain.hcdf");
    std::fs::write(
        &path,
        r#"<?xml version="1.0"?>
<hcdf version="1.0" name="doc-p" body-frame="FLU" world-frame="ENU"><comp name="c"/></hcdf>"#,
    )
    .expect("write plain doc");
    app.world_mut().write_message(LoadHcdf::Path(path));
    app.update();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(doc_name(&app), "doc-p", "the Path load is accepted");
    assert_eq!(
        store.asset_path(&uri_a),
        uri_a,
        "A's uri must be untracked: a Path load is a clean open"
    );
    assert!(
        store.0.get_asset(Path::new(&uri_a)).is_none(),
        "A's bytes must be gone from the store"
    );
    // And the stale uri is unresolvable through the real pipeline (the fallback root is empty).
    let handle: Handle<Mesh> = app
        .world()
        .resource::<AssetServer>()
        .load(uri_a.to_string());
    drive_until(&mut app, "the stale uri's load to fail", |app| {
        matches!(load_state(app, &handle), LoadState::Failed(_))
    });
}

/// An accepted `LoadHcdf::Xml` must RETAIN the store (the embedder republish seam): an embedder that
/// layered assets through `insert_bundle_assets` and republishes edited XML must not lose them. After
/// a bundle open, an Xml load replaces the document while the tracked set keeps serving under its
/// UNCHANGED stamp (no clear, no generation bump).
#[test]
fn accepted_xml_load_retains_the_prior_bundles_assets() {
    let mut app = build_app();
    let uri_a = open_bundle_a(&mut app);
    let store = app.world().resource::<MemAssetStore>().clone();
    let path_a = store.asset_path(&uri_a);

    let xml = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="doc-x" body-frame="FLU" world-frame="ENU"><comp name="c"/></hcdf>"#;
    app.world_mut()
        .write_message(LoadHcdf::Xml(xml.to_string()));
    app.update();

    assert_eq!(doc_name(&app), "doc-x", "the Xml load is accepted");
    assert_eq!(
        store.asset_path(&uri_a),
        path_a,
        "the tracked set and its stamp must be untouched by an Xml publish"
    );
    assert!(
        store.0.get_asset(Path::new(&uri_a)).is_some(),
        "A's bytes must still be in the store"
    );
    let handle: Handle<Mesh> = app.world().resource::<AssetServer>().load(path_a);
    drive_until(&mut app, "the retained mesh to load", |app| {
        vertex_count(app, &handle).is_some()
    });
    assert_eq!(
        vertex_count(&app, &handle),
        Some(3),
        "the retained asset must keep serving its bytes"
    );
}
