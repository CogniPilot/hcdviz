//! The in-memory asset store must defeat Bevy's PATH-KEYED asset cache across clean document opens.
//!
//! Bevy's `AssetServer` caches loaded assets by path and keeps them alive while any strong handle
//! survives, so merely swapping bytes in the store's `Dir` used to leave a clean open rendering the
//! previous document's geometry for a shared uri (and serving stale bytes for a removed one). These
//! tests drive the REAL asset pipeline headlessly (MinimalPlugins + AssetPlugin + the registered
//! mem source + the STL loader; no GPU, no window) and assert on what the LOADED store reflects, not
//! just the `Dir`: a clean open of different bytes at the SAME uri loads the new bytes even while the
//! old scene's handle is still alive, and a uri the new document does not carry is genuinely gone.
use bevy::asset::{AssetPlugin, LoadState, UnapprovedPathMode};
use bevy::mesh::{Mesh, VertexAttributeValues};
use bevy::prelude::*;
use hcdviz::mem_assets::{insert_bundle_assets, replace_bundle_assets, MemAssetStore};

/// ASCII STL with `tri_count` triangles, so the loaded mesh's vertex count reflects WHICH bytes the
/// asset pipeline actually read (each triangle contributes three unshared vertices).
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

/// Headless app over the REAL asset pipeline: the mem source registered as the default (exactly like
/// the bins, BEFORE `AssetPlugin`), with an EMPTY on-disk fallback root so only the store can serve.
/// `mode` picks the `AssetPlugin` unapproved-path policy: the hcdviz bin runs Bevy's default
/// (`Forbid`; its uris are all relative), while `dendrite_build` runs `Allow` because its placed
/// modules load leading-slash `/mem/<n>/...` uris, which `Forbid` rejects.
fn build_app(mode: UnapprovedPathMode) -> App {
    let fs_root = std::env::temp_dir().join("hcdviz_mem_pipeline_empty_root");
    std::fs::create_dir_all(&fs_root).expect("create empty fallback root");
    let mut app = App::new();
    hcdviz::mem_assets::register_mem_asset_source(&mut app, fs_root.to_str().unwrap());
    app.add_plugins((
        MinimalPlugins,
        AssetPlugin {
            unapproved_path_mode: mode,
            ..Default::default()
        },
    ))
    .init_asset::<Mesh>()
    .add_plugins(hcdviz::stl::StlPlugin);
    app
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

fn load_state(app: &App, handle: &Handle<Mesh>) -> LoadState {
    app.world()
        .resource::<AssetServer>()
        .load_state(handle.id())
}

/// Vertex count of the loaded mesh behind `handle` (`None` while not loaded).
fn vertex_count(app: &App, handle: &Handle<Mesh>) -> Option<usize> {
    let mesh = app.world().resource::<Assets<Mesh>>().get(handle)?;
    match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(v)) => Some(v.len()),
        _ => None,
    }
}

/// Open A, then a clean open of B carrying DIFFERENT bytes at the SAME uri: the loaded asset must
/// reflect B's bytes. A's handle is deliberately kept alive across the swap, since a surviving strong
/// handle is exactly what used to pin the stale cache entry onto the new document's load.
#[test]
fn clean_open_same_uri_loads_the_new_documents_bytes() {
    let mut app = build_app(UnapprovedPathMode::Forbid);
    let store = app.world().resource::<MemAssetStore>().clone();
    let uri = "assets/shared.stl";

    // Accepted open of document A: one triangle behind the uri.
    replace_bundle_assets(&store, &[(uri.to_string(), ascii_stl(1))]);
    let path_a = store.asset_path(uri);
    let handle_a: Handle<Mesh> = app.world().resource::<AssetServer>().load(path_a.clone());
    drive_until(&mut app, "document A's mesh to load", |app| {
        vertex_count(app, &handle_a).is_some()
    });
    assert_eq!(vertex_count(&app, &handle_a), Some(3), "A is one triangle");

    // Clean open of document B: two triangles behind the SAME uri. handle_a stays alive.
    replace_bundle_assets(&store, &[(uri.to_string(), ascii_stl(2))]);
    let path_b = store.asset_path(uri);
    assert_ne!(
        path_a, path_b,
        "each clean open must load fresh asset paths, or the cache serves A's bytes"
    );
    let handle_b: Handle<Mesh> = app.world().resource::<AssetServer>().load(path_b);
    drive_until(&mut app, "document B's mesh to load", |app| {
        vertex_count(app, &handle_b).is_some()
    });
    assert_eq!(
        vertex_count(&app, &handle_b),
        Some(6),
        "the loaded asset must reflect B's bytes, not A's cached mesh"
    );
    drop(handle_a);
}

/// Open A, then a clean open of a document that does NOT carry A's uri: the uri must be genuinely gone
/// from the LOADED store (a fresh load fails, and A's mesh leaves `Assets<Mesh>` once its handle
/// drops), not merely absent from the `Dir` while the cache keeps serving it.
#[test]
fn cleared_uri_is_gone_from_the_loaded_store() {
    let mut app = build_app(UnapprovedPathMode::Forbid);
    let store = app.world().resource::<MemAssetStore>().clone();
    let uri = "assets/only_in_a.stl";

    replace_bundle_assets(&store, &[(uri.to_string(), ascii_stl(1))]);
    let handle_a: Handle<Mesh> = app
        .world()
        .resource::<AssetServer>()
        .load(store.asset_path(uri));
    drive_until(&mut app, "document A's mesh to load", |app| {
        vertex_count(app, &handle_a).is_some()
    });

    // Clean open of a document without the uri (a plain `.hcdf`: empty asset set).
    replace_bundle_assets(&store, &[]);
    let plain = store.asset_path(uri);
    assert_eq!(plain, uri, "an untracked uri resolves plain");
    let handle_gone: Handle<Mesh> = app.world().resource::<AssetServer>().load(plain);
    drive_until(&mut app, "the cleared uri's load to fail", |app| {
        matches!(load_state(app, &handle_gone), LoadState::Failed(_))
    });
    assert!(
        vertex_count(&app, &handle_gone).is_none(),
        "a cleared uri must not serve any bytes"
    );

    // Once the outgoing scene's strong handle drops, the stale mesh leaves the loaded store too; its
    // stamped path is unreachable from any new load, so nothing can resurrect it.
    drop(handle_a);
    drive_until(
        &mut app,
        "A's stale mesh to leave the loaded store",
        |app| {
            app.world()
                .resource::<Assets<Mesh>>()
                .iter()
                .next()
                .is_none()
        },
    );
}

/// `dendrite_build` layers a placed module's meshes into the SHARED store under ABSOLUTE
/// `/mem/<n>/assets/<name>` keys (its wasm add-include and keep-live restore flows) and renders them
/// through the same `asset_path` translation as everything else. The stamp must strip back to that
/// key VERBATIM: a plain absolute load and a stamped load must both serve the inserted bytes, and a
/// clean-open generation bump must serve a re-inserted absolute key under the fresh stamp. Runs with
/// `UnapprovedPathMode::Allow`, as `dendrite_build` configures its `AssetPlugin` (a leading-slash
/// load is unapproved under Bevy's default `Forbid` and never reaches the reader).
#[test]
fn absolute_module_dir_keys_load_plain_and_stamped() {
    let mut app = build_app(UnapprovedPathMode::Allow);
    let store = app.world().resource::<MemAssetStore>().clone();
    let uri = "/mem/0/assets/rotor.stl";

    // The additive insert (the merge/place flow) keys the Dir by the absolute uri verbatim.
    insert_bundle_assets(&store, &[(uri.to_string(), ascii_stl(1))]);

    // A plain absolute load passes the reader untouched and hits the absolute key.
    let plain: Handle<Mesh> = app.world().resource::<AssetServer>().load(uri);
    drive_until(&mut app, "the plain absolute load", |app| {
        vertex_count(app, &plain).is_some()
    });
    assert_eq!(vertex_count(&app, &plain), Some(3), "one triangle, plain");

    // The stamped load path (what every scene load uses) must strip back to the SAME absolute key.
    let stamped = store.asset_path(uri);
    assert_ne!(stamped, uri, "a tracked uri resolves stamped");
    let handle: Handle<Mesh> = app.world().resource::<AssetServer>().load(stamped.clone());
    drive_until(&mut app, "the stamped absolute load", |app| {
        vertex_count(app, &handle).is_some()
    });
    assert_eq!(
        vertex_count(&app, &handle),
        Some(3),
        "the stamped load must serve the absolute key's bytes, not miss on a stripped-relative path"
    );

    // A clean open re-carrying the SAME absolute key: the bumped stamp still strips back to it and
    // serves the NEW bytes.
    replace_bundle_assets(&store, &[(uri.to_string(), ascii_stl(2))]);
    let restamped = store.asset_path(uri);
    assert_ne!(restamped, stamped, "a clean open moves the stamp");
    let handle_fresh: Handle<Mesh> = app.world().resource::<AssetServer>().load(restamped);
    drive_until(&mut app, "the re-stamped absolute load", |app| {
        vertex_count(app, &handle_fresh).is_some()
    });
    assert_eq!(
        vertex_count(&app, &handle_fresh),
        Some(6),
        "the fresh generation must serve the re-inserted bytes"
    );
}
