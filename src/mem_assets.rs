//! In-memory asset source: serve a runtime-opened `.hcdfz` bundle's meshes to Bevy from RAM.
//!
//! A robot is a `.hcdf` doc PLUS sibling mesh files (`assets/<name>.stl` / `.glb`). In the browser the
//! page is sandboxed: opening the single `.hcdf` does NOT bring its sibling meshes, and the native
//! filesystem [`bevy::asset::AssetPlugin`] source cannot fetch them (it reads over HTTP / a registered
//! reader, not the user's disk). On native the same problem appears at RUNTIME: the on-disk asset root
//! is fixed at launch, so a `.hcdfz` picked through the Open button used to extract its assets and then
//! discard them. The portable unit on both targets is therefore the self-contained `.hcdfz` bundle
//! (root `.hcdf` + content-addressed `assets/`); its assets are extracted in memory
//! ([`hcdformat::open_bundle_bytes`]) and served to Bevy through THIS module's reader so
//! `asset_server.load("assets/<name>.stl")` reads the bytes from RAM.
//!
//! ## How it works
//! Bevy 0.19 already ships a clone-able, `Arc`-backed in-memory filesystem ([`bevy::asset::io::memory::Dir`])
//! and a [`bevy::asset::io::memory::MemoryAssetReader`] over it. We:
//!   1. hold ONE [`Dir`] in the [`MemAssetStore`] resource (cloneable; every clone shares the same inner
//!      store);
//!   2. register, BEFORE `AssetPlugin`, a reader over that [`Dir`] as the **DEFAULT** asset source
//!      (`bevy::asset::AssetSourceId::Default`), so EVERY load of a document uri (visual GLB + collision
//!      STL, both formed by `scene.rs` as `asset_server.load(store.asset_path(uri))` from the
//!      document-relative `assets/<name>` path) resolves against the in-memory store;
//!   3. on a `.hcdfz` open, insert the bundle's `(assets/<name>, bytes)` entries into the [`Dir`] (keyed by
//!      that SAME document-relative path) and trigger a scene rebuild.
//!
//! ## Clean opens vs Bevy's path-keyed asset cache
//! Bevy's `AssetServer` caches loaded assets BY PATH: `load(path)` on an already-loaded path returns the
//! cached handle without re-reading the reader, and the cached value survives as long as any strong
//! handle does (during a document swap the outgoing scene's handles are still alive). Swapping bytes in
//! the [`Dir`] therefore does NOT invalidate anything already loaded: a shared uri would keep rendering
//! the PREVIOUS document's geometry, and a removed uri would keep serving its stale bytes. The store
//! defeats that cache with GENERATION-STAMPED load paths: every clean-open swap
//! ([`replace_bundle_assets`]) bumps a generation counter, and [`MemAssetStore::asset_path`] resolves a
//! tracked document uri to `mem-gen-<N>/<uri>`, a path Bevy has never loaded, so a clean open always
//! reads fresh bytes and a removed uri resolves plain and genuinely misses. The reader strips the
//! generation segment before the [`Dir`] lookup, so the [`Dir`] itself stays keyed by the PLAIN document
//! uri, whether that uri is document-relative (`assets/<name>`) or module-dir-absolute
//! (`/mem/<n>/assets/<name>`, `dendrite_build`'s placed modules): additive consumers keep resolving
//! their existing paths, and direct byte readback (`store.0.get_asset(uri)`) keeps working unchanged.
//!
//! ## Native vs wasm
//! On WASM the memory reader IS the whole source (the sandboxed page has no disk). On NATIVE the reader
//! chains a filesystem fallback at the startup asset root: a path missing from the store reads from disk
//! exactly as the plain `AssetPlugin` source did, so the CLI flow (a `.hcdf` with sibling meshes on
//! disk) is unchanged while a runtime-opened bundle's meshes come from RAM. Registering the default
//! source supersedes `AssetPlugin.file_path` (Bevy only applies it when no default source exists), which
//! is why the native [`register_mem_asset_source`] takes the root that `main` used to hand to
//! `AssetPlugin`.
use bevy::asset::io::memory::{Dir, MemoryAssetReader};
use bevy::asset::io::{
    AssetReader, AssetReaderError, AssetSourceBuilder, AssetSourceId, PathStream, Reader,
};
use bevy::asset::AssetApp;
use bevy::prelude::*;
use std::path::Path;
use std::sync::{Arc, Mutex};

#[cfg(not(target_arch = "wasm32"))]
use bevy::asset::io::file::FileAssetReader;

/// Leading segment of a generation-stamped load path: `mem-gen-<N>/<uri>`. The reader strips a
/// matching prefix before the [`Dir`] lookup ([`strip_generation`]), so every stamp of a uri reads the
/// same plain-keyed entry. A document uri that itself starts with this shape would be canonicalized
/// too; real store keys are document-relative `assets/<name>` paths or module-dir-absolute
/// `/mem/<n>/assets/<name>` paths (`dendrite_build`'s placed modules), so none do.
const GENERATION_PREFIX: &str = "mem-gen-";

/// The tracked state behind the store's mutex: which plain uris the CURRENT document's asset set
/// carries, and the clean-open generation whose stamp [`MemAssetStore::asset_path`] serves them under.
#[derive(Default)]
struct Ledger {
    /// Bumped by every [`replace_bundle_assets`] swap, so each clean open loads paths Bevy's path-keyed
    /// asset cache has never seen (a stale cached asset can then never be returned for a fresh open).
    generation: u64,
    /// The `assets/<name>` keys inserted through [`insert_bundle_assets`] / [`replace_bundle_assets`],
    /// so a CLEAN document open can remove EXACTLY the previous document's entries. Without it a later
    /// document that references a uri only an earlier bundle carried would be served the earlier bytes
    /// and mask its own missing asset.
    entries: Vec<String>,
}

/// The in-memory asset store: ONE clone-able [`Dir`] (an `Arc`-backed in-memory filesystem) that backs
/// the DEFAULT asset source's reader. Inserting into this resource's [`Dir`] makes the bytes fetchable
/// by `asset_server.load(store.asset_path(uri))` (the reader and this resource share the same inner
/// store). Held as a Bevy resource so the `.hcdfz` open path can populate it at runtime.
///
/// Field `.1` is the tracked [`Ledger`] (inserted keys + the clean-open generation). `Arc`-shared like
/// the [`Dir`], so every clone of the resource observes one live set.
#[derive(Resource, Clone, Default)]
pub struct MemAssetStore(pub Dir, Arc<Mutex<Ledger>>);

impl MemAssetStore {
    /// Lock the ledger, recovering from a poisoned lock rather than panicking: a file open must never
    /// crash the viewer.
    fn ledger(&self) -> std::sync::MutexGuard<'_, Ledger> {
        self.1.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The path to hand `asset_server.load` for a document uri.
    ///
    /// A uri the store currently tracks resolves to its generation-stamped form
    /// (`mem-gen-<N>/<uri>`): a path unique to the serving [`replace_bundle_assets`] swap, so Bevy's
    /// path-keyed asset cache can never return a PREVIOUS document's bytes for it (the cached value
    /// under an older stamp is unreachable from new loads and drops with the old scene's handles).
    /// An untracked uri resolves to itself, keeping the native on-disk fallback (and the honest
    /// missing-asset failure) on the plain path. Every load of a store-servable uri MUST go through
    /// this translation; a plain-uri load would resurrect exactly the stale-cache aliasing the stamp
    /// exists to prevent.
    ///
    /// The stamp prepends the generation segment plus exactly ONE separator, and the reader removes
    /// exactly that ([`strip_generation`]), so the [`Dir`] lookup key is the tracked uri VERBATIM. An
    /// ABSOLUTE tracked uri (`/mem/<n>/assets/<name>`) therefore stamps with a doubled separator
    /// (`mem-gen-<N>//mem/...`) and strips back to its leading-slash key intact.
    pub fn asset_path(&self, uri: &str) -> String {
        let ledger = self.ledger();
        if ledger.entries.iter().any(|e| e == uri) {
            format!("{GENERATION_PREFIX}{}/{uri}", ledger.generation)
        } else {
            uri.to_string()
        }
    }
}

/// Strip a leading `mem-gen-<N>/` stamp ([`MemAssetStore::asset_path`]'s) so the reader looks the
/// path up under its PLAIN document uri, which is how the [`Dir`] is keyed. The strip is TEXTUAL:
/// exactly the generation segment plus ONE separator are removed, so the remainder is byte-for-byte
/// the uri [`MemAssetStore::asset_path`] stamped, and an ABSOLUTE uri keeps its leading slash
/// (`mem-gen-1//mem/0/x` strips to `/mem/0/x`, the verbatim key `dendrite_build` inserts a placed
/// module's meshes under). A component-wise strip would collapse the doubled separator and hand back
/// a RELATIVE remainder that can never match an absolute-keyed [`Dir`] entry. A path without the
/// stamp (no such prefix, a non-digit generation, or no separator after the segment) is returned
/// unchanged.
fn strip_generation(path: &Path) -> &Path {
    let Some(s) = path.to_str() else {
        return path;
    };
    let Some(rest) = s.strip_prefix(GENERATION_PREFIX) else {
        return path;
    };
    let Some(sep) = rest.find('/') else {
        return path;
    };
    let digits = &rest[..sep];
    if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
        Path::new(&rest[sep + 1..])
    } else {
        path
    }
}

/// Register the in-memory reader as the DEFAULT asset source and insert the shared [`MemAssetStore`]
/// resource. MUST be called on the [`App`] BEFORE `DefaultPlugins`/`AssetPlugin` (the default source is
/// consumed when `AssetPlugin` builds; registering after it has no effect and Bevy logs an error). The
/// reader clones the SAME [`Dir`] the resource holds, so inserts made later through the resource are
/// visible to the reader.
///
/// NATIVE: the reader serves a path from the store when present and otherwise falls back to the on-disk
/// `fs_root` (the startup asset root, the loaded file's directory), replacing the `AssetPlugin.file_path`
/// source this registration supersedes.
#[cfg(not(target_arch = "wasm32"))]
pub fn register_mem_asset_source(app: &mut App, fs_root: &str) {
    let dir = Dir::default();
    let reader_dir = dir.clone();
    let fs_root = fs_root.to_string();
    // `AssetSourceBuilder::new(reader)` constructs the builder with our reader as the unprocessed
    // reader (it has no `default()`; a default source must carry a reader). The closure clones the
    // SAME `Dir` the resource holds, so runtime inserts are visible to the reader.
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || {
            Box::new(MemFirstAssetReader {
                mem: MemoryAssetReader {
                    root: reader_dir.clone(),
                },
                fs: FileAssetReader::new(&fs_root),
            })
        }),
    );
    app.insert_resource(MemAssetStore(dir, Default::default()));
}

/// Register the in-memory reader as the DEFAULT asset source and insert the shared [`MemAssetStore`]
/// resource. MUST be called on the [`App`] BEFORE `DefaultPlugins`/`AssetPlugin` (the default source is
/// consumed when `AssetPlugin` builds; registering after it has no effect and Bevy logs an error). The
/// reader clones the SAME [`Dir`] the resource holds, so inserts made later through the resource are
/// visible to the reader.
///
/// WASM: the memory reader is the whole source; the sandboxed page has no disk to fall back to.
#[cfg(target_arch = "wasm32")]
pub fn register_mem_asset_source(app: &mut App) {
    let dir = Dir::default();
    let reader_dir = dir.clone();
    // `AssetSourceBuilder::new(reader)` constructs the builder with our reader as the unprocessed
    // reader (it has no `default()`; a default source must carry a reader). The closure clones the
    // SAME `Dir` the resource holds, so runtime inserts are visible to the reader.
    app.register_asset_source(
        AssetSourceId::Default,
        AssetSourceBuilder::new(move || {
            Box::new(MemStoreReader {
                mem: MemoryAssetReader {
                    root: reader_dir.clone(),
                },
            })
        }),
    );
    app.insert_resource(MemAssetStore(dir, Default::default()));
}

/// WASM reader for the default source: the in-memory [`Dir`], addressed through
/// [`strip_generation`] so a generation-stamped load path reads its plain-keyed entry.
#[cfg(target_arch = "wasm32")]
struct MemStoreReader {
    mem: MemoryAssetReader,
}

#[cfg(target_arch = "wasm32")]
impl AssetReader for MemStoreReader {
    async fn read<'a>(&'a self, path: &'a Path) -> Result<impl Reader + 'a, AssetReaderError> {
        self.mem.read(strip_generation(path)).await
    }

    async fn read_meta<'a>(&'a self, path: &'a Path) -> Result<impl Reader + 'a, AssetReaderError> {
        self.mem.read_meta(strip_generation(path)).await
    }

    async fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<Box<PathStream>, AssetReaderError> {
        self.mem.read_directory(strip_generation(path)).await
    }

    async fn is_directory<'a>(&'a self, path: &'a Path) -> Result<bool, AssetReaderError> {
        self.mem.is_directory(strip_generation(path)).await
    }
}

/// NATIVE reader for the default source: memory-first, filesystem-fallback. A path present in the
/// in-memory [`Dir`] (a runtime-opened bundle's assets) is served from RAM; anything else falls back to
/// the on-disk asset root, preserving the CLI startup flow (sibling meshes next to the `.hcdf`). Only a
/// clean [`AssetReaderError::NotFound`] from the store falls through; a real error propagates.
///
/// The memory side is addressed through [`strip_generation`] (a stamped load path reads its
/// plain-keyed [`Dir`] entry); the filesystem fallback gets the ORIGINAL path, so a stamped path whose
/// entry was swapped away fails as missing instead of silently reading a same-named file off disk.
#[cfg(not(target_arch = "wasm32"))]
struct MemFirstAssetReader {
    mem: MemoryAssetReader,
    fs: FileAssetReader,
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetReader for MemFirstAssetReader {
    async fn read<'a>(&'a self, path: &'a Path) -> Result<impl Reader + 'a, AssetReaderError> {
        match self.mem.read(strip_generation(path)).await {
            Ok(r) => Ok(Box::new(r) as Box<dyn Reader + 'a>),
            Err(AssetReaderError::NotFound(_)) => {
                let r = self.fs.read(path).await?;
                Ok(Box::new(r) as Box<dyn Reader + 'a>)
            }
            Err(e) => Err(e),
        }
    }

    async fn read_meta<'a>(&'a self, path: &'a Path) -> Result<impl Reader + 'a, AssetReaderError> {
        match self.mem.read_meta(strip_generation(path)).await {
            Ok(r) => Ok(Box::new(r) as Box<dyn Reader + 'a>),
            Err(AssetReaderError::NotFound(_)) => {
                let r = self.fs.read_meta(path).await?;
                Ok(Box::new(r) as Box<dyn Reader + 'a>)
            }
            Err(e) => Err(e),
        }
    }

    async fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<Box<PathStream>, AssetReaderError> {
        match self.mem.read_directory(strip_generation(path)).await {
            Ok(stream) => Ok(stream),
            Err(AssetReaderError::NotFound(_)) => self.fs.read_directory(path).await,
            Err(e) => Err(e),
        }
    }

    async fn is_directory<'a>(&'a self, path: &'a Path) -> Result<bool, AssetReaderError> {
        // The store's answer is only authoritative when POSITIVE (a missing dir reports `false`, not
        // NotFound), so a negative defers to the on-disk root.
        match self.mem.is_directory(strip_generation(path)).await {
            Ok(true) => Ok(true),
            _ => self.fs.is_directory(path).await,
        }
    }
}

/// Insert every `(relative_path, bytes)` asset entry of an opened `.hcdfz` bundle into the in-memory
/// store ADDITIVELY, keyed by the document-relative `assets/<name>` uri the root doc carries. After
/// this, a scene rebuild's loads (`asset_server.load(store.asset_path(uri))`) resolve from RAM under
/// the CURRENT generation, so paths already resolved by earlier inserts keep loading unchanged.
/// Existing entries with the same path are overwritten in the [`Dir`]; note that Bevy's asset cache is
/// NOT invalidated by an additive overwrite, which is fine for the flows this serves (bundle asset
/// names are content-addressed, so a same-name overwrite carries the same bytes). This is the ADDITIVE
/// path (`dendrite_build`'s place/merge flows layer several sources into one store); a CLEAN document
/// open uses [`replace_bundle_assets`] so a prior document's assets cannot linger. Each key is recorded
/// so [`replace_bundle_assets`] can later remove exactly what was inserted.
pub fn insert_bundle_assets(store: &MemAssetStore, assets: &[(String, Vec<u8>)]) {
    let mut ledger = store.ledger();
    for (path, bytes) in assets {
        store.0.insert_asset(Path::new(path), bytes.clone());
        if !ledger.entries.iter().any(|e| e == path) {
            ledger.entries.push(path.clone());
        }
    }
}

/// Replace the store's tracked contents for a CLEAN document open: remove every asset a prior document
/// inserted (through this API), BUMP the serving generation, then insert this document's `assets`. A
/// newly opened document can thus never render a previous document's mesh bytes for a shared uri: the
/// prior entries are gone from the [`Dir`], and the generation bump makes [`MemAssetStore::asset_path`]
/// hand every subsequent load a path Bevy's asset cache has never seen, so even an asset still pinned
/// alive by the outgoing scene's strong handles cannot be returned for the new document's loads. A
/// plain `.hcdf` opens with an EMPTY `assets` set, which still clears the prior bundle. The ledger lock
/// is held across the whole clear-then-insert so concurrent opens serialize.
pub fn replace_bundle_assets(store: &MemAssetStore, assets: &[(String, Vec<u8>)]) {
    let mut ledger = store.ledger();
    for path in ledger.entries.drain(..) {
        store.0.remove_asset(Path::new(&path));
    }
    ledger.generation += 1;
    for (path, bytes) in assets {
        store.0.insert_asset(Path::new(path), bytes.clone());
        if !ledger.entries.iter().any(|e| e == path) {
            ledger.entries.push(path.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CLEAN open must not let a prior bundle's assets mask a later document's missing asset. Open
    /// bundle A (carrying an A-only uri), then a document that does not carry that uri; the uri must
    /// resolve to missing (cleared), not to A's stale bytes.
    #[test]
    fn clean_open_clears_prior_bundle_assets() {
        let store = MemAssetStore::default();
        let a_uri = "assets/only_in_a.glb";

        // Bundle A populates the store.
        insert_bundle_assets(&store, &[(a_uri.to_string(), vec![1, 2, 3])]);
        assert!(
            store.0.get_asset(Path::new(a_uri)).is_some(),
            "bundle A's asset must be served while A is open"
        );

        // A clean open of a document WITHOUT that uri (e.g. a plain `.hcdf`: empty asset set) clears A.
        replace_bundle_assets(&store, &[]);
        assert!(
            store.0.get_asset(Path::new(a_uri)).is_none(),
            "A's asset must be gone after a clean open, not served stale"
        );

        // A clean open of a DIFFERENT bundle B serves only B's assets, never A's.
        let b_uri = "assets/only_in_b.glb";
        replace_bundle_assets(&store, &[(b_uri.to_string(), vec![9])]);
        assert!(
            store.0.get_asset(Path::new(b_uri)).is_some(),
            "B's asset serves"
        );
        assert!(
            store.0.get_asset(Path::new(a_uri)).is_none(),
            "A's asset must never resurface under B"
        );
    }

    /// Additive inserts (the place/merge flow) accumulate across calls; only a [`replace_bundle_assets`]
    /// clears them. This pins the two APIs' distinct contracts so a future edit cannot silently make the
    /// additive path clear.
    #[test]
    fn additive_insert_accumulates_then_replace_clears_all() {
        let store = MemAssetStore::default();
        insert_bundle_assets(&store, &[("assets/one.glb".to_string(), vec![1])]);
        insert_bundle_assets(&store, &[("assets/two.glb".to_string(), vec![2])]);
        assert!(store.0.get_asset(Path::new("assets/one.glb")).is_some());
        assert!(store.0.get_asset(Path::new("assets/two.glb")).is_some());

        replace_bundle_assets(&store, &[("assets/three.glb".to_string(), vec![3])]);
        assert!(store.0.get_asset(Path::new("assets/one.glb")).is_none());
        assert!(store.0.get_asset(Path::new("assets/two.glb")).is_none());
        assert!(store.0.get_asset(Path::new("assets/three.glb")).is_some());
    }

    /// Load paths must be generation-stamped so Bevy's path-keyed asset cache can never serve a
    /// PREVIOUS clean open's bytes: a tracked uri resolves to a stamped path, each clean-open swap
    /// resolves it to a NEW path, additive inserts keep already-resolved paths stable, and an
    /// untracked uri resolves plain (the native disk-fallback + honest-missing path).
    #[test]
    fn asset_paths_are_stamped_per_clean_open() {
        let store = MemAssetStore::default();
        let uri = "assets/shared.glb";

        assert_eq!(
            store.asset_path(uri),
            uri,
            "an untracked uri resolves to itself"
        );

        replace_bundle_assets(&store, &[(uri.to_string(), vec![1])]);
        let first = store.asset_path(uri);
        assert_ne!(first, uri, "a tracked uri resolves to a stamped path");
        assert!(
            first.ends_with(uri),
            "the stamp only prefixes; the uri stays intact: {first}"
        );

        // Additive inserts share the serving generation: existing paths keep resolving unchanged, and
        // the newly added uri joins under the SAME stamp (dendrite_build's merge/place contract).
        insert_bundle_assets(&store, &[("assets/added.glb".to_string(), vec![2])]);
        assert_eq!(
            store.asset_path(uri),
            first,
            "an additive insert must not move already-resolved paths"
        );
        assert!(store
            .asset_path("assets/added.glb")
            .ends_with("assets/added.glb"));

        // A clean open carrying the SAME uri serves it under a FRESH path (the cache-busting core).
        replace_bundle_assets(&store, &[(uri.to_string(), vec![9])]);
        let second = store.asset_path(uri);
        assert_ne!(second, first, "each clean open must load fresh paths");

        // A clean open WITHOUT the uri drops it back to the plain (missing) resolution.
        replace_bundle_assets(&store, &[]);
        assert_eq!(store.asset_path(uri), uri);
    }

    /// The reader canonicalizes a stamped load path back to the plain-keyed [`Dir`] entry; plain and
    /// lookalike paths pass through untouched.
    #[test]
    fn generation_stamp_strips_for_reader_lookups() {
        assert_eq!(
            strip_generation(Path::new("mem-gen-7/assets/x.stl")),
            Path::new("assets/x.stl")
        );
        assert_eq!(
            strip_generation(Path::new("mem-gen-7//mem/0/x.stl")),
            Path::new("/mem/0/x.stl"),
            "an absolute key's stamp strips to the verbatim absolute uri, leading slash intact"
        );
        assert_eq!(
            strip_generation(Path::new("assets/x.stl")),
            Path::new("assets/x.stl"),
            "a plain uri is untouched"
        );
        assert_eq!(
            strip_generation(Path::new("/mem/0/assets/x.stl")),
            Path::new("/mem/0/assets/x.stl"),
            "a plain absolute uri is untouched"
        );
        assert_eq!(
            strip_generation(Path::new("mem-gen-x/assets/x.stl")),
            Path::new("mem-gen-x/assets/x.stl"),
            "a lookalike segment without a numeric suffix is untouched"
        );
        assert_eq!(
            strip_generation(Path::new("mem-gen-/assets/x.stl")),
            Path::new("mem-gen-/assets/x.stl"),
            "an empty numeric suffix is untouched"
        );
    }

    /// `asset_path` then [`strip_generation`] must round-trip to the EXACT key the [`Dir`] was keyed
    /// under, byte-for-byte, for relative and module-dir-absolute uris alike: the identity the reader
    /// depends on to serve every tracked entry.
    #[test]
    fn stamp_then_strip_round_trips_the_tracked_key() {
        let store = MemAssetStore::default();
        for uri in ["assets/x.stl", "/mem/4/assets/x.stl"] {
            replace_bundle_assets(&store, &[(uri.to_string(), vec![1])]);
            let stamped = store.asset_path(uri);
            assert_ne!(stamped, uri, "a tracked uri stamps");
            assert_eq!(
                strip_generation(Path::new(&stamped)).as_os_str(),
                uri,
                "the strip must recover the inserted key verbatim"
            );
        }
    }
}
