//! Runtime file-open: pick a `.hcdf` / `.hcdfz` from the UI and load it into the read-only
//! [`crate::doc::HcdfDoc`] via [`crate::doc::LoadHcdf`].
//!
//! WHY: the native bin loads its document from a CLI arg at startup; the BROWSER bin has no CLI and the
//! page is sandboxed, so without an in-app picker the wasm viewer opens EMPTY (`StartupArg` is `None` on
//! wasm and `kick_load` does nothing). This module adds the picker plumbing behind a small "Open" button
//! (rendered by [`crate::ui::panels`]) so the viewer is standalone-browser-testable. It is wired by
//! [`crate::HcdvizUiPlugin`] ONLY; an embedder that composes [`crate::HcdvizCorePlugin`] directly (e.g.
//! `dendrite_build`, which owns its own open flow and is the sole writer of its doc) does NOT get it, so
//! there is never a second, conflicting writer to the document.
//!
//! The cross-target seam mirrors `dendrite_build::platform`: a pick delivers `(name, bytes)` into a shared
//! [`OpenChannel`] (`Arc<Mutex<…>>`); the Bevy Update system [`drain_open_channel`] applies them each
//! frame. NATIVE: `rfd::FileDialog` is BLOCKING and the bytes are read via `std::fs` synchronously (the
//! result lands the SAME frame). WASM: `rfd::AsyncFileDialog` runs on the browser microtask queue and the
//! bytes arrive a later frame. Either way the handler is identical (see `open_to_xml`, private).

use crate::doc::LoadHcdf;
use bevy::prelude::*;
use std::sync::{Arc, Mutex};

/// One file the user picked through the Open control: a display NAME (the file name; the full path's
/// string on native) and its raw BYTES (read on native via `std::fs`, on wasm via the async handle: the
/// sandboxed page has no path to re-read later). Classification (`.hcdfz` bundle vs plain `.hcdf`) and
/// loading happen in [`drain_open_channel`] / `open_to_xml` (private).
#[derive(Debug, Clone)]
pub struct PickedDoc {
    /// Native: the picked path as a string. Wasm: the picked file's base name (no directory exists).
    pub name: String,
    /// The file's raw bytes (a plain `.hcdf`'s XML text, or a `.hcdfz` ZIP archive's bytes).
    pub bytes: Vec<u8>,
}

/// The shared queue picked docs land in. The picker (native blocking / wasm async future) locks this and
/// pushes; [`drain_open_channel`] drains it on the main thread each frame. Cloning shares the same inner
/// queue, so the wasm future's `Arc` clone outlives the [`request_open`] call.
#[derive(Resource, Clone, Default)]
pub struct OpenChannel(pub Arc<Mutex<Vec<PickedDoc>>>);

impl OpenChannel {
    /// Take all currently-queued picks (leaving the queue empty); called by [`drain_open_channel`].
    fn drain(&self) -> Vec<PickedDoc> {
        match self.0.lock() {
            Ok(mut q) => std::mem::take(&mut *q),
            // The only other holder is the picker push; recover the data rather than panic so a file pick
            // can never crash the viewer.
            Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
        }
    }

    /// Push a pick onto the queue (native blocking path and the wasm future both use this same lock).
    fn push(&self, doc: PickedDoc) {
        match self.0.lock() {
            Ok(mut q) => q.push(doc),
            Err(poisoned) => poisoned.into_inner().push(doc),
        }
    }
}

// ───────────────────────────────── NATIVE ─────────────────────────────────

/// Start an open-file pick, delivering the chosen file into `channel`.
///
/// NATIVE: BLOCKING modal `rfd::FileDialog` (the standard desktop gesture; on Linux it is the XDG Desktop
/// Portal chooser in a separate process, so it does not contend with bevy/winit). The picked path is read
/// via `std::fs` and pushed the SAME frame, so the drain sees it immediately. A cancel or a read error
/// pushes nothing (the viewer simply keeps its current document).
#[cfg(not(target_arch = "wasm32"))]
pub fn request_open(channel: &OpenChannel) {
    let Some(path) = rfd::FileDialog::new()
        .add_filter("HCDF / bundle", &["hcdf", "hcdfz"])
        .add_filter("All files", &["*"])
        .pick_file()
    else {
        return;
    };
    if let Ok(bytes) = std::fs::read(&path) {
        channel.push(PickedDoc {
            name: path.to_string_lossy().into_owned(),
            bytes,
        });
    }
}

// ────────────────────────────────── WASM ──────────────────────────────────

/// Start an open-file pick, delivering the chosen file into `channel`.
///
/// WASM: `rfd`'s file API is ASYNC-only, so spawn the picker future on the browser microtask queue
/// (`wasm_bindgen_futures::spawn_local`). When the user chooses a file the future reads its bytes and
/// pushes a [`PickedDoc`] onto `channel`, which [`drain_open_channel`] applies on a later frame. A cancel
/// resolves the future with nothing pushed. The `channel` clone (an `Arc`) is moved into the future so the
/// queue outlives this call.
#[cfg(target_arch = "wasm32")]
pub fn request_open(channel: &OpenChannel) {
    let channel = channel.clone();
    wasm_bindgen_futures::spawn_local(async move {
        // "All files" mirrors the native picker: the extension filter is only a hint, and `open_to_xml`
        // already detects a bundle by ZIP magic bytes, so a bundle saved with an odd/missing extension
        // stays pickable instead of being hidden by the `.hcdf,.hcdfz` accept list.
        let dialog = rfd::AsyncFileDialog::new()
            .add_filter("HCDF / bundle", &["hcdf", "hcdfz"])
            .add_filter("All files", &["*"]);
        if let Some(handle) = dialog.pick_file().await {
            let name = handle.file_name();
            let bytes = handle.read().await;
            channel.push(PickedDoc { name, bytes });
        }
    });
}

// ──────────────────────────────── HANDLING ────────────────────────────────

/// Route a native startup-arg file into the load pipeline: if it is a `.hcdfz` bundle (ZIP magic),
/// enqueue its bytes onto `channel` so it flows through the SAME byte pipeline as a user pick
/// ([`open_to_xml`] + [`drain_open_channel`]: bundle extraction, a staged clean-open asset swap on
/// acceptance, meshes served from RAM) and return `true`. Anything else (a plain `.hcdf`, or an unreadable path) returns `false`, so
/// the caller keeps the [`LoadHcdf::Path`] flow, which reads the file as text, resolves native
/// `<include>`s, and falls back to sibling meshes on disk. This exists because the `LoadHcdf::Path` flow
/// reads the file to a `String`, which fails on a bundle's ZIP bytes; a `.hcdfz` passed as the CLI
/// argument used to error at startup. Reads the file once; a read error simply defers to the Path flow,
/// which then surfaces the precise error in the status line.
pub fn enqueue_if_bundle(channel: &OpenChannel, path: &std::path::Path) -> bool {
    match std::fs::read(path) {
        Ok(bytes) if hcdformat::zip_store::is_zip(&bytes) => {
            channel.push(PickedDoc {
                name: path.to_string_lossy().into_owned(),
                bytes,
            });
            true
        }
        _ => false,
    }
}

/// One accepted open payload, including every resource needed by the standalone document-set resolver.
struct OpenPayload {
    xml: String,
    root_key: String,
    documents: Vec<(String, Vec<u8>)>,
    assets: Vec<(String, Vec<u8>)>,
    filesystem_fallback: bool,
}

/// Open a picked file to its root HCDF XML, plus (for a bundle) its extracted `assets/<name>` entries.
///
/// A `.hcdfz` BUNDLE (detected by the `.hcdfz` extension OR by ZIP magic ([`hcdformat::zip_store::is_zip`])
/// so a mis-named-but-valid bundle still opens) is opened IN MEMORY with [`hcdformat::open_bundle_bytes`];
/// its `(assets/<name>, bytes)` entries are returned so [`drain_open_channel`] can stage them for the
/// in-memory asset store's acceptance swap (the bundle is the self-contained, mesh-bearing path). The XML returned is the root entry's
/// RAW bytes, NOT a re-serialization of the parsed doc: the typed parse silently swallows schema-shape
/// errors (a legacy text-content `<pose>` deserializes to an EMPTY pose), so re-serializing would launder
/// exactly what the loader's warn-on-open check (`doc::load_hcdf_system`) exists to surface. (A keep-live
/// bundle's MODULE docs are not raw-validated here; their integrity is covered by the include @sha
/// verification.) A plain `.hcdf` is returned as its own UTF-8 text with NO assets; its meshes render
/// only if their bytes are otherwise reachable (the native filesystem fallback at the launch asset root;
/// nothing in the sandboxed browser, which brings no sibling mesh files).
fn open_to_xml(picked: &PickedDoc) -> Result<OpenPayload, String> {
    let is_bundle = picked.name.to_ascii_lowercase().ends_with(".hcdfz")
        || hcdformat::zip_store::is_zip(&picked.bytes);
    if is_bundle {
        // Enforce the bundle contract (STORED zip, first entry = a parseable root `.hcdf`) and extract
        // the asset tree first, so a malformed archive fails with open_bundle_bytes' precise errors.
        hcdformat::open_bundle_bytes(&picked.bytes)?;
        // Then re-read the archive for the root entry's RAW bytes (see the doc comment). The second
        // read is a one-shot per open; open_bundle_bytes just proved every fallible step below.
        let entries =
            hcdformat::zip_store::read_stored(&picked.bytes).map_err(|e| e.to_string())?;
        let root = entries
            .first()
            .ok_or_else(|| "empty bundle zip (no entries)".to_string())?;
        let xml = String::from_utf8(root.data.clone())
            .map_err(|e| format!("root .hcdf is not utf-8: {e}"))?;
        let root_key = root.name.clone();
        let mut documents = Vec::new();
        let mut assets = Vec::new();
        for entry in entries.into_iter().skip(1) {
            if entry.name == "_keeplive" {
                continue;
            }
            if entry.name.ends_with(".hcdf") || entry.name.ends_with(".xml") {
                documents.push((entry.name, entry.data));
            } else {
                assets.push((entry.name, entry.data));
            }
        }
        Ok(OpenPayload {
            xml,
            root_key,
            documents,
            assets,
            filesystem_fallback: false,
        })
    } else {
        let xml = std::str::from_utf8(&picked.bytes)
            .map_err(|e| format!("source is not utf-8: {e}"))?
            .to_string();
        Ok(OpenPayload {
            xml,
            root_key: picked.name.clone(),
            documents: Vec::new(),
            assets: Vec::new(),
            filesystem_fallback: cfg!(not(target_arch = "wasm32")),
        })
    }
}

/// Drain the [`OpenChannel`] each frame and stage every picked file for the read-only doc loader.
///
/// A pick is a CLEAN document open, but the in-memory asset store is NOT touched here: the `.hcdfz`'s
/// extracted `assets/<name>` entries (empty for a plain `.hcdf`) are STAGED alongside the root XML in a
/// [`LoadHcdf::Open`], and [`crate::doc::load_hcdf_system`] swaps the
/// [`crate::mem_assets::MemAssetStore`] contents only once the XML has parsed. Swapping before the
/// parse verdict stranded a still-rendered previous document without its backing assets whenever the
/// picked file failed to parse; staging makes the failure path a pure no-op for both the document and
/// its assets. On acceptance the swap makes the doc's document-relative mesh URIs resolve from RAM on
/// both targets. (A runtime-opened plain `.hcdf` still resolves meshes through the native filesystem
/// fallback at the LAUNCH asset root, so one picked from a different directory renders geometry-less;
/// bundle it to carry its meshes.)
pub fn drain_open_channel(
    channel: Res<OpenChannel>,
    mut load: MessageWriter<LoadHcdf>,
    mut status: ResMut<crate::doc::SchemaStatus>,
) {
    for picked in channel.drain() {
        match open_to_xml(&picked) {
            Ok(payload) => {
                load.write(LoadHcdf::Open {
                    xml: payload.xml,
                    root_key: payload.root_key,
                    documents: payload.documents,
                    assets: payload.assets,
                    filesystem_fallback: payload.filesystem_fallback,
                });
            }
            Err(e) => status.message = format!("open {}: {e}", picked.name),
        }
    }
}
