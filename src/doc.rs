//! The loaded HCDF document as a read-only resource, schema verification, and the load pipeline.
//!
//! hcdviz never mutates `HcdfDoc`; rendering reacts to it changing. `dendrite_build` is the only
//! writer: it edits its own doc and publishes a fresh `Arc<Hcdf>` here, which hcdviz re-renders.
use bevy::prelude::*;
use hcdformat::Hcdf;
use std::path::PathBuf;
use std::sync::Arc;

/// The current official HCDF document (read-only for rendering systems).
#[derive(Resource, Default)]
pub struct HcdfDoc(pub Option<Arc<Hcdf>>);

/// Schema-pin + load status, surfaced in the UI status bar.
#[derive(Resource, Default)]
pub struct SchemaStatus {
    pub pin_ok: bool,
    pub xsd_sha: String,
    pub message: String,
    /// Warn-on-open: the RAW bytes of the currently shown document failed XSD validation (`None` ⇒ valid,
    /// dismissed, or nothing opened yet). Set by [`load_hcdf_system`] only on a SUCCESSFUL open (a failed
    /// parse leaves the previous doc shown and must keep ITS warning, never the rejected file's), never on
    /// the re-publish path (an embedder writing [`HcdfDoc`] directly), and cleared by a clean open or by
    /// the dismiss button in [`crate::ui::panels`]. Non-blocking: the doc still loads what parsed.
    pub open_warning: Option<String>,
}

/// Request to load an HCDF document.
#[derive(Message)]
pub enum LoadHcdf {
    /// A CLEAN document open from a file path (the native startup arg). Reads the file as text and
    /// anchors relative `<include>`s on its directory. Like `Open`, acceptance CLEARS the in-memory
    /// asset store: a path-opened document's meshes resolve through the native filesystem fallback
    /// and are never store-tracked, so anything the store still tracks belongs to a PRIOR document
    /// and must not stay resolvable under the new one. Staged as an EMPTY asset set, so the clear is
    /// as transactional as an `Open`'s swap (a failed read or parse leaves the store untouched).
    Path(PathBuf),
    /// Re-publish a document WITHOUT touching the in-memory asset store: the embedder seam. A caller
    /// that manages the store itself (layering [`crate::mem_assets::insert_bundle_assets`] and then
    /// republishing edited XML) must keep its assets across every publish, so this variant NEVER
    /// clears, by contract; a clean open belongs on `Open` (or `Path`) instead.
    Xml(String),
    /// A CLEAN document open staged with the in-memory asset set that must back it: the root XML plus
    /// the `(document-relative uri, bytes)` entries extracted from its bundle (EMPTY for a plain
    /// `.hcdf`, which still clears the prior document's assets on acceptance). The asset-store swap is
    /// transactional with document ACCEPTANCE: [`load_hcdf_system`] replaces the
    /// [`crate::mem_assets::MemAssetStore`] contents with `assets` only after the XML parses, so a
    /// failed open leaves the previous document rendering with its backing assets untouched. Emitted by
    /// the picker/startup byte pipeline ([`crate::open::drain_open_channel`]); `Xml` stays the
    /// no-store-effect load for callers that manage assets themselves.
    Open {
        xml: String,
        /// Stable root resource key used to resolve document-relative dependencies.
        root_key: String,
        /// In-memory dependency documents keyed exactly as resolved from `root_key`.
        documents: Vec<(String, Vec<u8>)>,
        assets: Vec<(String, Vec<u8>)>,
        /// Native plain-file opens may resolve sibling resources from the filesystem. Bundles and
        /// browser opens remain confined to their in-memory resource maps.
        filesystem_fallback: bool,
    },
}

/// Verify the embedded, sha-pinned official schema once at startup.
pub fn verify_schema(mut status: ResMut<SchemaStatus>) {
    match hcdformat::schema::verify_embedded_schema() {
        Ok(()) => {
            status.pin_ok = true;
            status.xsd_sha = hcdformat::schema::HCDF_XSD_SHA256_1_0.to_string();
            status.message = "ready: load an HCDF".into();
        }
        Err(e) => {
            status.pin_ok = false;
            status.message = format!("schema pin FAILED: {e}");
        }
    }
}

/// Parse requested documents into the read-only `HcdfDoc` resource.
///
/// The [`crate::mem_assets::MemAssetStore`] is optional so worlds that never registered it (embedders
/// with their own asset flow, headless tests) still load; a staged asset set (an [`LoadHcdf::Open`]'s
/// extracted entries, a [`LoadHcdf::Path`]'s empty clean-open set) is the only thing that touches it,
/// and only on ACCEPTANCE (see the parse match below).
pub fn load_hcdf_system(
    mut events: MessageReader<LoadHcdf>,
    mut doc: ResMut<HcdfDoc>,
    mut status: ResMut<SchemaStatus>,
    mem_store: Option<Res<crate::mem_assets::MemAssetStore>>,
    mut standalone_staging: Option<ResMut<crate::standalone_connectivity::StandaloneLoadStaging>>,
    mut standalone_accepted: Option<
        ResMut<crate::standalone_connectivity::AcceptedStandaloneProjection>,
    >,
) {
    // A clean open with nothing to serve from RAM: staging it still clears a prior document's
    // tracked entries on acceptance (the `Path` contract; see [`LoadHcdf`]).
    const NO_ASSETS: &[(String, Vec<u8>)] = &[];
    for ev in events.read() {
        let standalone = standalone_staging
            .as_deref_mut()
            .and_then(crate::standalone_connectivity::StandaloneLoadStaging::pop);
        // `base` is the directory that `<include>` relative uris resolve against: a Path load anchors
        // on the file's own directory; an Xml/Open load has no source directory (relative includes
        // can't resolve, handled in `flatten_includes`). `staged` carries the CLEAN-OPEN asset set (as
        // `Option<&[(uri, bytes)]>`, an Open's extracted entries or a Path's empty set) until the
        // parse verdict decides whether it may replace the store's contents; `None` marks the one
        // variant that must NOT touch the store (`Xml`, the embedder republish seam).
        let (xml, base, staged) = match ev {
            LoadHcdf::Path(p) => match std::fs::read_to_string(p) {
                Ok(s) => (s, p.parent().map(|d| d.to_path_buf()), Some(NO_ASSETS)),
                Err(e) => {
                    status.message = format!("read {}: {e}", p.display());
                    continue;
                }
            },
            LoadHcdf::Xml(s) => (s.clone(), None, None),
            LoadHcdf::Open { xml, assets, .. } => (xml.clone(), None, Some(assets.as_slice())),
        };
        // Warn-on-open: validate the RAW bytes ONCE per open, BEFORE the typed parse: the parse
        // silently swallows schema-shape errors (a legacy text-content `<pose>` deserializes to an
        // EMPTY pose), so only the raw string can reveal them. Every open funnels through a `LoadHcdf`
        // message (startup Path, picked/bundle Open, pasted Xml) and nothing else does; an embedder's
        // re-publish writes `HcdfDoc` directly, so this is exactly one-shot per open, never on
        // re-publish/re-flatten. Non-blocking: the doc still loads below with whatever parsed. The result
        // is stamped into the status only on a SUCCESSFUL parse (see the Err arm).
        let new_warning = open_warning(&xml);
        match Hcdf::from_xml_str(&xml) {
            Ok(mut h) => {
                // The document is ACCEPTED: only now may a staged asset set replace the store's
                // contents. Swapping any earlier (in the open flow, before the parse verdict) would
                // strand a still-rendered previous document without its backing assets when the new
                // file fails to parse. Swapping here, in the same system write as the doc, also
                // guarantees the scene rebuild triggered by this doc change resolves the NEW set.
                if let (Some(assets), Some(store)) = (staged, mem_store.as_deref()) {
                    crate::mem_assets::replace_bundle_assets(store, assets);
                } else if staged.is_some_and(|assets| !assets.is_empty()) {
                    // Only a NON-EMPTY staged set is worth a warning: an empty clean-open set with no
                    // store registered has nothing to clear and nothing that could fail to resolve.
                    warn!(
                        "staged open assets dropped: no MemAssetStore is registered \
                         (register_mem_asset_source was not called), meshes will not resolve"
                    );
                }
                let suffix = match standalone.as_ref() {
                    Some(crate::standalone_connectivity::StagedStandaloneLoad::Projection(
                        projected,
                    )) if projected.is_ok() => {
                        let projected = projected.as_ref().as_ref().expect("checked projection");
                        h = projected.flattened().clone();
                        crate::standalone_connectivity::projection_status_suffix(projected)
                    }
                    _ => flatten_includes(&mut h, base.as_deref()),
                };
                let alias_note = duplicate_joint_note(&h);
                status.open_warning = new_warning;
                status.message = format!(
                    "loaded \"{}\" v{}: {} comp(s){}{}",
                    h.name,
                    h.version,
                    h.comp.len(),
                    suffix,
                    alias_note,
                );
                doc.0 = Some(Arc::new(h));
                if let (
                    Some(accepted),
                    Some(crate::standalone_connectivity::StagedStandaloneLoad::Projection(result)),
                ) = (standalone_accepted.as_deref_mut(), standalone)
                {
                    accepted.accept(*result);
                }
            }
            // A failed parse leaves the previously loaded doc rendered, and any STAGED asset set is
            // simply dropped: the store still holds (and serves) the shown document's assets. Do NOT
            // stamp the rejected file's warn-on-open onto it either: keep the shown doc's own
            // `open_warning` untouched, and note in the status line that the previous document is
            // still on screen so the failure cannot mislabel it.
            Err(e) => {
                let shown = if doc.0.is_some() {
                    "; the previously loaded document is still shown"
                } else {
                    ""
                };
                status.message = format!("parse error: {e}{shown}");
            }
        }
    }
}

/// Run the one-shot warn-on-open schema check on a document's RAW xml, returning the warning line for
/// [`SchemaStatus::open_warning`] (`None` ⇒ schema-valid).
///
/// Uses [`hcdformat::validate_xsd`] (pure-Rust, wasm-clean) so the check runs identically in the
/// browser. The line leads with the issue COUNT and quotes the FIRST issue (with its line:col), enough
/// to identify the offending element without flooding the panel when a legacy doc trips dozens of sites.
fn open_warning(raw_xml: &str) -> Option<String> {
    let issues = hcdformat::validate_xsd(raw_xml);
    let first = issues.first()?;
    Some(format!(
        "document is not schema-valid ({} XSD issue(s); first: {}); showing what parsed",
        issues.len(),
        first.message
    ))
}

/// A load-line suffix naming joints whose name is used more than once (empty when names are unique).
///
/// Joint sliders, and any future external writer, key [`crate::joints::JointPositions`] by joint NAME, so
/// two joints sharing a name ALIAS: one slider drives every joint with that name. Duplicate joint names
/// also violate the schema `jointKey`, so the amber warn-on-open already fires; this spells out the slider
/// consequence, which that generic line does not. The name keying is the deliberate external-writer design
/// seam, so this only WARNS: it never re-keys the map.
fn duplicate_joint_note(h: &Hcdf) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut dups: Vec<&str> = Vec::new();
    for name in h
        .joint
        .iter()
        .filter_map(|j| j.name.as_deref())
        .filter(|n| !n.is_empty())
    {
        if !seen.insert(name) && !dups.contains(&name) {
            dups.push(name);
        }
    }
    if dups.is_empty() {
        String::new()
    } else {
        format!(
            " [duplicate joint name(s) {}: one slider drives all joints sharing a name]",
            dups.join(", ")
        )
    }
}

/// Flatten `<include>` elements in place, returning a note suffix the caller folds into the status
/// line (empty when there are no includes).
///
/// Native: resolves includes from the filesystem relative to `base` (the document's own directory).
/// An `Xml` load has no `base`, so relative includes cannot resolve; they are left in place with a
/// surfaced note rather than crashing. WASM: the filesystem is unavailable, so includes are left as-is
/// (the doc renders whatever it defines inline); this keeps the wasm target building and non-fatal.
fn flatten_includes(h: &mut Hcdf, base: Option<&std::path::Path>) -> String {
    if h.include.is_empty() {
        return String::new();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let Some(base) = base else {
            return format!(
                " [{} include(s) unresolved: no source dir]",
                h.include.len()
            );
        };
        match hcdformat::flatten(h, base) {
            Ok(notes) => {
                let kept = h.include.len();
                if kept > 0 {
                    format!(" [{kept} include(s) unresolved]")
                } else if !notes.is_empty() {
                    format!(" [+{} included comp set(s)]", notes.len())
                } else {
                    String::new()
                }
            }
            Err(e) => format!(" [include error: {e}]"),
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = base;
        format!(" [{} include(s) not resolved on web]", h.include.len())
    }
}
