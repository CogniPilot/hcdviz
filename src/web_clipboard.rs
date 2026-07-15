//! Browser clipboard bridge: **copy / cut / paste** for egui text fields.
//!
//! Desktop web gets NO clipboard out of the box: bevy_egui's own web support (`manage_clipboard`)
//! listens for the browser's `copy`/`cut`/`paste` `ClipboardEvent`s, but winit `preventDefault()`s every
//! canvas `keydown` (bevy's `Window::prevent_default_event_handling`, default ON), which cancels the
//! browser's clipboard action, so those events never fire while the winit `<canvas>` has focus. (The
//! hidden text-agent input that would make them fire is only focused on MOBILE.) Typing works (winit
//! feeds keydown directly); Ctrl/Cmd+C/X/V silently do nothing.
//!
//! This plugin installs an app-level `keydown` handler for the three shortcuts:
//!  * **Copy / Cut** inject `egui::Event::Copy` / `egui::Event::Cut`, exactly what bevy_egui does on
//!    native for Ctrl+C/X. The copied text then flows OUT through bevy_egui's existing wasm output
//!    path (`OutputCommand::CopyText` → `WebClipboard::set_text` → `navigator.clipboard.writeText`,
//!    permitted: the shortcut keydown is a transient user activation).
//!  * **Paste** reads `navigator.clipboard.readText()` (a user gesture in a secure context: Chrome
//!    shows a one-time permission prompt, Firefox 125+ a per-paste confirmation) and injects
//!    `egui::Event::Paste`. Gated on `defaultPrevented`, i.e. it acts exactly when winit suppressed
//!    the browser's native paste: if focus is ever OUTSIDE the canvas, the real `paste` `ClipboardEvent`
//!    fires and bevy_egui's own listener injects the text, so skipping here avoids a double insert.
//!    Copy/cut are NOT gated: their native events only fire on a DOM selection, which a canvas-only
//!    page never has.
//!
//! Events are injected for the primary context via bevy_egui's `EguiInputEvent`, the SAME hook
//! bevy_egui's own web clipboard uses, scheduled in the same `PreUpdate` slot (before
//! `EguiInputSet::ReadBevyMessages`, so each command is consumed the frame it drains). No-op on native,
//! where bevy_egui's `manage_clipboard`/arboard backend already handles all three shortcuts.
//!
//! Added explicitly by each egui app (hcdviz's `HcdvizUiPlugin` and dendrite_build) rather than by the
//! egui-free `HcdvizCorePlugin`.
use bevy::prelude::*;

/// Wires the browser Ctrl/Cmd+C/X/V → egui clipboard bridge (wasm only; native no-op).
pub struct WebClipboardPlugin;

impl Plugin for WebClipboardPlugin {
    fn build(&self, _app: &mut App) {
        #[cfg(target_arch = "wasm32")]
        {
            _app.init_resource::<wasm::WebClipboardChannel>()
                .add_systems(Startup, wasm::install_shortcut_listener)
                .add_systems(
                    PreUpdate,
                    wasm::drain_web_clipboard.before(bevy_egui::EguiInputSet::ReadBevyMessages),
                );
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use bevy::prelude::*;
    use bevy_egui::input::EguiInputEvent;
    use bevy_egui::{egui, PrimaryEguiContext};
    use std::sync::{Arc, Mutex};
    use wasm_bindgen::{closure::Closure, JsCast};

    /// One captured clipboard shortcut, queued for injection into egui as the matching `egui::Event`.
    pub enum ClipboardCmd {
        /// Ctrl/Cmd+C → `egui::Event::Copy`.
        Copy,
        /// Ctrl/Cmd+X → `egui::Event::Cut`.
        Cut,
        /// Ctrl/Cmd+V (+ the resolved `readText`) → `egui::Event::Paste`.
        Paste(String),
    }

    /// Clipboard commands captured by the shortcut handler, drained into egui each frame. `Arc<Mutex>`
    /// because the async clipboard read resolves on the microtask queue, outside the Bevy schedule.
    #[derive(Resource, Clone, Default)]
    pub struct WebClipboardChannel(Arc<Mutex<Vec<ClipboardCmd>>>);

    /// Queue a command, recovering the (single-threaded wasm, so effectively unreachable) poison case.
    fn push(queue: &Arc<Mutex<Vec<ClipboardCmd>>>, cmd: ClipboardCmd) {
        match queue.lock() {
            Ok(mut q) => q.push(cmd),
            Err(p) => p.into_inner().push(cmd),
        }
    }

    /// Install a `document` `keydown` listener (once, at Startup): Ctrl/Cmd+C/X queue a copy/cut
    /// command directly; Ctrl/Cmd+V kicks off an async `navigator.clipboard.readText()` and queues the
    /// result. The closure is `forget()`-leaked so it lives for the page lifetime (there is no teardown
    /// for a single-page wasm app).
    pub fn install_shortcut_listener(channel: Res<WebClipboardChannel>) {
        let Some(window) = web_sys::window() else {
            return;
        };
        let Some(document) = window.document() else {
            return;
        };
        let queue = channel.0.clone();
        let closure =
            Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
                // Ctrl (Win/Linux) or Cmd (macOS) chords only. `key()` is "c"/"C" etc. depending on Shift.
                if !(e.ctrl_key() || e.meta_key()) {
                    return;
                }
                let key = e.key();
                if key.eq_ignore_ascii_case("c") {
                    push(&queue, ClipboardCmd::Copy);
                } else if key.eq_ignore_ascii_case("x") {
                    push(&queue, ClipboardCmd::Cut);
                } else if key.eq_ignore_ascii_case("v") {
                    // Bridge exactly when winit's canvas handler suppressed the browser's native paste
                    // (it `preventDefault()`s the keydown). If it did NOT (focus outside the canvas), the
                    // real `paste` ClipboardEvent fires and bevy_egui's own listener injects the text, so
                    // skipping here avoids a double insert.
                    if !e.default_prevented() {
                        return;
                    }
                    let Some(win) = web_sys::window() else { return };
                    let promise = win.navigator().clipboard().read_text();
                    let queue = queue.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        if let Ok(val) = wasm_bindgen_futures::JsFuture::from(promise).await {
                            if let Some(text) = val.as_string() {
                                if !text.is_empty() {
                                    push(&queue, ClipboardCmd::Paste(text));
                                }
                            }
                        }
                    });
                }
            });
        let _ =
            document.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref());
        closure.forget();
    }

    /// Drain queued commands into the primary egui context as `egui::Event::{Copy, Cut, Paste}`
    /// (bevy_egui's own clipboard hooks) so the focused `TextEdit` copies/cuts its selection or inserts
    /// the pasted text. Runs before `ReadBevyMessages` consumes the message.
    pub fn drain_web_clipboard(
        channel: Res<WebClipboardChannel>,
        primary: Query<Entity, With<PrimaryEguiContext>>,
        mut writer: MessageWriter<EguiInputEvent>,
    ) {
        let Ok(context) = primary.single() else {
            return;
        };
        let mut queue = match channel.0.lock() {
            Ok(q) => q,
            Err(p) => p.into_inner(),
        };
        for cmd in queue.drain(..) {
            let event = match cmd {
                ClipboardCmd::Copy => egui::Event::Copy,
                ClipboardCmd::Cut => egui::Event::Cut,
                ClipboardCmd::Paste(text) => egui::Event::Paste(text),
            };
            writer.write(EguiInputEvent { context, event });
        }
    }
}
