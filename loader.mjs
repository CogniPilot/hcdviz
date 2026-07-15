// Trunk custom initializer (rel="rust" data-initializer): taps the wasm-bindgen
// init pipeline so the boot overlay in index.html can show real download/instantiate
// progress instead of a blank #1b1b1f page. Self-contained: no imports, no external
// assets. Trunk copies this module into dist/ and wires the import automatically.
//
// API (trunk >= 0.19, verified against trunk-rs/trunk guide + examples/initializer):
//   default export returns { onStart, onProgress({current,total}), onComplete,
//   onSuccess(wasm), onFailure(error) }. onProgress.current/total are WASM bytes;
//   total is 0 when the server sends no content-length.
export default function loader() {
  const overlay = document.getElementById("boot-overlay");
  const bar = document.getElementById("boot-bar");
  const track = document.getElementById("boot-track");
  const status = document.getElementById("boot-status");
  const errBox = document.getElementById("boot-error");
  let done = false; // set once we either faded out or showed the error card

  const mb = (bytes) => (bytes / (1024 * 1024)).toFixed(1);

  const onError = (e) => fail(e && (e.error || e.message) || e);
  const onRejection = (e) => fail(e && e.reason || e);
  const finish = () => {
    if (done) return;
    done = true;
    removeListeners();
    if (!overlay) return;
    overlay.classList.add("boot-hide");
    setTimeout(() => { if (overlay && overlay.parentNode) overlay.remove(); }, 450);
  };
  function addListeners() {
    // Catch init panics (console_error_panic_hook logs to console but the page
    // would otherwise stay blank) that surface before the app starts.
    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onRejection);
    // Trunk dispatches this once the app's start() has run; the definitive
    // "app is live" signal, so fade the overlay and stop watching for errors.
    window.addEventListener("TrunkApplicationStarted", finish);
  }
  function removeListeners() {
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onRejection);
    window.removeEventListener("TrunkApplicationStarted", finish);
  }
  function fail(error) {
    if (done) return;
    done = true;
    removeListeners();
    if (!overlay) return;
    overlay.classList.add("boot-failed");
    if (status) status.textContent = "failed to start; open the console (F12) for details";
    if (errBox) {
      let msg = "";
      try { msg = (error && (error.stack || error.message)) || String(error); }
      catch (_) { msg = "unknown error"; }
      errBox.textContent = msg;
    }
  }

  return {
    onStart: () => {
      addListeners();
      if (status) status.textContent = "downloading…";
    },
    onProgress: ({ current, total }) => {
      if (done) return;
      if (total) {
        const pct = Math.min(100, (current / total) * 100);
        if (bar) bar.style.width = pct.toFixed(1) + "%";
        if (current >= total) {
          if (status) status.textContent = "instantiating…";
          if (track) track.classList.add("boot-indeterminate");
        } else {
          if (track) track.classList.remove("boot-indeterminate");
          if (status) status.textContent = `downloading… ${mb(current)} / ${mb(total)} MB`;
        }
      } else {
        // No content-length: can't show a ratio, go indeterminate.
        if (track) track.classList.add("boot-indeterminate");
        if (status) status.textContent = `downloading… ${mb(current)} MB`;
      }
    },
    onComplete: () => {},
    onSuccess: () => {
      if (done) return;
      if (bar) bar.style.width = "100%";
      if (track) track.classList.add("boot-indeterminate");
      if (status) status.textContent = "starting…";
      // Overlay removal waits for TrunkApplicationStarted (see finish) so error
      // listeners stay armed through the final startup step.
    },
    onFailure: (error) => fail(error),
  };
}
