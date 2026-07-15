//! Headless tests for the document load status ([`hcdviz::doc::load_hcdf_system`]): no GPU / no window.
use bevy::prelude::*;
use hcdviz::doc::{load_hcdf_system, HcdfDoc, LoadHcdf, SchemaStatus};

/// A minimal app running just the loader: send a `LoadHcdf::Xml` with [`open`], then read the resulting
/// `HcdfDoc` / `SchemaStatus`.
fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .init_resource::<HcdfDoc>()
        .init_resource::<SchemaStatus>()
        .add_message::<LoadHcdf>()
        .add_systems(Update, load_hcdf_system);
    app
}

fn open(app: &mut App, xml: &str) {
    app.world_mut()
        .write_message(LoadHcdf::Xml(xml.to_string()));
    app.update();
}

const VALID_A: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="doc-a" body-frame="FLU" world-frame="ENU">
  <comp name="c"/>
</hcdf>"#;

// Unterminated element: well-formed enough to reach the typed parse, which then errors.
const MALFORMED_B: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="doc-b" body-frame="FLU" world-frame="ENU"><comp"#;

// Two joints both named "shoulder": they alias the name-keyed slider map (one slider drives both).
const DUP_JOINTS: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="dupjoints" body-frame="FLU" world-frame="ENU">
  <comp name="base"/>
  <comp name="a"/>
  <comp name="b"/>
  <joint name="shoulder" type="revolute"><parent comp="base"/><child comp="a"/></joint>
  <joint name="shoulder" type="revolute"><parent comp="base"/><child comp="b"/></joint>
</hcdf>"#;

/// A failed open must not relabel the still-rendered previous document. After a valid doc A loads, a
/// malformed doc B fails to parse; A stays shown, keeps its OWN warn-on-open (never B's), and the status
/// line reports the failure and that the previous document is still shown.
#[test]
fn failed_open_preserves_shown_doc_and_its_warning() {
    let mut app = build_app();
    open(&mut app, VALID_A);
    assert_eq!(
        app.world().resource::<HcdfDoc>().0.as_ref().unwrap().name,
        "doc-a"
    );
    assert!(app
        .world()
        .resource::<SchemaStatus>()
        .message
        .contains("loaded"));

    // Simulate that A surfaced an amber warn-on-open which must survive a later failed open: the bug was
    // that a failed parse overwrote this with the REJECTED file's warning while A stayed on screen.
    app.world_mut().resource_mut::<SchemaStatus>().open_warning =
        Some("a-specific-warning".to_string());

    open(&mut app, MALFORMED_B);

    let doc = app.world().resource::<HcdfDoc>();
    let status = app.world().resource::<SchemaStatus>();
    // Doc A is still the shown document...
    assert_eq!(
        doc.0
            .as_ref()
            .expect("A must stay shown after a failed open")
            .name,
        "doc-a"
    );
    // ...wearing its OWN warning, not the rejected file's...
    assert_eq!(
        status.open_warning.as_deref(),
        Some("a-specific-warning"),
        "the shown doc must keep its own warn-on-open, not the rejected file's"
    );
    // ...and the status line names the failure and that the previous document is still on screen.
    assert!(
        status.message.contains("parse error"),
        "message: {}",
        status.message
    );
    assert!(
        status
            .message
            .contains("previously loaded document is still shown"),
        "message: {}",
        status.message
    );
}

/// A document with duplicate joint names still loads, but the load line names the duplicated joint and
/// warns that one slider drives all joints sharing the name (the name keying is not changed; it is the
/// external-writer design seam).
#[test]
fn duplicate_joint_names_surface_a_slider_alias_warning() {
    let mut app = build_app();
    open(&mut app, DUP_JOINTS);
    let status = app.world().resource::<SchemaStatus>();
    assert!(
        status.message.contains("loaded"),
        "the doc still loads: {}",
        status.message
    );
    assert!(
        status.message.contains("duplicate joint name"),
        "the load line must warn about duplicate joint names: {}",
        status.message
    );
    assert!(
        status.message.contains("shoulder"),
        "the warning must name the duplicated joint: {}",
        status.message
    );
}
