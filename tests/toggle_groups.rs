//! Headless integration test for visual toggle groups (no GPU, no window).
//!
//! Builds the REAL scene from an HCDF whose visuals carry `toggle="…"` groups (primitive visuals,
//! synchronous, no async glTF), then drives the live `VisualDisplay` visibility system through the
//! `VisualToggleGroups` resource: all groups start visible, hiding a group hides exactly its member
//! visuals (across comps), and re-showing restores them.
use bevy::asset::AssetPlugin;
use bevy::prelude::*;
use hcdviz::display::{AddDisplayExt, DisplayRegistry};
use hcdviz::doc::HcdfDoc;
use hcdviz::pick::{IsolateSelection, IsolateSet, Selected, SelectionOverrides};
use hcdviz::scene::{ScenePlugin, VisualDisplay, VisualItem, VisualToggleGroups};
use hcdviz::schema::Hcdf;
use std::collections::HashMap;
use std::sync::Arc;

// A bare PCB with a two-part case in group "case" and a lid in group "lid", the legacy b3rb-style
// use: hide the enclosure to inspect the board. The bare visual has no group.
const GROUPED: &str = r#"<?xml version="1.0"?>
<hcdf version="1.0" name="grouped" body-frame="FLU" world-frame="ENU">
  <comp name="board">
    <visual name="pcb"><geometry><box size="0.10 0.10 0.01"/></geometry></visual>
    <visual name="case_top" toggle="case"><geometry><box size="0.11 0.11 0.02"/></geometry></visual>
    <visual name="lid" toggle="lid"><geometry><box size="0.11 0.11 0.005"/></geometry></visual>
  </comp>
  <comp name="mount">
    <visual name="case_bottom" toggle="case"><geometry><box size="0.11 0.11 0.02"/></geometry></visual>
  </comp>
  <joint name="j0" type="fixed"><parent comp="board"/><child comp="mount"/></joint>
</hcdf>"#;

/// Visibility of every `VisualItem`, keyed by its spawned `Name` (`visual:{name}`).
fn visual_vis(app: &mut App) -> HashMap<String, Visibility> {
    let world = app.world_mut();
    let mut q = world.query_filtered::<(&Name, &Visibility), With<VisualItem>>();
    q.iter(world)
        .map(|(n, v)| (n.as_str().to_string(), *v))
        .collect()
}

#[test]
fn hiding_a_toggle_group_hides_exactly_its_members() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<IsolateSelection>()
        .init_resource::<IsolateSet>()
        .init_resource::<SelectionOverrides>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin)
        .add_display(VisualDisplay); // registers "Visual" (default-on) + sync_visual_visibility

    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(GROUPED).unwrap()));
    app.update(); // rebuild_on_change builds the scene; refresh_toggle_groups recollects
    app.update(); // flush command queue (visual children materialize)

    // The doc's groups were collected sorted, with nothing hidden: the all-visible default.
    {
        let groups = app.world().resource::<VisualToggleGroups>();
        assert_eq!(groups.groups, vec!["case".to_string(), "lid".to_string()]);
        assert!(groups.hidden.is_empty(), "all groups must start visible");
    }
    let vis = visual_vis(&mut app);
    assert_eq!(vis.len(), 4, "expected 4 spawned visuals, got {vis:?}");
    assert!(
        vis.values().all(|v| *v == Visibility::Inherited),
        "default: everything shown: {vis:?}"
    );

    // Hide "case" → both case visuals (across BOTH comps) hide; pcb + lid stay shown.
    app.world_mut()
        .resource_mut::<VisualToggleGroups>()
        .hidden
        .insert("case".into());
    app.update();
    let vis = visual_vis(&mut app);
    assert_eq!(vis["visual:case_top"], Visibility::Hidden);
    assert_eq!(vis["visual:case_bottom"], Visibility::Hidden);
    assert_eq!(
        vis["visual:pcb"],
        Visibility::Inherited,
        "ungrouped visual must be untouched"
    );
    assert_eq!(
        vis["visual:lid"],
        Visibility::Inherited,
        "other group must be untouched"
    );

    // Show it again → everything returns.
    app.world_mut()
        .resource_mut::<VisualToggleGroups>()
        .hidden
        .remove("case");
    app.update();
    let vis = visual_vis(&mut app);
    assert!(
        vis.values().all(|v| *v == Visibility::Inherited),
        "re-shown: everything back: {vis:?}"
    );
}

#[test]
fn reload_resets_hidden_groups_to_visible() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .add_plugins(AssetPlugin::default())
        .init_asset::<Mesh>()
        .init_asset::<StandardMaterial>()
        .init_resource::<Selected>()
        .init_resource::<IsolateSelection>()
        .init_resource::<IsolateSet>()
        .init_resource::<SelectionOverrides>()
        .init_resource::<DisplayRegistry>()
        .init_resource::<HcdfDoc>()
        .add_plugins(ScenePlugin)
        .add_display(VisualDisplay);

    app.world_mut().resource_mut::<HcdfDoc>().0 =
        Some(Arc::new(Hcdf::from_xml_str(GROUPED).unwrap()));
    app.update();
    app.update();
    app.world_mut()
        .resource_mut::<VisualToggleGroups>()
        .hidden
        .insert("case".into());
    app.update();

    // Republish the doc (same content, any doc change rebuilds): the hide set must clear so the
    // fresh scene starts all-visible, and no stale group names linger.
    let doc = app.world().resource::<HcdfDoc>().0.clone();
    app.world_mut().resource_mut::<HcdfDoc>().0 = doc;
    app.update();
    app.update();
    let groups = app.world().resource::<VisualToggleGroups>();
    assert!(
        groups.hidden.is_empty(),
        "doc reload must reset hidden groups"
    );
    let vis = visual_vis(&mut app);
    assert!(
        vis.values().all(|v| *v == Visibility::Inherited),
        "reload: everything shown: {vis:?}"
    );
}
