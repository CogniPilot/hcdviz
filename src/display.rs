//! rviz-style Display plugin system: a toggleable visualization concern that owns its Bevy systems.
//!
//! Defined once here as hcdviz's public contract; `dendrite_build` adds its own displays via the same
//! `AddDisplayExt` without touching hcdviz internals. Display systems run `.run_if(display_enabled(id))`
//! so toggling is a one-flag flip and disabled displays cost ~nothing.
use bevy::prelude::*;
use std::collections::HashMap;

pub trait Display: Send + Sync + 'static {
    /// Stable id, e.g. "hcdviz.visual", "dendrite_build.discovery".
    fn id(&self) -> &'static str;
    fn label(&self) -> &str {
        self.id()
    }
    fn default_enabled(&self) -> bool {
        true
    }
    /// Register the Bevy systems this display contributes (gate them on `display_enabled(self.id())`).
    fn build(&self, app: &mut App);
}

#[derive(Clone)]
pub struct DisplayEntry {
    pub id: &'static str,
    pub label: String,
}

#[derive(Resource, Default)]
pub struct DisplayRegistry {
    entries: Vec<DisplayEntry>,
    enabled: HashMap<&'static str, bool>,
}

impl DisplayRegistry {
    pub fn register(&mut self, id: &'static str, label: String, enabled: bool) {
        self.entries.push(DisplayEntry { id, label });
        self.enabled.insert(id, enabled);
    }
    pub fn enabled(&self, id: &str) -> bool {
        self.enabled.get(id).copied().unwrap_or(false)
    }
    pub fn set_enabled(&mut self, id: &'static str, on: bool) {
        self.enabled.insert(id, on);
    }
    pub fn entries(&self) -> &[DisplayEntry] {
        &self.entries
    }
}

pub trait AddDisplayExt {
    fn add_display<D: Display>(&mut self, display: D) -> &mut Self;
}

impl AddDisplayExt for App {
    fn add_display<D: Display>(&mut self, display: D) -> &mut Self {
        let (id, label, enabled) = (
            display.id(),
            display.label().to_string(),
            display.default_enabled(),
        );
        if let Some(mut reg) = self.world_mut().get_resource_mut::<DisplayRegistry>() {
            reg.register(id, label, enabled);
        }
        display.build(self);
        self
    }
}

/// Run-condition: is this display currently enabled?
pub fn display_enabled(id: &'static str) -> impl Fn(Res<DisplayRegistry>) -> bool + Clone {
    move |reg: Res<DisplayRegistry>| reg.enabled(id)
}
