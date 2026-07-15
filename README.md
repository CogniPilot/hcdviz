# HCDViz

HCDViz is a read-only, interactive 3D viewer for
[Hardware Configuration Description Format](https://hcdformat.org/) systems. It runs as a native
desktop application or in a web browser.

Use HCDViz to:

- Open `.hcdf` descriptions and self-contained `.hcdfz` bundles.
- Visualize component geometry, frames, joints, sensors, collisions, and inertial properties.
- Articulate joints and inspect kinematic chains and loop closures.
- Explore communication networks, ports, pins, antennas, and other connectivity endpoints.
- Select objects from the scene or hierarchy and inspect their HCDF properties.
- Check schema validation warnings without modifying the source document.

For authoring and editing HCDF systems, use
[Dendrite Build](https://dendrite.hcdformat.org/).

## Live viewer

Open [hcdviz.hcdformat.org](https://hcdviz.hcdformat.org/) and select an `.hcdf` or `.hcdfz` file.
Use an `.hcdfz` bundle when the description depends on mesh assets that must travel with it.

## Build from source

HCDViz requires a stable Rust toolchain and an `hcdformat` source checkout. Point the local Cargo
configuration at that checkout before building:

```bash
cp .cargo/config.toml.example .cargo/config.toml
# Edit the hcdformat path in .cargo/config.toml if your checkout is elsewhere.
```

Build and run the native application:

```bash
cargo build --release --locked
cargo run --release --locked
```

An HCDF file can also be opened at startup:

```bash
cargo run --release --locked -- path/to/system.hcdf
```

For a local browser build:

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk --version 0.21.14 --locked
trunk serve
```

Open `http://localhost:8780/`. To produce an optimized site in `dist/`, run:

```bash
trunk build --release --locked
```
