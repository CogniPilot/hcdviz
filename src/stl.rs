//! Self-contained STL → [`Mesh`] asset loader (no external crates; wasm-clean pure Rust).
//!
//! HCDF collision geometry references STL meshes (`<collision><geometry><mesh uri="…stl">`), but Bevy
//! ships no STL loader. This module parses **both** STL encodings into a Bevy [`Mesh`] and registers
//! itself as an [`AssetLoader`] for the `stl` extension, so `asset_server.load::<Mesh>("assets/x.stl")`
//! resolves document-relative URIs under the asset root exactly like glTF.
//!
//! Encodings (auto-detected):
//!   * **Binary**: 80-byte header, `u32` triangle count, then 50 bytes/triangle (a `[f32; 3]` facet
//!     normal + three `[f32; 3]` vertices + a `u16` attribute byte count).
//!   * **ASCII**: `solid … facet normal nx ny nz / outer loop / vertex … / endloop / endfacet … endsolid`.
//!
//! Output: a `TriangleList` mesh with positions + per-vertex normals. The facet normal is used for all
//! three vertices of a triangle; when it is zero (or unparsable in ASCII) a flat normal is computed
//! from the triangle's geometry. Vertices are emitted unshared (flat shading), matching STL semantics.
use bevy::asset::io::Reader;
use bevy::asset::{AssetLoader, LoadContext, RenderAssetUsages};
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use bevy::prelude::*;

/// hcdviz's STL → [`Mesh`] asset loader. Registered for the `stl` extension.
#[derive(Default, TypePath)]
pub struct StlLoader;

/// Errors surfaced while loading an STL asset. Hand-rolled (no `thiserror` dependency) so the crate
/// stays wasm-clean with no extra deps; `AssetLoader::Error` only needs `Into<BevyError>`, satisfied
/// by any `std::error::Error + Send + Sync`.
#[derive(Debug)]
pub enum StlError {
    /// The underlying reader failed.
    Io(std::io::Error),
    /// The byte stream was too short or otherwise not valid STL.
    Malformed(&'static str),
}

impl std::fmt::Display for StlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StlError::Io(e) => write!(f, "failed to read STL bytes: {e}"),
            StlError::Malformed(m) => write!(f, "malformed STL: {m}"),
        }
    }
}

impl std::error::Error for StlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StlError::Io(e) => Some(e),
            StlError::Malformed(_) => None,
        }
    }
}

impl From<std::io::Error> for StlError {
    fn from(e: std::io::Error) -> Self {
        StlError::Io(e)
    }
}

impl AssetLoader for StlLoader {
    type Asset = Mesh;
    type Settings = ();
    type Error = StlError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Mesh, StlError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        parse_stl(&bytes)
    }

    fn extensions(&self) -> &[&str] {
        &["stl"]
    }
}

/// Register the STL loader so `asset_server.load::<Mesh>("…stl")` works.
pub struct StlPlugin;

impl Plugin for StlPlugin {
    fn build(&self, app: &mut App) {
        app.register_asset_loader(StlLoader);
    }
}

/// Parse STL bytes (binary or ASCII) into a Bevy [`Mesh`].
///
/// Binary is detected structurally (an STL whose byte length equals the binary header + the size
/// implied by its triangle-count field) rather than by the `solid` keyword, since some binary STL
/// files begin with the ASCII word "solid" in their 80-byte header.
pub fn parse_stl(bytes: &[u8]) -> Result<Mesh, StlError> {
    if is_binary_stl(bytes) {
        parse_binary(bytes)
    } else {
        parse_ascii(bytes)
    }
}

/// Heuristic: a file is binary STL iff its length exactly matches `84 + 50 * tri_count`, where
/// `tri_count` is the `u32` at offset 80. Otherwise treat it as ASCII.
fn is_binary_stl(bytes: &[u8]) -> bool {
    if bytes.len() < 84 {
        return false;
    }
    let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let expected = 84 + tri_count * 50;
    bytes.len() == expected
}

/// Build a `TriangleList` mesh from flat (unshared) vertex positions + normals.
fn build_mesh(positions: Vec<[f32; 3]>, normals: Vec<[f32; 3]>) -> Mesh {
    let count = positions.len() as u32;
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32((0..count).collect()))
}

/// Per-vertex normal for one triangle: the facet normal if non-degenerate, else a flat normal from
/// the triangle geometry (right-hand winding), else +Z as a last resort.
fn vertex_normal(facet: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let n = Vec3::from(facet);
    if n.length_squared() > 1e-12 {
        return n.normalize().into();
    }
    let (va, vb, vc) = (Vec3::from(a), Vec3::from(b), Vec3::from(c));
    let computed = (vb - va).cross(vc - va);
    if computed.length_squared() > 1e-12 {
        computed.normalize().into()
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn parse_binary(bytes: &[u8]) -> Result<Mesh, StlError> {
    if bytes.len() < 84 {
        return Err(StlError::Malformed("binary STL shorter than header"));
    }
    let tri_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
    let mut positions = Vec::with_capacity(tri_count * 3);
    let mut normals = Vec::with_capacity(tri_count * 3);

    let read_f32 = |off: usize| -> f32 {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let read_vec3 =
        |off: usize| -> [f32; 3] { [read_f32(off), read_f32(off + 4), read_f32(off + 8)] };

    for t in 0..tri_count {
        let base = 84 + t * 50;
        if base + 50 > bytes.len() {
            return Err(StlError::Malformed("binary STL truncated mid-triangle"));
        }
        let facet = read_vec3(base);
        let a = read_vec3(base + 12);
        let b = read_vec3(base + 24);
        let c = read_vec3(base + 36);
        let n = vertex_normal(facet, a, b, c);
        positions.extend_from_slice(&[a, b, c]);
        normals.extend_from_slice(&[n, n, n]);
    }
    Ok(build_mesh(positions, normals))
}

fn parse_ascii(bytes: &[u8]) -> Result<Mesh, StlError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| StlError::Malformed("STL is neither valid binary nor UTF-8 ASCII"))?;
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();

    let mut facet = [0.0f32; 3];
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(3);

    for line in text.lines() {
        let mut tok = line.split_whitespace();
        match tok.next() {
            Some("facet") => {
                // `facet normal nx ny nz`
                let _ = tok.next(); // "normal"
                facet = parse_three(&mut tok).unwrap_or([0.0; 3]);
                verts.clear();
            }
            Some("vertex") => {
                if let Some(v) = parse_three(&mut tok) {
                    verts.push(v);
                }
            }
            Some("endfacet") => {
                // Emit the (first) triangle of this facet. STL facets are triangles; if a loop ever
                // listed >3 vertices we fan-triangulate defensively.
                if verts.len() >= 3 {
                    for i in 1..verts.len() - 1 {
                        let (a, b, c) = (verts[0], verts[i], verts[i + 1]);
                        let n = vertex_normal(facet, a, b, c);
                        positions.extend_from_slice(&[a, b, c]);
                        normals.extend_from_slice(&[n, n, n]);
                    }
                }
                verts.clear();
            }
            _ => {}
        }
    }

    if positions.is_empty() {
        return Err(StlError::Malformed("ASCII STL contained no triangles"));
    }
    Ok(build_mesh(positions, normals))
}

/// Parse the next three whitespace-separated `f32`s from a token iterator.
fn parse_three<'a>(tok: &mut impl Iterator<Item = &'a str>) -> Option<[f32; 3]> {
    let x = tok.next()?.parse().ok()?;
    let y = tok.next()?.parse().ok()?;
    let z = tok.next()?.parse().ok()?;
    Some([x, y, z])
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::VertexAttributeValues;

    /// One-triangle binary STL: 80-byte header, count=1, facet normal +Z, three unit-ish verts.
    fn one_triangle_binary() -> Vec<u8> {
        let mut b = vec![0u8; 80]; // header (zeros)
        b.extend_from_slice(&1u32.to_le_bytes()); // triangle count
                                                  // facet normal (0,0,1)
        b.extend_from_slice(&0f32.to_le_bytes());
        b.extend_from_slice(&0f32.to_le_bytes());
        b.extend_from_slice(&1f32.to_le_bytes());
        // v0 (0,0,0)
        for _ in 0..3 {
            b.extend_from_slice(&0f32.to_le_bytes());
        }
        // v1 (1,0,0)
        b.extend_from_slice(&1f32.to_le_bytes());
        b.extend_from_slice(&0f32.to_le_bytes());
        b.extend_from_slice(&0f32.to_le_bytes());
        // v2 (0,1,0)
        b.extend_from_slice(&0f32.to_le_bytes());
        b.extend_from_slice(&1f32.to_le_bytes());
        b.extend_from_slice(&0f32.to_le_bytes());
        // attribute byte count
        b.extend_from_slice(&0u16.to_le_bytes());
        b
    }

    fn position_count(mesh: &Mesh) -> usize {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(v)) => v.len(),
            _ => 0,
        }
    }

    #[test]
    fn parses_tiny_binary_stl_one_triangle() {
        let bytes = one_triangle_binary();
        assert!(
            is_binary_stl(&bytes),
            "should detect binary by exact length"
        );
        let mesh = parse_stl(&bytes).expect("binary STL parses");
        assert_eq!(mesh.primitive_topology(), PrimitiveTopology::TriangleList);
        // One triangle → exactly 3 (unshared) vertices.
        assert_eq!(position_count(&mesh), 3);
        // Normals present, one per vertex, equal to the facet normal (+Z).
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(n)) => {
                assert_eq!(n.len(), 3);
                for normal in n {
                    assert!(
                        (normal[2] - 1.0).abs() < 1e-6,
                        "facet +Z normal expected, got {normal:?}"
                    );
                }
            }
            _ => panic!("expected per-vertex Float32x3 normals"),
        }
    }

    #[test]
    fn parses_ascii_stl_one_triangle_with_zero_normal_falls_back_to_flat() {
        // facet normal is 0 0 0 → flat normal computed from CCW winding in the XY plane → +Z.
        let ascii = "solid t\n\
            facet normal 0 0 0\n\
            outer loop\n\
            vertex 0 0 0\n\
            vertex 1 0 0\n\
            vertex 0 1 0\n\
            endloop\n\
            endfacet\n\
            endsolid t\n";
        let mesh = parse_stl(ascii.as_bytes()).expect("ASCII STL parses");
        assert_eq!(position_count(&mesh), 3);
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(n)) => {
                assert!(
                    (n[0][2] - 1.0).abs() < 1e-6,
                    "computed flat normal should be +Z, got {:?}",
                    n[0]
                );
            }
            _ => panic!("expected normals"),
        }
    }
}
