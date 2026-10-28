// Actual Vulkan GPU optical sampling: one work item per virtual facet.
// Shader reads head-mounted offscreen color texture; source R/G/B may be
// linearized by the hardware for sRGB format. Temporal phototransduction is
// deliberately separate in engine/src/retina.rs.
@group(0) @binding(0) var eye_image: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> projected_facets: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> sampled_rgb: array<vec4<f32>>;

@compute @workgroup_size(64)
fn sample_facets(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= 1442u) {
        return;
    }
    let dimensions = textureDimensions(eye_image, 0);
    let half = i32(dimensions.x) / 2;
    let base = vec2<i32>(projected_facets[i].xy + vec2<f32>(0.5, 0.5));
    let x_min = select(0, half, i >= 721u);
    let x_max = x_min + half - 1;
    var color = vec4<f32>(0.0);
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let x = clamp(base.x + dx, x_min, x_max);
            let y = clamp(base.y + dy, 0, i32(dimensions.y) - 1);
            let wx = select(1.0, 2.0, dx == 0);
            let wy = select(1.0, 2.0, dy == 0);
            color = color + textureLoad(eye_image, vec2<i32>(x, y), 0) * (wx * wy);
        }
    }
    sampled_rgb[i] = color / 16.0;
}
