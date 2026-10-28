struct Draw {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0)
var<storage, read> instances: array<Draw>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) tint: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VsOut {
    // Unit cube expanded from 12 triangles; geometry generated on the GPU.
    let corners = array<vec3<f32>, 8>(
        vec3<f32>(-0.5, -0.5, -0.5),
        vec3<f32>( 0.5, -0.5, -0.5),
        vec3<f32>( 0.5,  0.5, -0.5),
        vec3<f32>(-0.5,  0.5, -0.5),
        vec3<f32>(-0.5, -0.5,  0.5),
        vec3<f32>( 0.5, -0.5,  0.5),
        vec3<f32>( 0.5,  0.5,  0.5),
        vec3<f32>(-0.5,  0.5,  0.5)
    );
    let triangles = array<u32, 36>(
        4u, 5u, 6u, 4u, 6u, 7u,
        1u, 0u, 3u, 1u, 3u, 2u,
        0u, 4u, 7u, 0u, 7u, 3u,
        5u, 1u, 2u, 5u, 2u, 6u,
        3u, 7u, 6u, 3u, 6u, 2u,
        0u, 1u, 5u, 0u, 5u, 4u
    );
    let draw = instances[instance];
    let face = vertex / 6u;
    let shade = array<f32, 6>(1.0, 0.68, 0.79, 0.89, 1.1, 0.62);
    var out: VsOut;
    out.clip = draw.mvp * vec4<f32>(corners[triangles[vertex]], 1.0);
    out.tint = vec4<f32>(draw.color.rgb * shade[face], 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.tint;
}
