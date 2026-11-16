// Texture WGSL shader: samples a texture and multiplies by vertex color
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@group(0) @binding(0)
var diff_sampler: sampler;
@group(0) @binding(1)
var diff_texture: texture_2d<f32>;

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;
    out.position = position;
    out.color = color;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(diff_texture, diff_sampler, in.uv);
    return tex_color * in.color;
}
