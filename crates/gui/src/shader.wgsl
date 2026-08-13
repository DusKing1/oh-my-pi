// omp-gui render pipelines: instanced SDF rounded rects + atlas glyphs.
// Two atlas textures share one bind group: R8 coverage for outlined text,
// RGBA for color bitmap emoji. Everything is alpha-premultiplied so the
// window surface can composite translucently over the desktop.

struct Globals {
    screen: vec2<f32>,      // target size, physical px
    atlas: vec2<f32>,       // mask atlas size (color atlas matches)
};

@group(0) @binding(0) var<uniform> G: Globals;

fn to_clip(p: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(p / G.screen * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

// ---------------------------------------------------------------- rects ----

struct RectIn {
    @location(0) pos: vec2<f32>,          // top-left, physical px
    @location(1) size: vec2<f32>,
    @location(2) color: vec4<f32>,        // straight-alpha fill
    @location(3) params: vec4<f32>,       // radius, softness, border_w, dash period
    @location(4) border_color: vec4<f32>,
    @location(5) color2: vec4<f32>,
    @location(6) grad: vec4<f32>,         // direction, minimum, inverse span
};

struct RectOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) params: vec4<f32>,
    @location(3) border_color: vec4<f32>,
    @location(4) half_size: vec2<f32>,
    @location(5) color2: vec4<f32>,
    @location(6) grad: vec4<f32>,
};

@vertex
fn vs_rect(@builtin(vertex_index) vi: u32, r: RectIn) -> RectOut {
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u));
    // Expand the quad past the rect so soft shadows have room to fall off.
    let pad = r.params.y * 3.0 + 1.0;
    let half = r.size * 0.5;
    let local = corner * (r.size + 2.0 * pad) - pad - half;
    let world = r.pos + half + local;

    var o: RectOut;
    o.clip = to_clip(world);
    o.local = local;
    o.color = r.color;
    o.params = r.params;
    o.border_color = r.border_color;
    o.color2 = r.color2;
    o.grad = r.grad;
    o.half_size = half;
    return o;
}

fn sd_round_box(p: vec2<f32>, b: vec2<f32>, r: f32) -> f32 {
    let q = abs(p) - b + vec2<f32>(r, r);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_rect(i: RectOut) -> @location(0) vec4<f32> {
    let radius = min(i.params.x, min(i.half_size.x, i.half_size.y));
    let d = sd_round_box(i.local, i.half_size, radius);
    let soft = max(i.params.y, 0.5);
    let alpha = 1.0 - smoothstep(-soft, soft, d);

    var fill_color = i.color;
    var border_color = i.border_color;
    let bw = i.params.z;
    if (i.grad.w != 0.0) {
        let t = clamp((dot(i.local + i.half_size, i.grad.xy) - i.grad.z) * i.grad.w, 0.0, 1.0);
        if (bw > 0.0) {
            border_color = mix(border_color, i.color2, t);
        } else {
            fill_color = mix(fill_color, i.color2, t);
        }
    }
    var color = fill_color;
    if (bw > 0.0) {
        if (i.params.w > 0.0) {
            let t = select(i.local.y, i.local.x, abs(i.local.x) < abs(i.local.y));
            if (fract(t / i.params.w) >= 0.5) {
                border_color = vec4<f32>(border_color.rgb, 0.0);
            }
        }
        // Border band occupies d in [-bw, 0]; fill sits deeper inside.
        let inner = 1.0 - smoothstep(-bw - 0.5, -bw + 0.5, d);
        color = mix(border_color, fill_color, inner);
    }
    let a = color.a * alpha;
    return vec4<f32>(color.rgb * a, a);
}

// --------------------------------------------------------------- glyphs ----

@group(1) @binding(0) var mask_tex: texture_2d<f32>;
@group(1) @binding(1) var mask_smp: sampler;
@group(1) @binding(2) var color_tex: texture_2d<f32>;

struct GlyphIn {
    @location(0) pos: vec2<f32>,          // top-left of the glyph quad
    @location(1) size: vec2<f32>,
    @location(2) uv: vec2<f32>,           // top-left texel in the atlas
    @location(3) color: vec4<f32>,        // straight-alpha tint (mask glyphs)
    @location(4) slant: f32,              // synthetic oblique shear
    @location(5) kind: f32,               // 0 = coverage mask, 1 = RGBA bitmap
};

struct GlyphOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) kind: f32,
};

@vertex
fn vs_glyph(@builtin(vertex_index) vi: u32, g: GlyphIn) -> GlyphOut {
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u));
    var p = g.pos + corner * g.size;
    // Shear around the quad's bottom edge (≈ baseline) for oblique text.
    p.x += g.slant * (g.size.y - corner.y * g.size.y);

    var o: GlyphOut;
    o.clip = to_clip(p);
    o.uv = (g.uv + corner * g.size) / G.atlas;
    o.color = g.color;
    o.kind = g.kind;
    return o;
}

@fragment
fn fs_glyph(i: GlyphOut) -> @location(0) vec4<f32> {
    if i.kind > 0.5 {
        let c = textureSample(color_tex, mask_smp, i.uv);
        let a = c.a * i.color.a;
        return vec4<f32>(c.rgb * a, a);
    }
    let cov = textureSample(mask_tex, mask_smp, i.uv).r;
    let a = i.color.a * cov;
    return vec4<f32>(i.color.rgb * a, a);
}
