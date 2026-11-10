//! Rendered slides as GPU textures: the canvas slide and thumbnails, cached by the slide's and its
//! master's `Arc` identity (an unchanged slide keeps its `Arc`, so it never re-renders).
//!
//! The canvas texture covers the slide plus whatever of the slide's drawing reaches off it and is
//! in view (PowerPoint's Normal view shows off-slide objects; thumbnails clip to the slide).

use std::collections::HashMap;
use std::sync::Arc;

use deckcraft_geom::{Rect, Size};
use deckcraft_model::style::Fill;
use deckcraft_model::{Geom, Master, Presentation, ShapeId, ShapeKind, Slide};
use deckcraft_render::{Image, Region, RenderOpts};
use egui::{ColorImage, TextureFilter, TextureHandle, TextureOptions, TextureWrapMode};

#[derive(Clone, PartialEq)]
struct Key {
    slide: usize,
    master: usize,
    size: (u32, u32),
    edit: bool,
    extra: u64,
    /// The pasteboard region the slide's drawing calls for (canvas only).
    want: Region,
    editing_text: Option<ShapeId>,
}

struct Entry {
    key: Key,
    tex: TextureHandle,
    /// The region the texture covers (at least `key.want`).
    region: Region,
    // Keep the Arcs alive so their addresses can't be reused while cached.
    _slide: Arc<Slide>,
    _master: Option<Arc<Master>>,
    stamp: u64,
}

struct GpuDragPreview {
    key: Key,
    shape: ShapeId,
    slide: Arc<Slide>,
    master: Option<Arc<Master>>,
    texture: TextureHandle,
    rect: [f64; 4],
    comparison: Option<DragComparison>,
}

struct DragComparison {
    base: Image,
    overlay: Image,
    region: Region,
    scale: f64,
}

/// Maximum normalized RGBA mean absolute error for the GPU drag composition against vello_cpu.
pub const GPU_DRAG_PREVIEW_MAE_TOLERANCE: f64 = 0.02;

#[derive(Default)]
pub struct Textures {
    canvas: Option<Entry>,
    drag_preview: Option<GpuDragPreview>,
    /// How far the canvas slide's shapes reach (from their boxes), for the slide and master `Arc`s
    /// held (kept alive so their addresses can't be reused), at this page size.
    estimate: Option<(Arc<Slide>, Option<Arc<Master>>, Size, Rect)>,
    /// How far the last canvas render of a slide (by index) actually drew: text spilling out of
    /// its box shows there.
    drawn: Option<(usize, Rect)>,
    /// Canvas renders so far (tests, performance checks).
    pub canvas_renders: u64,
    thumbs: HashMap<usize, Entry>,
    /// Thumbnails actually rendered, over the life of this cache. The number *held*
    /// ([`Textures::thumb_count`]) cannot see a pane that renders the same rows over and over --
    /// the cache thrash behind #969 -- because the count settles while the work does not.
    thumb_renders: u64,
    /// SmartArt gallery pictures by layout, style and colours (and, for hover previews, the
    /// outline's hash).
    smartart: HashMap<String, TextureHandle>,
    clock: u64,
    pub last_render_ms: f64,
}

fn shape_overlay_rect(xfrm: deckcraft_geom::Xfrm) -> [f64; 4] {
    let bounds = xfrm.bounds().inflate(8.0, 8.0);
    [bounds.x0, bounds.y0, bounds.x1, bounds.y1]
}

impl Textures {
    /// SmartArt gallery pictures cached so far (tests).
    pub fn smartart_cached(&self) -> usize {
        self.smartart.len()
    }

    /// A SmartArt layout drawn small for a gallery (`nodes` for a preview of the selected
    /// graphic's own outline), cached; `look` is the layout kind, style and colours; drawn with the active presentation's theme.
    pub fn smartart(
        &mut self,
        ctx: &egui::Context,
        p: &deckcraft_model::Presentation,
        slide: usize,
        (kind, style, colors): (&str, &str, &str),
        nodes: Option<&[deckcraft_model::DiagramNode]>,
        width_px: u32,
    ) -> Option<TextureHandle> {
        let mut key = format!("{kind}|{style}|{colors}|{width_px}");
        if let Some(n) = nodes {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for x in n {
                (x.text.as_str(), x.level).hash(&mut h);
            }
            key.push_str(&format!("|{:x}", h.finish()));
        }
        if let Some(t) = self.smartart.get(&key) {
            return Some(t.clone());
        }
        let mut s = p.slides.get(slide).or_else(|| p.slides.first())?.as_ref().clone();
        let g = deckcraft_engine::cmd::smartart::preview(kind, style, colors, nodes);
        let id = g.id;
        s.shapes = vec![g];
        let img = deckcraft_render::render_shape(p, &s, id, width_px as f64 / 336.0)?;
        let tex = upload(ctx, format!("smartart-{key}"), to_color_image(&img, false));
        if self.smartart.len() > 300 {
            self.smartart.clear();
        }
        self.smartart.insert(key, tex.clone());
        Some(tex)
    }
}

/// Snap `rect` onto the physical pixel grid.
///
/// The editor draws the slide as a texture of device pixels. A rect that starts or ends
/// between pixels is sampled with a linear filter, so every glyph edge is a blend of two
/// texels and small text looks soft. On a Retina display that offset is half a point, which
/// is the whole edge of an editing-size glyph. The slide show draws much larger glyphs, so
/// the same blend is not visible there (#184).
pub fn device_pixel_rect(rect: egui::Rect, ppp: f32) -> egui::Rect {
    if !ppp.is_finite() || ppp <= 0.0 || !rect.is_finite() {
        return rect;
    }
    let snap = |v: f32| {
        let p = v * ppp;
        if p.is_finite() { p.round() / ppp } else { v }
    };
    let min = egui::pos2(snap(rect.min.x), snap(rect.min.y));
    let step = 1.0 / ppp;
    let max = egui::pos2(snap(rect.max.x).max(min.x + step), snap(rect.max.y).max(min.y + step));
    egui::Rect::from_min_max(min, max)
}

/// Editor slide textures. Nearest magnification keeps a texel that is stretched by a
/// fraction of a pixel from blurring the glyph; minification still blends when many texels
/// share a screen pixel.
pub fn editor_texture_options() -> TextureOptions {
    TextureOptions {
        magnification: TextureFilter::Nearest,
        minification: TextureFilter::Linear,
        wrap_mode: TextureWrapMode::ClampToEdge,
        mipmap_mode: None,
    }
}

pub fn to_color_image(img: &Image, grayscale: bool) -> ColorImage {
    let w = img.width as usize;
    let h = img.height as usize;
    let mut px = Vec::with_capacity(w * h);
    for c in img.pixels.as_chunks::<4>().0 {
        let (r, g, b, a) = (c[0], c[1], c[2], c[3]);
        if grayscale {
            let y = ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29) >> 8) as u8;
            px.push(egui::Color32::from_rgba_premultiplied(y, y, y, a));
        } else {
            px.push(egui::Color32::from_rgba_premultiplied(r, g, b, a));
        }
    }
    px.resize(w * h, egui::Color32::TRANSPARENT);
    ColorImage::new([w, h], px)
}

/// The largest texture side the renderer (GPU or software fallback) takes; egui panics on
/// a bigger texture.
pub fn max_side(ctx: &egui::Context) -> u32 {
    ctx.input(|i| i.max_texture_side).clamp(1, u32::MAX as usize) as u32
}

/// `(w, h)` scaled down, keeping its shape, so neither side exceeds `max` (and none is 0).
/// Render sizes go through this so a huge display, DPI or slide shape doesn't render a
/// picture the GPU can't hold.
pub fn fit_size((w, h): (u32, u32), max: u32) -> (u32, u32) {
    let (w, h, max) = (w.max(1), h.max(1), max.max(1));
    let big = w.max(h);
    if big <= max {
        return (w, h);
    }
    let k = max as f64 / big as f64;
    (((w as f64 * k).floor() as u32).clamp(1, max), ((h as f64 * k).floor() as u32).clamp(1, max))
}

/// `img` shrunk by whole-pixel box averaging until both sides fit `max`.
pub fn shrink_to(img: ColorImage, max: u32) -> ColorImage {
    let [w, h] = img.size;
    let max = max.max(1) as usize;
    if w <= max && h <= max {
        return img;
    }
    let f = w.max(h).div_ceil(max);
    let (nw, nh) = (w.div_ceil(f).max(1), h.div_ceil(f).max(1));
    let mut px = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for sy in y * f..((y + 1) * f).min(h) {
                for sx in x * f..((x + 1) * f).min(w) {
                    if let Some(c) = img.pixels.get(sy * w + sx) {
                        for (a, v) in acc.iter_mut().zip(c.to_array()) {
                            *a += v as u32;
                        }
                        n += 1;
                    }
                }
            }
            let n = n.max(1);
            px.push(egui::Color32::from_rgba_premultiplied((acc[0] / n) as u8, (acc[1] / n) as u8, (acc[2] / n) as u8, (acc[3] / n) as u8));
        }
    }
    ColorImage::new([nw, nh], px)
}

/// Upload a texture, shrunk to the renderer's limit if need be.
pub fn upload(ctx: &egui::Context, name: impl Into<String>, img: ColorImage) -> TextureHandle {
    ctx.load_texture(name, shrink_to(img, max_side(ctx)), TextureOptions::LINEAR)
}

/// Replace a texture's picture, shrunk to the renderer's limit if need be.
pub fn replace(ctx: &egui::Context, tex: &mut TextureHandle, img: ColorImage) {
    tex.set(shrink_to(img, max_side(ctx)), TextureOptions::LINEAR);
}

/// [`upload`] for editor slide pictures, drawn on the device pixel grid with
/// [`editor_texture_options`] (#184).
pub fn upload_editor(ctx: &egui::Context, name: impl Into<String>, img: ColorImage) -> TextureHandle {
    ctx.load_texture(name, shrink_to(img, max_side(ctx)), editor_texture_options())
}

/// [`replace`] for editor slide pictures (#184).
pub fn replace_editor(ctx: &egui::Context, tex: &mut TextureHandle, img: ColorImage) {
    tex.set(shrink_to(img, max_side(ctx)), editor_texture_options());
}

fn master_of(p: &Presentation, slide: &Slide) -> Option<Arc<Master>> {
    p.masters.iter().find(|m| m.layout(slide.layout).is_some()).cloned()
}

fn same(a: &Option<Arc<Master>>, b: &Option<Arc<Master>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

/// The only difference is the transform of one plain shape or picture. Compare slide metadata
/// directly so drag frames don't clone every shape just to validate the cache fast path.
fn moved_shape(old: &Slide, new: &Slide) -> Option<(usize, deckcraft_geom::Xfrm, deckcraft_geom::Xfrm)> {
    if old.shapes.len() != new.shapes.len()
        || old.id != new.id
        || old.layout != new.layout
        || old.name != new.name
        || old.background != new.background
        || old.hidden != new.hidden
        || old.show_master_shapes != new.show_master_shapes
        || old.transition != new.transition
        || old.animations != new.animations
        || old.notes != new.notes
        || old.comments != new.comments
        || old.color_map != new.color_map
        || old.tags != new.tags
        || old.raw_ext != new.raw_ext
    {
        return None;
    }
    let mut changed = None;
    for (index, (before, after)) in old.shapes.iter().zip(&new.shapes).enumerate() {
        if before == after {
            continue;
        }
        if changed.replace(index).is_some() {
            return None;
        }
        if !drag_shape_is_simple(before) || !drag_shape_is_simple(after) {
            return None;
        }
        let (Some(a), Some(b)) = (before.xfrm, after.xfrm) else { return None };
        if a.w != b.w || a.h != b.h || a.flip_h != b.flip_h || a.flip_v != b.flip_v {
            return None;
        }
        let mut reverted = after.clone();
        reverted.xfrm = Some(a);
        if reverted != *before {
            return None;
        }
    }
    let Some(index) = changed else { return None };
    Some((index, old.shapes[index].xfrm?, new.shapes[index].xfrm?))
}

fn drag_shape_is_simple(shape: &deckcraft_model::Shape) -> bool {
    shape.xfrm.is_some_and(|xfrm| xfrm.is_finite() && xfrm.rot == 0.0)
        && matches!(&shape.kind, ShapeKind::Shape | ShapeKind::Picture { .. })
        && matches!(&shape.geom, Geom::Preset { .. })
        && shape.text.is_none()
        && !shape.effects.as_ref().is_some_and(|effects| !effects.is_empty())
        && shape.ph.is_none()
        && !shape.style.as_ref().is_some_and(|style| style.effect_ref.0 != 0)
        && shape.line.as_ref().is_some_and(|line| line.fill.as_ref().is_some_and(Fill::is_none))
        && !shape.hidden
}

/// A dragged layer is painted above every later shape, so only use it when their visible bounds
/// stay clear. Complex later shapes fall back to CPU because their bounds can extend past xfrm.
fn later_shapes_overlap(slide: &Slide, index: usize, moved: deckcraft_geom::Xfrm) -> bool {
    let moved = moved.bounds().inflate(8.0, 8.0);
    slide.shapes.iter().skip(index + 1).filter(|shape| !shape.hidden).any(|shape| {
        if !matches!(&shape.kind, ShapeKind::Shape | ShapeKind::Picture { .. })
            || !matches!(&shape.geom, Geom::Preset { .. })
            || shape.text.is_some()
            || shape.effects.as_ref().is_some_and(|effects| !effects.is_empty())
            || shape.ph.is_some()
        {
            return true;
        }
        let Some(xfrm) = shape.xfrm else { return true };
        let bounds = xfrm.bounds().inflate(16.0, 16.0);
        moved.x0 < bounds.x1 && bounds.x0 < moved.x1 && moved.y0 < bounds.y1 && bounds.y0 < moved.y1
    })
}

/// A conservative pixel region for a slide whose only change is translating one plain shape or
/// picture. Effects, text, groups, rotations and size changes fall back to the full renderer.
fn moved_shape_region(_index: usize, a: deckcraft_geom::Xfrm, b: deckcraft_geom::Xfrm, scale: f64, texture: Region) -> Option<(Region, [usize; 2])> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let bounds = a.bounds().union(b.bounds()).inflate(2.0 / scale, 2.0 / scale);
    if !bounds.is_finite() {
        return None;
    }
    let edges = [
        (bounds.x0 * scale).floor() as i64,
        (bounds.y0 * scale).floor() as i64,
        (bounds.x1 * scale).ceil() as i64,
        (bounds.y1 * scale).ceil() as i64,
    ];
    let tx0 = i64::from(texture.x);
    let ty0 = i64::from(texture.y);
    let tx1 = tx0 + i64::from(texture.w);
    let ty1 = ty0 + i64::from(texture.h);
    let (x0, y0, x1, y1) = (edges[0].max(tx0), edges[1].max(ty0), edges[2].min(tx1), edges[3].min(ty1));
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    let x = i32::try_from(x0).ok()?;
    let y = i32::try_from(y0).ok()?;
    let patch = Region { x, y, w: u32::try_from(x1 - x0).ok()?, h: u32::try_from(y1 - y0).ok()? };
    Some((patch, [usize::try_from(x0 - tx0).ok()?, usize::try_from(y0 - ty0).ok()?]))
}

/// Shapes that cannot reach a damage rectangle need not be parsed and rasterized again when the
/// editor patches a moved shape. Only skip simple, unrotated, untextured shapes with no effects or
/// outline; complex shapes stay in the patch render so text spill, shadows and custom geometry
/// continue to follow the full renderer's behavior.
fn skip_outside_patch(slide: &Slide, patch: Region, scale: f64) -> Vec<ShapeId> {
    if !scale.is_finite() || scale <= 0.0 {
        return Vec::new();
    }
    let damage = [
        f64::from(patch.x) / scale,
        f64::from(patch.y) / scale,
        (f64::from(patch.x) + f64::from(patch.w)) / scale,
        (f64::from(patch.y) + f64::from(patch.h)) / scale,
    ];
    slide
        .shapes
        .iter()
        .filter_map(|shape| {
            let xfrm = shape.xfrm?;
            let line_is_none = shape.line.as_ref().is_some_and(|line| line.fill.as_ref().is_some_and(Fill::is_none));
            let simple = matches!(&shape.kind, ShapeKind::Shape | ShapeKind::Picture { .. })
                && matches!(&shape.geom, Geom::Preset { .. })
                && shape.text.is_none()
                && shape.effects.as_ref().is_none_or(|effects| effects.is_empty())
                && shape.ph.is_none()
                && shape.style.as_ref().is_none_or(|style| style.effect_ref.0 == 0)
                && line_is_none
                && !shape.hidden
                && xfrm.is_finite()
                && xfrm.rot == 0.0;
            if !simple {
                return None;
            }
            let bounds = xfrm.bounds().inflate(2.0 / scale, 2.0 / scale);
            let intersects = bounds.x0 < damage[2] && bounds.x1 > damage[0] && bounds.y0 < damage[3] && bounds.y1 > damage[1];
            (!intersects).then_some(shape.id)
        })
        .collect()
}

/// Mix the preview playhead into the canvas cache key. Zero means "no preview", so a playhead of
/// 0s still misses the at-rest texture (entrances start hidden).
fn anim_key(anim: Option<(&deckcraft_anim::Timeline, f64)>) -> u64 {
    let Some((_, t)) = anim else { return 0 };
    let ms = if t.is_finite() { (t * 1000.0).clamp(0.0, 1.0e12) as u64 } else { 0 };
    ms.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn threads() -> u16 {
    if cfg!(target_arch = "wasm32") { 0 } else { std::thread::available_parallelism().map(|n| n.get().min(8) as u16).unwrap_or(0) }
}

impl Textures {
    /// The canvas texture for slide `index`, the slide being `size` pixels, and the region it
    /// covers: the slide, grown over the pasteboard to show what reaches off the slide within
    /// `visible` (the canvas, in pixels from the slide's top-left corner). `extra` changes with the
    /// GIF frames. `editing_text` replaces its placeholder prompt with gray empty list markers.
    #[allow(clippy::too_many_arguments)]
    pub fn slide(
        &mut self,
        ctx: &egui::Context,
        p: &Presentation,
        index: usize,
        size: (u32, u32),
        visible: Region,
        edit: bool,
        extra: u64,
        grayscale: bool,
        gif_times: &[(ShapeId, f64)],
        editing_text: Option<ShapeId>,
        anim: Option<(&deckcraft_anim::Timeline, f64)>,
        dragging_shape: Option<ShapeId>,
    ) -> Option<(TextureHandle, Region)> {
        let slide = p.slides.get(index)?.clone();
        let master = master_of(p, &slide);
        let scale = size.0 as f64 / p.slide_size.width.max(1.0);
        let movement = self.canvas.as_ref().and_then(|entry| same(&entry._master, &master).then(|| moved_shape(&entry._slide, &slide)).flatten());
        let cached_estimate = self.estimate.as_ref().map(|(s, m, page, r)| (s.clone(), m.clone(), *page, *r));
        let reusable_extent = match (&cached_estimate, movement) {
            (Some((cached_slide, cached_master, page, extent)), Some((_, _, new_xfrm)))
                if *page == p.slide_size
                    && self.canvas.as_ref().is_some_and(|entry| Arc::ptr_eq(cached_slide, &entry._slide))
                    && same(cached_master, &master)
                    && new_xfrm.bounds().x0 >= extent.x0
                    && new_xfrm.bounds().y0 >= extent.y0
                    && new_xfrm.bounds().x1 <= extent.x1
                    && new_xfrm.bounds().y1 <= extent.y1 =>
            {
                Some(*extent)
            }
            _ => None,
        };
        let estimate = match cached_estimate {
            Some((cached_slide, cached_master, page, extent))
                if page == p.slide_size
                    && same(&cached_master, &master)
                    && (Arc::ptr_eq(&cached_slide, &slide) || (movement.is_none() && cached_slide.as_ref() == slide.as_ref())) =>
            {
                extent
            }
            Some(_) if reusable_extent.is_some() => {
                // The baseline still contains this translation. Retaining the current editable
                // slide here would make Arc::make_mut copy the whole slide on every pointer frame.
                reusable_extent.unwrap_or_default()
            }
            _ => {
                let extent = deckcraft_render::slide_extent(p, index);
                self.estimate = Some((slide.clone(), master.clone(), p.slide_size, extent));
                extent
            }
        };
        let reach = match self.drawn {
            Some((i, d)) if i == index => estimate.union(d),
            _ => estimate,
        };
        // The GPU's (or software fallback's) texture limit: a larger canvas texture panics in egui.
        let max_side = max_side(ctx);
        let want = fit_texture(deckcraft_render::pasteboard_region(size, scale, reach, visible), visible, max_side);
        let key = Key {
            slide: Arc::as_ptr(&slide) as usize,
            master: master.as_ref().map(|m| Arc::as_ptr(m) as usize).unwrap_or(0),
            size,
            edit,
            extra: extra ^ grayscale as u64 ^ anim_key(anim),
            want,
            editing_text,
        };
        let t0 = crate::now_ms();
        if let Some(preview) = self.drag_preview.as_ref() {
            let mut comparable = key.clone();
            comparable.slide = preview.key.slide;
            let moved = moved_shape(&preview.slide, &slide);
            let preview_target = moved.map(|(shape_index, _, new_xfrm)| (shape_index, new_xfrm)).or_else(|| {
                if preview.slide.as_ref() != slide.as_ref() {
                    return None;
                }
                slide.shapes.iter().enumerate().find(|(_, shape)| shape.id == preview.shape).and_then(|(i, shape)| shape.xfrm.map(|xfrm| (i, xfrm)))
            });
            if let Some(selected) = dragging_shape
                && comparable == preview.key
                && same(&preview.master, &master)
                && let Some((shape_index, new_xfrm)) = preview_target
                && slide.shapes.get(shape_index).is_some_and(|shape| shape.id == selected && shape.id == preview.shape)
                && !later_shapes_overlap(&slide, shape_index, new_xfrm)
                && let Some(entry) = self.canvas.as_ref()
            {
                if let Some(preview) = self.drag_preview.as_mut() {
                    preview.rect = shape_overlay_rect(new_xfrm);
                }
                self.last_render_ms = crate::now_ms() - t0;
                return Some((entry.tex.clone(), entry.region));
            }
            self.drag_preview = None;
        }
        if let Some(e) = &self.canvas
            && e.key == key
            && dragging_shape.is_none()
        {
            return Some((e.tex.clone(), e.region));
        }
        let render = |region: Region, skip: &[ShapeId]| {
            let state_of = |id: ShapeId| anim.map(|(tl, time)| crate::preview::shape_state_at(tl, id, time)).unwrap_or_default();
            let state: Option<&dyn Fn(ShapeId) -> deckcraft_render::ShapeState> = if anim.is_some() { Some(&state_of) } else { None };
            let opts = RenderOpts {
                scale,
                edit,
                skip,
                gif_times,
                threads: threads(),
                offset: region.offset(),
                size: Some(region.size()),
                state,
                editing_text,
                ..Default::default()
            };
            deckcraft_render::render_slide_drawn(p, index, &opts)
        };
        let preview_target = dragging_shape.and_then(|selected| {
            if let Some((shape_index, _, new_xfrm)) = movement
                && slide.shapes.get(shape_index).is_some_and(|shape| shape.id == selected)
            {
                return Some((shape_index, new_xfrm));
            }
            self.canvas.as_ref().and_then(|entry| {
                let mut comparable = key.clone();
                comparable.slide = entry.key.slide;
                if entry.key != comparable || !same(&entry._master, &master) || entry._slide.as_ref() != slide.as_ref() {
                    return None;
                }
                let (shape_index, shape) = slide.shapes.iter().enumerate().find(|(_, shape)| shape.id == selected)?;
                if !drag_shape_is_simple(shape) {
                    return None;
                }
                Some((shape_index, shape.xfrm?))
            })
        });
        if anim.is_none()
            && gif_times.is_empty()
            && editing_text.is_none()
            && let Some(selected) = dragging_shape
            && let Some((shape_index, new_xfrm)) = preview_target
            && slide.shapes.get(shape_index).is_some_and(|shape| shape.id == selected)
            && !later_shapes_overlap(&slide, shape_index, new_xfrm)
            && let Some((entry_key, entry_slide, entry_master, region)) = self.canvas.as_ref().and_then(|entry| {
                let mut comparable = key.clone();
                comparable.slide = entry.key.slide;
                (entry.key == comparable && same(&entry._master, &master))
                    .then(|| (entry.key.clone(), entry._slide.clone(), entry._master.clone(), entry.region))
            })
        {
            let bounds = new_xfrm.bounds().inflate(8.0, 8.0);
            if bounds.is_finite() && bounds.width() * scale <= 2048.0 && bounds.height() * scale <= 2048.0 {
                let (base, _) = render(region, &[selected]);
                if base.width == region.w
                    && base.height == region.h
                    && let Some(overlay) = deckcraft_render::render_shapes(p, &slide, &[selected], scale)
                    && let Some(entry) = self.canvas.as_mut()
                {
                    replace_editor(ctx, &mut entry.tex, to_color_image(&base, grayscale));
                    let texture = upload_editor(ctx, format!("slide-drag-overlay-{}", selected.0), to_color_image(&overlay, grayscale));
                    let comparison = std::env::var_os("DECKCRAFT_GPU_DIFF_CHECK").map(|_| DragComparison {
                        base: base.clone(),
                        overlay: overlay.clone(),
                        region,
                        scale,
                    });
                    entry.key = entry_key.clone();
                    entry._slide = entry_slide.clone();
                    self.drag_preview = Some(GpuDragPreview {
                        key: entry_key,
                        shape: selected,
                        slide: entry_slide,
                        master: entry_master,
                        texture,
                        rect: shape_overlay_rect(new_xfrm),
                        comparison,
                    });
                    self.canvas_renders += 1;
                    self.last_render_ms = crate::now_ms() - t0;
                    return Some((entry.tex.clone(), entry.region));
                }
            }
        }
        if anim.is_none()
            && gif_times.is_empty()
            && editing_text.is_none()
            && let Some((index, old_xfrm, new_xfrm)) = movement
            && let Some((patch, pos)) = self.canvas.as_ref().and_then(|entry| {
                let mut comparable = key.clone();
                comparable.slide = entry.key.slide;
                (entry.key == comparable).then(|| moved_shape_region(index, old_xfrm, new_xfrm, scale, entry.region)).flatten()
            })
        {
            let skip = skip_outside_patch(&slide, patch, scale);
            let (img, _) = render(patch, &skip);
            if img.width == patch.w
                && img.height == patch.h
                && let Some(entry) = self.canvas.as_mut()
            {
                entry.tex.set_partial(pos, to_color_image(&img, grayscale), editor_texture_options());
                entry.key = key;
                entry._slide = slide;
                self.canvas_renders += 1;
                self.last_render_ms = crate::now_ms() - t0;
                return Some((entry.tex.clone(), entry.region));
            }
        }
        let (mut img, drawn) = render(want, &[]);
        self.canvas_renders += 1;
        let mut region = want;
        // Text spilling out of its box can reach further than the boxes said. Once drawn, the
        // region it calls for is known (and is next frame's key): render again only if it shows
        // more than this render covered.
        let key =
            Key { want: fit_texture(deckcraft_render::pasteboard_region(size, scale, estimate.union(drawn), visible), visible, max_side), ..key };
        if !region.contains(&key.want) {
            region = key.want;
            img = render(region, &[]).0;
            self.canvas_renders += 1;
        }
        self.drawn = Some((index, drawn));
        self.last_render_ms = crate::now_ms() - t0;
        let ci = to_color_image(&img, grayscale);
        let tex = match self.canvas.take() {
            Some(mut e) => {
                replace_editor(ctx, &mut e.tex, ci);
                e.tex
            }
            None => upload_editor(ctx, "slide-canvas", ci),
        };
        self.canvas = Some(Entry { key, tex: tex.clone(), region, _slide: slide, _master: master, stamp: 0 });
        Some((tex, region))
    }

    /// Cached thumbnails (bounded; soak tests check it).
    pub fn thumb_count(&self) -> usize {
        self.thumbs.len()
    }

    /// How many thumbnails this cache has rendered, ever. A pane that is done asks for no more:
    /// the number stops climbing, which is what tells a thrashing pane from a settled one.
    pub fn thumb_renders(&self) -> u64 {
        self.thumb_renders
    }

    /// Force the canvas to re-render next time (fonts loaded, theme changed in place…).
    pub fn invalidate(&mut self) {
        self.canvas = None;
        self.drag_preview = None;
        self.estimate = None;
        self.drawn = None;
        self.thumbs.clear();
    }

    /// Texture and slide-space bounds of the active one-shape GPU-composited drag preview.
    pub fn drag_preview(&self) -> Option<(TextureHandle, [f64; 4])> {
        self.drag_preview.as_ref().map(|preview| (preview.texture.clone(), preview.rect))
    }

    /// The CPU-rendered texture layers used by the native GPU comparison journey.
    pub fn drag_preview_layers(&self) -> Option<(Image, Image, [f64; 4], Region, f64)> {
        let preview = self.drag_preview.as_ref()?;
        let comparison = preview.comparison.as_ref()?;
        Some((comparison.base.clone(), comparison.overlay.clone(), preview.rect, comparison.region, comparison.scale))
    }

    /// A thumbnail texture; renders at most `budget` new thumbnails per frame. Over budget it
    /// returns the slide's previous thumbnail if there is one (and asks for another frame), else
    /// None, so the caller draws a placeholder.
    /// When `grayscale` is true, the thumbnail renders in black-and-white (matching Grayscale view).
    pub fn thumb(
        &mut self,
        ctx: &egui::Context,
        p: &Presentation,
        index: usize,
        width_px: u32,
        grayscale: bool,
        budget: &mut u32,
    ) -> Option<TextureHandle> {
        let slide = p.slides.get(index)?.clone();
        let master = master_of(p, &slide);
        let h = (width_px as f64 * p.slide_size.height / p.slide_size.width.max(1.0)).round().max(1.0) as u32;
        let key = Key {
            slide: Arc::as_ptr(&slide) as usize,
            master: master.as_ref().map(|m| Arc::as_ptr(m) as usize).unwrap_or(0),
            size: (width_px, h),
            edit: false,
            extra: grayscale as u64,
            want: Region::slide(width_px, h),
            editing_text: None,
        };
        self.clock += 1;
        let clock = self.clock;
        if let Some(e) = self.thumbs.get_mut(&key.slide)
            && e.key == key
        {
            e.stamp = clock;
            return Some(e.tex.clone());
        }
        if *budget == 0 {
            // Keep showing the stale thumbnail while the new one waits; request a repaint to finish rendering.
            ctx.request_repaint();
            return self.thumbs.get(&key.slide).map(|e| e.tex.clone());
        }
        *budget -= 1;
        let (tw, th) = fit_size((width_px, h), max_side(ctx));
        let scale = tw as f64 / p.slide_size.width.max(1.0);
        let img = deckcraft_render::render_slide(p, index, &RenderOpts { scale, size: Some((tw, th)), threads: 0, ..Default::default() });
        self.thumb_renders += 1;
        let tex = upload_editor(ctx, format!("thumb-{index}"), to_color_image(&img, grayscale));
        // Drop the old entry for this slide address; entries for slides that no longer exist age out.
        let region = key.want;
        self.thumbs.insert(key.slide, Entry { key, tex: tex.clone(), region, _slide: slide, _master: master, stamp: clock });
        if self.thumbs.len() > 600 {
            let mut v: Vec<(usize, u64)> = self.thumbs.iter().map(|(k, e)| (*k, e.stamp)).collect();
            v.sort_by_key(|x| x.1);
            for (k, _) in v.into_iter().take(200) {
                self.thumbs.remove(&k);
            }
        }
        Some(tex)
    }
}

#[cfg(test)]
mod tests {
    use super::{RenderOpts, device_pixel_rect, moved_shape, moved_shape_region};
    use egui::{Rect, pos2, vec2};

    fn on_grid(v: f32, ppp: f32) -> bool {
        let px = v * ppp;
        (px - px.round()).abs() < 1e-3
    }

    #[test]
    fn a_retina_slide_rect_lands_on_device_pixels() {
        let ppp = 2.0;
        let r = Rect::from_min_size(pos2(10.3, 20.25), vec2(100.2, 50.4));
        let s = device_pixel_rect(r, ppp);
        assert!(on_grid(s.min.x, ppp) && on_grid(s.min.y, ppp) && on_grid(s.max.x, ppp) && on_grid(s.max.y, ppp));
        assert!(s.width() > 0.0 && s.height() > 0.0);
        // One texture pixel per device pixel: the snapped size is an integer pixel count.
        let px = ((s.width() * ppp).round() as u32, (s.height() * ppp).round() as u32);
        assert!((s.width() / px.0 as f32 * ppp - 1.0).abs() < 1e-3);
        assert!((s.height() / px.1 as f32 * ppp - 1.0).abs() < 1e-3);
        let aligned = Rect::from_min_max(pos2(1.0, 2.5), pos2(80.0, 40.0));
        assert_eq!(device_pixel_rect(aligned, ppp), aligned);
    }

    #[test]
    fn to_color_image_turns_colors_to_grayscale_when_flag_is_set() {
        let img = deckcraft_render::Image { width: 2, height: 1, pixels: vec![0, 0, 255, 255, 255, 0, 0, 255] };
        let color = super::to_color_image(&img, false);
        assert_eq!(color.pixels[0], egui::Color32::from_rgba_premultiplied(0, 0, 255, 255));
        assert_eq!(color.pixels[1], egui::Color32::from_rgba_premultiplied(255, 0, 0, 255));

        let gray = super::to_color_image(&img, true);
        assert_eq!(gray.pixels[0].r(), gray.pixels[0].g());
        assert_eq!(gray.pixels[0].g(), gray.pixels[0].b());
        assert_eq!(gray.pixels[1].r(), gray.pixels[1].g());
        assert_eq!(gray.pixels[1].g(), gray.pixels[1].b());
    }

    #[test]
    fn a_bad_scale_does_not_move_the_rect() {
        let r = Rect::from_min_size(pos2(3.0, 4.0), vec2(10.0, 12.0));
        assert_eq!(device_pixel_rect(r, 0.0), r);
        assert_eq!(device_pixel_rect(r, f32::NAN), r);
    }

    #[test]
    fn a_drag_layer_falls_back_when_it_would_cover_a_later_shape() {
        use deckcraft_geom::Xfrm;
        use deckcraft_model::{Shape, ShapeId, Slide};

        let slide = Slide {
            shapes: vec![
                Shape { id: ShapeId(1), xfrm: Some(Xfrm::new(0.0, 0.0, 20.0, 20.0)), ..Default::default() },
                Shape { id: ShapeId(2), xfrm: Some(Xfrm::new(25.0, 0.0, 20.0, 20.0)), ..Default::default() },
            ],
            ..Default::default()
        };
        let moved = Xfrm::new(0.0, 0.0, 20.0, 20.0);
        assert!(super::later_shapes_overlap(&slide, 0, moved));
        assert!(!super::later_shapes_overlap(&slide, 1, moved));
    }

    #[test]
    fn a_drag_starting_at_rest_keeps_the_editable_slide_unique() {
        use deckcraft_color::SchemeSlot;
        use deckcraft_geom::Xfrm;
        use deckcraft_model::{Line, Presentation, Shape, ShapeId, ShapeStyle};
        use std::sync::Arc;

        let mut p = Presentation::default();
        Arc::make_mut(&mut p.slides[0]).shapes = vec![Shape {
            id: ShapeId(42),
            xfrm: Some(Xfrm::new(24.0, 30.0, 120.0, 90.0)),
            line: Some(Line::none()),
            style: Some(ShapeStyle::accent(SchemeSlot::Accent1)),
            effects: Some(Default::default()),
            ..Default::default()
        }];
        // An unchanged off-slide shape makes the cached extent wider than the page.
        Arc::make_mut(&mut p.slides[0]).shapes.push(Shape {
            id: ShapeId(43),
            xfrm: Some(Xfrm::new(1000.0, 30.0, 120.0, 90.0)),
            line: Some(Line::none()),
            ..Default::default()
        });
        let ctx = egui::Context::default();
        let viewport = deckcraft_render::Region::slide(960, 540);
        let mut textures = super::Textures::default();
        assert!(textures.slide(&ctx, &p, 0, (960, 540), viewport, true, 0, false, &[], None, None, None).is_some());
        // Match a pointer gesture: its first frame has not moved, and it can return to its origin.
        Arc::make_mut(&mut p.slides[0]);
        let editable = Arc::as_ptr(&p.slides[0]);
        for offset in [0.0, 3.0, 6.0, 0.0, 9.0] {
            Arc::make_mut(&mut p.slides[0]).shapes[0].xfrm.as_mut().unwrap().x = 24.0 + offset;
            assert!(textures.slide(&ctx, &p, 0, (960, 540), viewport, true, 0, false, &[], None, None, Some(ShapeId(42))).is_some());
            assert_eq!(Arc::strong_count(&p.slides[0]), 1, "the next pointer update must not clone the entire slide");
            assert_eq!(Arc::as_ptr(&p.slides[0]), editable, "translations must mutate the same editable slide");
        }
        // Equal slide contents do not make an extent valid after the page itself grows.
        Arc::make_mut(&mut p.slides[0]).shapes[0].xfrm.as_mut().unwrap().x = 24.0;
        p.slide_size.width = 1280.0;
        let wider = deckcraft_render::Region::slide(1280, 540);
        assert!(textures.slide(&ctx, &p, 0, (1280, 540), wider, true, 0, false, &[], None, None, None).is_some());
        assert!(textures.estimate.as_ref().unwrap().3.x1 >= 1280.0, "the cached bounds must include the resized page");
    }

    #[test]
    fn translating_one_shape_and_patching_its_dirty_region_matches_a_full_render() {
        use deckcraft_color::SchemeSlot;
        use deckcraft_geom::Xfrm;
        use deckcraft_model::{Line, Presentation, Shape, ShapeId, ShapeStyle};
        use std::sync::Arc;

        let mut before = Presentation::default();
        let shape_index = before.slides[0].shapes.len();
        Arc::make_mut(&mut before.slides[0]).shapes.push(Shape {
            id: ShapeId(42),
            xfrm: Some(Xfrm::new(24.0, 30.0, 120.0, 90.0)),
            line: Some(Line::none()),
            style: Some(ShapeStyle::accent(SchemeSlot::Accent1)),
            effects: Some(Default::default()),
            ..Default::default()
        });
        let mut after = before.clone();
        let shape = &mut Arc::make_mut(&mut after.slides[0]).shapes[shape_index];
        let mut moved = shape.xfrm.unwrap_or_default();
        moved.x += 70.0;
        moved.y += 35.0;
        shape.xfrm = Some(moved);

        let viewport = deckcraft_render::Region::slide(960, 540);
        let damage =
            moved_shape(&before.slides[0], &after.slides[0]).and_then(|(index, old, new)| moved_shape_region(index, old, new, 1.0, viewport));
        assert!(damage.is_some(), "single translation is patchable");
        let (patch, pos) = damage.unwrap_or_default();
        let mut reconstructed =
            deckcraft_render::render_slide(&before, 0, &RenderOpts { scale: 1.0, edit: true, size: Some(viewport.size()), ..Default::default() });
        let expected =
            deckcraft_render::render_slide(&after, 0, &RenderOpts { scale: 1.0, edit: true, size: Some(viewport.size()), ..Default::default() });
        let patch_img = deckcraft_render::render_slide(
            &after,
            0,
            &RenderOpts { scale: 1.0, edit: true, offset: patch.offset(), size: Some(patch.size()), ..Default::default() },
        );
        assert_eq!((patch_img.width, patch_img.height), patch.size());
        for y in 0..patch.h as usize {
            let source = y * patch.w as usize * 4;
            let target = ((pos[1] + y) * reconstructed.width as usize + pos[0]) * 4;
            let length = patch.w as usize * 4;
            reconstructed.pixels[target..target + length].copy_from_slice(&patch_img.pixels[source..source + length]);
        }
        assert_eq!(reconstructed.pixels, expected.pixels);
    }
}

/// `r` cut to at most `max` pixels a side, keeping as much of `visible` as fits.
pub(crate) fn fit_texture(mut r: Region, visible: Region, max: u32) -> Region {
    fn axis(start: &mut i32, len: &mut u32, vis: i32, max: u32) {
        if *len > max {
            let hi = *start as i64 + (*len - max) as i64;
            *start = (vis as i64).clamp(*start as i64, hi) as i32;
            *len = max;
        }
    }
    axis(&mut r.x, &mut r.w, visible.x, max);
    axis(&mut r.y, &mut r.h, visible.y, max);
    r
}

#[cfg(test)]
mod fit_tests {
    use super::*;

    #[test]
    fn canvas_textures_stay_within_the_texture_limit() {
        let r = Region { x: -100, y: -50, w: 5000, h: 1000 };
        let f = fit_texture(r, Region { x: 300, y: 0, w: 800, h: 600 }, 2048);
        assert_eq!((f.x, f.y, f.w, f.h), (300, -50, 2048, 1000));
        // Visible past the far end: the cut is pinned there.
        let f = fit_texture(r, Region { x: 4800, y: 0, w: 800, h: 600 }, 2048);
        assert_eq!((f.x, f.w), (-100 + 5000 - 2048, 2048));
        assert_eq!(fit_texture(r, r, 8192), r);
    }

    #[test]
    fn render_sizes_and_pictures_shrink_to_the_limit_keeping_their_shape() {
        assert_eq!(fit_size((195, 10920), 2048), (36, 2048));
        assert_eq!(fit_size((3000, 1500), 2048), (2048, 1024));
        assert_eq!(fit_size((800, 600), 2048), (800, 600));
        assert_eq!(fit_size((0, 0), 0), (1, 1));
        // A 5000 × 3 picture: half red, half blue, averaged down to 1667 × 1.
        let img = ColorImage::new(
            [5000, 3],
            (0..3)
                .flat_map(|y| (0..5000).map(move |x| (x, y)))
                .map(|(x, _)| if x < 2500 { egui::Color32::RED } else { egui::Color32::BLUE })
                .collect(),
        );
        let small = shrink_to(img, 2048);
        assert_eq!(small.size, [1667, 1]);
        assert_eq!(small.pixels[0], egui::Color32::RED);
        assert_eq!(small.pixels[1666], egui::Color32::BLUE);
        let ok = ColorImage::filled([10, 10], egui::Color32::WHITE);
        assert_eq!(shrink_to(ok.clone(), 2048).size, ok.size);
    }
}
