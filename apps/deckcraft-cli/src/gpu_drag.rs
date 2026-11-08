use bytemuck::{Pod, Zeroable};
use deckcraft_engine::render::{Image, Region};
use egui_wgpu::wgpu::{self, util::DeviceExt as _};

const SHADER: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@vertex fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.uv = input.uv;
    return output;
}
@fragment fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(source, source_sampler, input.uv);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

pub struct Result {
    pub adapter: String,
    pub frame_ms: Vec<f64>,
    pub cpu_submit_ms: Vec<f64>,
    pub gpu_pass_ms: Vec<f64>,
    pub final_pixels: Vec<u8>,
}

/// Composite the retained slide background and moved transparent shape layer on Vulkan.
/// Timestamp queries measure the GPU pass; the final readback is compared with vello_cpu.
pub fn benchmark(base: &Image, overlay: &Image, rect: [f64; 4], region: Region, scale: f64, frames: usize) -> std::result::Result<Result, String> {
    if frames == 0 || !scale.is_finite() || scale <= 0.0 {
        return Err("GPU drag benchmark needs a positive frame count and scale".into());
    }
    for image in [base, overlay] {
        let expected_len = (image.width as usize).checked_mul(image.height as usize).and_then(|n| n.checked_mul(4));
        if image.width == 0 || image.height == 0 || expected_len != Some(image.pixels.len()) {
            return Err("GPU drag layer has invalid RGBA dimensions".into());
        }
    }

    pollster::block_on(async {
        let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(instance_descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("Vulkan adapter request failed: {e}"))?;
        let info = adapter.get_info();
        let expected = std::env::var("CRAFT_EXPECT_GPU_ADAPTER").unwrap_or_else(|_| "RTX 2080 Ti".into());
        if !info.name.contains(&expected) {
            return Err(format!("selected GPU adapter {:?} does not contain {expected:?}", info.name));
        }
        let required_features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        if !adapter.features().contains(required_features) {
            return Err(format!("GPU adapter {:?} does not support Vulkan timestamp queries", info.name));
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("DeckCraft drag compositor"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| format!("GPU device request failed: {e}"))?;

        let make_texture = |label: &'static str, image: &Image| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &image.pixels,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(image.width * 4), rows_per_image: Some(image.height) },
                wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
            );
            texture
        };
        let base_texture = make_texture("DeckCraft retained slide background", base);
        let overlay_texture = make_texture("DeckCraft transparent moved shape", overlay);
        let base_view = base_texture.create_view(&Default::default());
        let overlay_view = overlay_texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DeckCraft editor texture sampling"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeckCraft image layer bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DeckCraft texture layer"),
                layout: &bind_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            })
        };
        let base_bind = bind(&base_view);
        let overlay_bind = bind(&overlay_view);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DeckCraft drag layer shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DeckCraft drag layer pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let attributes = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DeckCraft drag image compositor"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &attributes,
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DeckCraft drag preview frame"),
            size: wgpu::Extent3d { width: base.width, height: base.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let query = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("DeckCraft drag GPU timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DeckCraft drag timestamp resolve"),
            size: 256,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DeckCraft drag timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let padded_row_bytes = (base.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DeckCraft final GPU preview readback"),
            size: u64::from(padded_row_bytes) * u64::from(base.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let base_vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DeckCraft background quad"),
            contents: bytemuck::cast_slice(&quad(0.0, 0.0, base.width as f32, base.height as f32, base.width, base.height)),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let overlay_vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DeckCraft moved shape quad"),
            size: (std::mem::size_of::<Vertex>() * 6) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut frame_ms = Vec::with_capacity(frames);
        let mut cpu_submit_ms = Vec::with_capacity(frames);
        let mut gpu_pass_ms = Vec::with_capacity(frames);
        let mut final_pixels = Vec::new();
        for frame in 0..frames {
            let submit_started = std::time::Instant::now();
            let movement = (frame as f64 - frames.saturating_sub(1) as f64) * 3.0;
            let frame_rect = [rect[0] + movement, rect[1] + movement * 0.4, rect[2] + movement, rect[3] + movement * 0.4];
            let x0 = (frame_rect[0] * scale - f64::from(region.x)) as f32;
            let y0 = (frame_rect[1] * scale - f64::from(region.y)) as f32;
            let x1 = (frame_rect[2] * scale - f64::from(region.x)) as f32;
            let y1 = (frame_rect[3] * scale - f64::from(region.y)) as f32;
            let vertices = quad(x0, y0, x1, y1, base.width, base.height);
            queue.write_buffer(&overlay_vertices, 0, bytemuck::cast_slice(&vertices));
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("DeckCraft GPU drag frame") });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("DeckCraft GPU drag image composition"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
                        query_set: &query,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &base_bind, &[]);
                pass.set_vertex_buffer(0, base_vertices.slice(..));
                pass.draw(0..6, 0..1);
                pass.set_bind_group(0, &overlay_bind, &[]);
                pass.set_vertex_buffer(0, overlay_vertices.slice(..));
                pass.draw(0..6, 0..1);
            }
            encoder.resolve_query_set(&query, 0..2, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &timestamp_readback, 0, 16);
            if frame + 1 == frames {
                encoder.copy_texture_to_buffer(
                    target.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &output_readback,
                        layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row_bytes), rows_per_image: Some(base.height) },
                    },
                    wgpu::Extent3d { width: base.width, height: base.height, depth_or_array_layers: 1 },
                );
            }
            let submission = queue.submit([encoder.finish()]);
            let submit_ms = submit_started.elapsed().as_secs_f64() * 1000.0;
            let timestamp_bytes = map_readback(&device, &timestamp_readback, 16, submission.clone())?;
            let start = u64::from_le_bytes(timestamp_bytes[0..8].try_into().map_err(|_| "GPU timestamp start is truncated")?);
            let end = u64::from_le_bytes(timestamp_bytes[8..16].try_into().map_err(|_| "GPU timestamp end is truncated")?);
            let elapsed = end.wrapping_sub(start) as f64 * f64::from(queue.get_timestamp_period()) / 1_000_000.0;
            if !elapsed.is_finite() || elapsed < 0.0 {
                return Err("GPU timestamp result is not finite".into());
            }
            cpu_submit_ms.push(submit_ms);
            gpu_pass_ms.push(elapsed);
            frame_ms.push(submit_ms + elapsed);
            if frame + 1 == frames {
                let bytes = map_readback(&device, &output_readback, output_readback.size(), submission)?;
                final_pixels.reserve((base.width * base.height * 4) as usize);
                for y in 0..base.height as usize {
                    let row = y * padded_row_bytes as usize;
                    final_pixels.extend_from_slice(&bytes[row..row + (base.width * 4) as usize]);
                }
            }
        }
        Ok(Result { adapter: info.name, frame_ms, cpu_submit_ms, gpu_pass_ms, final_pixels })
    })
}

fn map_readback(device: &wgpu::Device, buffer: &wgpu::Buffer, size: u64, submission: wgpu::SubmissionIndex) -> std::result::Result<Vec<u8>, String> {
    let slice = buffer.slice(..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result.map_err(|e| e.to_string()));
    });
    device
        .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(std::time::Duration::from_secs(5)) })
        .map_err(|e| e.to_string())?;
    rx.recv_timeout(std::time::Duration::from_secs(5)).map_err(|e| e.to_string())??;
    let view = slice.get_mapped_range().map_err(|e| e.to_string())?;
    let result = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(result)
}

fn quad(x0: f32, y0: f32, x1: f32, y1: f32, width: u32, height: u32) -> [Vertex; 6] {
    let ndc = |x: f32, y: f32| [x / width as f32 * 2.0 - 1.0, 1.0 - y / height as f32 * 2.0];
    let tl = Vertex { position: ndc(x0, y0), uv: [0.0, 0.0] };
    let tr = Vertex { position: ndc(x1, y0), uv: [1.0, 0.0] };
    let br = Vertex { position: ndc(x1, y1), uv: [1.0, 1.0] };
    let bl = Vertex { position: ndc(x0, y1), uv: [0.0, 1.0] };
    [tl, tr, br, tl, br, bl]
}
