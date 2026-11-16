//! LPPwgpu — wgpu graphics package for LPPO.
//!
//! Provides WebGPU rendering via wgpu, exposed to LPPO through a C-ABI FFI
//! layer (libloading). Each exported symbol follows the fixed LPPO native
//! signature `extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64`.
//!
//! WGSL shaders can be supplied two ways:
//!   * Embedded directly in the package (compile-time `include_str!`)
//!   * Loaded from `.wgsl` shader files at runtime
//!
//! All wgpu state lives in static registries so the stateless FFI calls can
//! reach live wgpu objects.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowAttributes};

//////////////////////////////////////////////////////////////////////////////
// Type aliases
//////////////////////////////////////////////////////////////////////////////

type WindowId = i64;
type SurfaceId = i64;
type DeviceId = i64;
type ShaderId = i64;
type PipelineId = i64;
type BufferId = i64;
type TextureId = i64;

//////////////////////////////////////////////////////////////////////////////
// Global registries
//////////////////////////////////////////////////////////////////////////////

struct WindowEntry {
    window: Window,
}

struct SurfaceEntry {
    surface: wgpu::Surface<'static>,
    config: Option<wgpu::SurfaceConfiguration>,
    size: wgpu::Extent3d,
}

struct DeviceEntry {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct ShaderEntry {
    module: wgpu::ShaderModule,
    source: String,
}

struct PipelineEntry {
    pipeline: wgpu::RenderPipeline,
}

struct BufferEntry {
    buffer: wgpu::Buffer,
    size: wgpu::BufferAddress,
}

struct TextureEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: wgpu::Extent3d,
}

static WINDOWS: LazyLock<Mutex<HashMap<WindowId, WindowEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_WINDOW_ID: LazyLock<Mutex<WindowId>> = LazyLock::new(|| Mutex::new(1));

static INSTANCE: LazyLock<wgpu::Instance> = LazyLock::new(|| {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        backend_options: Default::default(),
        display: None,
        memory_budget_thresholds: Default::default(),
    })
});

static SURFACES: LazyLock<Mutex<HashMap<SurfaceId, SurfaceEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_SURFACE_ID: LazyLock<Mutex<SurfaceId>> = LazyLock::new(|| Mutex::new(1));

static DEVICES: LazyLock<Mutex<HashMap<DeviceId, DeviceEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_DEVICE_ID: LazyLock<Mutex<DeviceId>> = LazyLock::new(|| Mutex::new(1));

static SHADERS: LazyLock<Mutex<HashMap<ShaderId, ShaderEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_SHADER_ID: LazyLock<Mutex<ShaderId>> = LazyLock::new(|| Mutex::new(1));

static PIPELINES: LazyLock<Mutex<HashMap<PipelineId, PipelineEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_PIPELINE_ID: LazyLock<Mutex<PipelineId>> = LazyLock::new(|| Mutex::new(1));

static BUFFERS: LazyLock<Mutex<HashMap<BufferId, BufferEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_BUFFER_ID: LazyLock<Mutex<BufferId>> = LazyLock::new(|| Mutex::new(1));

static TEXTURES: LazyLock<Mutex<HashMap<TextureId, TextureEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_TEXTURE_ID: LazyLock<Mutex<TextureId>> = LazyLock::new(|| Mutex::new(1));

static LAST_ERROR: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

fn set_error(msg: &str) {
    if let Ok(mut guard) = LAST_ERROR.lock() {
        *guard = Some(msg.to_string());
    }
}

fn clear_error() {
    if let Ok(mut guard) = LAST_ERROR.lock() {
        *guard = None;
    }
}

//////////////////////////////////////////////////////////////////////////////
// Embedded WGSL shaders (compile-time)
// Add more shaders here to embed them directly in the package binary.
// These are referenced by name from LPPO via wgpu_create_shader_embedded.
// Example .wgsl files are provided in examples/shaders/.
//////////////////////////////////////////////////////////////////////////////

const EMBEDDED_SHADERS: &[(&str, &str)] = &[
    (
        "basic",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/shaders/basic.wgsl"
        )),
    ),
    (
        "color",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/shaders/color.wgsl"
        )),
    ),
    (
        "texture",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/shaders/texture.wgsl"
        )),
    ),
];

fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn find_embedded_shader(name_hash: u64) -> Option<&'static str> {
    for (name, source) in EMBEDDED_SHADERS {
        if hash_str(name) == name_hash {
            return Some(*source);
        }
    }
    None
}

//////////////////////////////////////////////////////////////////////////////
// C-ABI exports for LPPO FFI integration
//
// Fixed signature: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64
//
// LPPO pads missing arguments with 0. Str arguments are hashed to i64,
// so shader names and file paths are passed as their FNV-1a hash.
//////////////////////////////////////////////////////////////////////////////

#[no_mangle]
pub extern "C" fn wgpu_init(_: i64, _: i64, _: i64, _: i64, _: i64, _: i64, _: i64, _: i64) -> i64 {
    clear_error();
    eprintln!("[wgpu] init");
    0
}

#[no_mangle]
pub extern "C" fn wgpu_create_window(
    _: i64,
    width: i64,
    height: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let id = *NEXT_WINDOW_ID.lock().unwrap();
    *NEXT_WINDOW_ID.lock().unwrap() += 1;

    let event_loop = EventLoop::new().unwrap();
    let mut attrs = WindowAttributes::default();
    attrs.inner_size = Some(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
        width as f64,
        height as f64,
    )));
    attrs.title = format!("LPPwgpu window {}", id);

    let window = match event_loop.create_window(attrs) {
        Ok(w) => w,
        Err(e) => {
            set_error(&format!("failed to create window: {e}"));
            return 0;
        }
    };

    // Leak the event loop so the window remains valid after this function returns.
    // winit's EventLoop is not Send/Sync, so it cannot be stored in a static Mutex.
    let _leaked = Box::leak(Box::new(event_loop));

    WINDOWS.lock().unwrap().insert(id, WindowEntry { window });
    id
}

#[no_mangle]
pub extern "C" fn wgpu_destroy_window(
    window_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let removed = WINDOWS.lock().unwrap().remove(&window_id).is_some();
    if removed {
        1
    } else {
        set_error(&format!("window {window_id} not found"));
        0
    }
}

#[no_mangle]
pub extern "C" fn wgpu_create_surface(
    window_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let size = {
        let windows = WINDOWS.lock().unwrap();
        let entry = match windows.get(&window_id) {
            Some(e) => e,
            None => {
                set_error(&format!("window {window_id} not found"));
                return 0;
            }
        };
        entry.window.inner_size()
    };

    let id = *NEXT_SURFACE_ID.lock().unwrap();
    *NEXT_SURFACE_ID.lock().unwrap() += 1;

    let surface = {
        let windows = WINDOWS.lock().unwrap();
        let entry = windows.get(&window_id).expect("window checked above");
        let target = wgpu::SurfaceTarget::from_window_without_display(&entry.window);
        let surface = INSTANCE.create_surface(target).map_err(|e| {
            set_error(&format!("failed to create surface: {e}"));
            0i64
        });
        match surface {
            Ok(s) => unsafe {
                std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(s)
            },
            Err(_) => return 0,
        }
    };

    SURFACES.lock().unwrap().insert(
        id,
        SurfaceEntry {
            surface,
            config: None,
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        },
    );

    id
}

#[no_mangle]
pub extern "C" fn wgpu_destroy_surface(
    surface_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let removed = SURFACES.lock().unwrap().remove(&surface_id).is_some();
    if removed {
        1
    } else {
        set_error(&format!("surface {surface_id} not found"));
        0
    }
}

#[no_mangle]
pub extern "C" fn wgpu_request_adapter(
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    set_error("use wgpu_create_device to create adapter+device");
    0
}

#[no_mangle]
pub extern "C" fn wgpu_create_device(
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let id = *NEXT_DEVICE_ID.lock().unwrap();
    *NEXT_DEVICE_ID.lock().unwrap() += 1;

    let result = pollster::block_on(async {
        let adapter = INSTANCE
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| format!("no adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("no device: {e}"))?;
        Ok::<(wgpu::Device, wgpu::Queue), String>((device, queue))
    });

    match result {
        Ok((device, queue)) => {
            DEVICES.lock().unwrap().insert(id, DeviceEntry { device, queue });
            id
        }
        Err(e) => {
            set_error(&e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_get_queue(
    device_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    if DEVICES.lock().unwrap().get(&device_id).is_none() {
        set_error(&format!("device {device_id} not found"));
        return 0;
    }
    device_id
}

#[no_mangle]
pub extern "C" fn wgpu_configure_surface(
    surface_id: i64,
    device_id: i64,
    width: i64,
    height: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let mut surfaces = SURFACES.lock().unwrap();
    let devices = DEVICES.lock().unwrap();

    let surface_entry = match surfaces.get_mut(&surface_id) {
        Some(e) => e,
        None => {
            set_error(&format!("surface {surface_id} not found"));
            return 0;
        }
    };
    let device_entry = match devices.get(&device_id) {
        Some(e) => e,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let device = &device_entry.device;

    let size = wgpu::Extent3d {
        width: width as u32,
        height: height as u32,
        depth_or_array_layers: 1,
    };

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Bgra8UnormSrgb,
        width: size.width,
        height: size.height,
        color_space: wgpu::SurfaceColorSpace::Auto,
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    surface_entry.surface.configure(device, &config);
    surface_entry.config = Some(config);
    surface_entry.size = size;

    1
}

fn compile_wgsl(device: &wgpu::Device, source: &str) -> Result<wgpu::ShaderModule, String> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    Ok(module)
}

#[no_mangle]
pub extern "C" fn wgpu_create_shader_embedded(
    device_id: i64,
    name_hash: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let devices = DEVICES.lock().unwrap();
    let device_entry = match devices.get(&device_id) {
        Some(e) => e,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let device = &device_entry.device;

    let source = match find_embedded_shader(name_hash as u64) {
        Some(s) => s,
        None => {
            set_error(&format!("embedded shader with hash {name_hash} not found"));
            return 0;
        }
    };

    match compile_wgsl(device, source) {
        Ok(module) => {
            let id = *NEXT_SHADER_ID.lock().unwrap();
            *NEXT_SHADER_ID.lock().unwrap() += 1;
            SHADERS.lock().unwrap().insert(
                id,
                ShaderEntry {
                    module,
                    source: source.to_string(),
                },
            );
            id
        }
        Err(e) => {
            set_error(&e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_create_shader_file(
    device_id: i64,
    path_hash: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let devices = DEVICES.lock().unwrap();
    let device_entry = match devices.get(&device_id) {
        Some(e) => e,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let device = &device_entry.device;

    let search_dirs = [
        PathBuf::from("."),
        PathBuf::from("shaders"),
        PathBuf::from("./shaders"),
    ];
    let mut found_source: Option<String> = None;
    for dir in &search_dirs {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "wgsl") {
                    let path_str = path.to_string_lossy();
                    if hash_str(&path_str) == path_hash as u64 {
                        found_source = fs::read_to_string(&path).ok();
                        break;
                    }
                }
            }
            if found_source.is_some() {
                break;
            }
        }
    }

    let source = match found_source {
        Some(s) => s,
        None => {
            set_error(&format!("shader file with hash {path_hash} not found"));
            return 0;
        }
    };

    match compile_wgsl(device, &source) {
        Ok(module) => {
            let id = *NEXT_SHADER_ID.lock().unwrap();
            *NEXT_SHADER_ID.lock().unwrap() += 1;
            SHADERS.lock().unwrap().insert(
                id,
                ShaderEntry {
                    module,
                    source,
                },
            );
            id
        }
        Err(e) => {
            set_error(&e);
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_create_render_pipeline(
    device_id: i64,
    shader_id: i64,
    _swapchain_format: i64,
    _vertex_count: i64,
    vertex_stride: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let devices = DEVICES.lock().unwrap();
    let device_entry = match devices.get(&device_id) {
        Some(e) => e,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let device = &device_entry.device;

    let shaders = SHADERS.lock().unwrap();
    let shader_entry = match shaders.get(&shader_id) {
        Some(e) => e,
        None => {
            set_error(&format!("shader {shader_id} not found"));
            return 0;
        }
    };
    let module = &shader_entry.module;

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: vertex_stride as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x4,
                        offset: 16,
                        shader_location: 1,
                    },
                ],
            })],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let id = *NEXT_PIPELINE_ID.lock().unwrap();
    *NEXT_PIPELINE_ID.lock().unwrap() += 1;
    PIPELINES.lock().unwrap().insert(id, PipelineEntry { pipeline });
    id
}

#[no_mangle]
pub extern "C" fn wgpu_create_buffer(
    device_id: i64,
    size: i64,
    usage: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let devices = DEVICES.lock().unwrap();
    let device_entry = match devices.get(&device_id) {
        Some(e) => e,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let device = &device_entry.device;

    let buffer_usage = wgpu::BufferUsages::from_bits(usage as u32).unwrap_or(
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    );

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: buffer_usage,
        mapped_at_creation: false,
    });

    let id = *NEXT_BUFFER_ID.lock().unwrap();
    *NEXT_BUFFER_ID.lock().unwrap() += 1;
    BUFFERS.lock().unwrap().insert(
        id,
        BufferEntry {
            buffer,
            size: size as u64,
        },
    );
    id
}

#[no_mangle]
pub extern "C" fn wgpu_write_buffer(
    _buffer_id: i64,
    _offset: i64,
    _data_ptr: i64,
    _data_size: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    set_error("wgpu_write_buffer: pointer args not supported in 8-i64 FFI");
    0
}

#[no_mangle]
pub extern "C" fn wgpu_create_texture(
    device_id: i64,
    width: i64,
    height: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let devices = DEVICES.lock().unwrap();
    let device_entry = match devices.get(&device_id) {
        Some(e) => e,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let device = &device_entry.device;

    let size = wgpu::Extent3d {
        width: width as u32,
        height: height as u32,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let id = *NEXT_TEXTURE_ID.lock().unwrap();
    *NEXT_TEXTURE_ID.lock().unwrap() += 1;
    TEXTURES.lock().unwrap().insert(
        id,
        TextureEntry {
            texture,
            view,
            size,
        },
    );
    id
}

#[no_mangle]
pub extern "C" fn wgpu_begin_render_pass(
    device_id: i64,
    surface_id: i64,
    clear_r: i64,
    clear_g: i64,
    clear_b: i64,
    clear_a: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let mut surfaces = SURFACES.lock().unwrap();
    let devices = DEVICES.lock().unwrap();

    let surface_entry = match surfaces.get_mut(&surface_id) {
        Some(e) => e,
        None => {
            set_error(&format!("surface {surface_id} not found"));
            return 0;
        }
    };
    let device_entry = match devices.get(&device_id) {
        Some(e) => e,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let surface = &mut surface_entry.surface;
    let device = &device_entry.device;
    let queue = &device_entry.queue;

    let frame = match surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
        wgpu::CurrentSurfaceTexture::Occluded => {
            set_error("surface texture occluded");
            return 0;
        }
        wgpu::CurrentSurfaceTexture::Timeout => {
            set_error("surface texture timeout");
            return 0;
        }
        wgpu::CurrentSurfaceTexture::Outdated => {
            set_error("surface texture outdated");
            return 0;
        }
        wgpu::CurrentSurfaceTexture::Lost => {
            set_error("surface texture lost");
            return 0;
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            set_error("surface texture validation error");
            return 0;
        }
    };

    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let clear_color = wgpu::Color {
        r: clear_r as f64 / 255.0,
        g: clear_g as f64 / 255.0,
        b: clear_b as f64 / 255.0,
        a: clear_a as f64 / 255.0,
    };

    {
        let _rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear_color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    queue.submit(Some(encoder.finish()));
    drop(frame);

    1
}

#[no_mangle]
pub extern "C" fn wgpu_set_pipeline(
    pipeline_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    if PIPELINES.lock().unwrap().get(&pipeline_id).is_none() {
        set_error(&format!("pipeline {pipeline_id} not found"));
        return 0;
    }
    1
}

#[no_mangle]
pub extern "C" fn wgpu_set_vertex_buffer(
    buffer_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    if BUFFERS.lock().unwrap().get(&buffer_id).is_none() {
        set_error(&format!("buffer {buffer_id} not found"));
        return 0;
    }
    1
}

#[no_mangle]
pub extern "C" fn wgpu_draw(
    _: i64,
    _vertex_count: i64,
    _instance_count: i64,
    _first_vertex: i64,
    _first_instance: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    1
}

#[no_mangle]
pub extern "C" fn wgpu_end_render_pass(
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    1
}

#[no_mangle]
pub extern "C" fn wgpu_submit(
    device_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    if DEVICES.lock().unwrap().get(&device_id).is_none() {
        set_error(&format!("device {device_id} not found"));
        return 0;
    }
    1
}

#[no_mangle]
pub extern "C" fn wgpu_present(
    surface_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    if SURFACES.lock().unwrap().get(&surface_id).is_none() {
        set_error(&format!("surface {surface_id} not found"));
        return 0;
    }
    1
}

#[no_mangle]
pub extern "C" fn wgpu_poll_events(
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    1
}

#[no_mangle]
pub extern "C" fn wgpu_shader_list(
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let count = SHADERS.lock().unwrap().len();
    eprintln!("[wgpu] shader_list: {} shaders", count);
    count as i64
}

#[no_mangle]
pub extern "C" fn wgpu_shader_name(
    shader_id: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let source = {
        let binding = SHADERS.lock().unwrap();
        binding.get(&shader_id).map(|s| s.source.clone())
    };
    match source {
        Some(s) => {
            eprintln!(
                "[wgpu] shader {}: {}",
                shader_id,
                s.lines().next().unwrap_or("")
            );
            shader_id
        }
        None => {
            set_error(&format!("shader {shader_id} not found"));
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn wgpu_last_error(
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    let msg = LAST_ERROR.lock().unwrap().take();
    match msg {
        Some(m) => {
            eprintln!("[wgpu] last_error: {m}");
            hash_str(&m) as i64
        }
        None => 0,
    }
}
