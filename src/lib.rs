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
//! All wgpu state lives in registries so the stateless FFI calls can
//! reach live wgpu objects. On native targets the registries are
//! `LazyLock<Mutex<...>>`; on wasm they are `thread_local!<RefCell<...>>`
//! because wasm wgpu types are not `Send`/`Sync`.

#![allow(clippy::too_many_arguments)]

//////////////////////////////////////////////////////////////////////////////
// Imports
//////////////////////////////////////////////////////////////////////////////

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::collections::HashMap;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use std::ptr::NonNull;

//////////////////////////////////////////////////////////////////////////////
// Wasm surface handle wrapper
// wgpu's SurfaceTarget::Window requires HasWindowHandle + HasDisplayHandle.
// On wasm there is no display handle, so we implement HasDisplayHandle
// to return Unavailable.
//////////////////////////////////////////////////////////////////////////////

#[cfg(target_arch = "wasm32")]
struct WasmSurfaceHandle {
    handle: raw_window_handle::WebCanvasWindowHandle,
}

#[cfg(target_arch = "wasm32")]
impl raw_window_handle::HasWindowHandle for WasmSurfaceHandle {
    fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        let raw = raw_window_handle::RawWindowHandle::WebCanvas(self.handle.clone());
        Ok(unsafe { raw_window_handle::WindowHandle::borrow_raw(raw) })
    }
}

#[cfg(target_arch = "wasm32")]
impl raw_window_handle::HasDisplayHandle for WasmSurfaceHandle {
    fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        Err(raw_window_handle::HandleError::Unavailable)
    }
}

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
// Global registries (native)
//////////////////////////////////////////////////////////////////////////////

#[cfg(not(target_arch = "wasm32"))]
struct WindowEntry {
    window: winit::window::Window,
}

#[cfg(not(target_arch = "wasm32"))]
struct SurfaceEntry {
    surface: wgpu::Surface<'static>,
    config: Option<wgpu::SurfaceConfiguration>,
    size: wgpu::Extent3d,
}

#[cfg(not(target_arch = "wasm32"))]
struct DeviceEntry {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(not(target_arch = "wasm32"))]
struct ShaderEntry {
    module: wgpu::ShaderModule,
    source: String,
}

#[cfg(not(target_arch = "wasm32"))]
struct PipelineEntry {
    pipeline: wgpu::RenderPipeline,
}

#[cfg(not(target_arch = "wasm32"))]
struct BufferEntry {
    buffer: wgpu::Buffer,
    size: wgpu::BufferAddress,
}

#[cfg(not(target_arch = "wasm32"))]
struct TextureEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: wgpu::Extent3d,
}

#[cfg(not(target_arch = "wasm32"))]
static WINDOWS: LazyLock<Mutex<HashMap<WindowId, WindowEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(not(target_arch = "wasm32"))]
static NEXT_WINDOW_ID: LazyLock<Mutex<WindowId>> = LazyLock::new(|| Mutex::new(1));

#[cfg(not(target_arch = "wasm32"))]
static INSTANCE: LazyLock<wgpu::Instance> = LazyLock::new(|| {
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::default(),
        backend_options: Default::default(),
        display: None,
        memory_budget_thresholds: Default::default(),
    })
});

#[cfg(not(target_arch = "wasm32"))]
static SURFACES: LazyLock<Mutex<HashMap<SurfaceId, SurfaceEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(not(target_arch = "wasm32"))]
static NEXT_SURFACE_ID: LazyLock<Mutex<SurfaceId>> = LazyLock::new(|| Mutex::new(1));

#[cfg(not(target_arch = "wasm32"))]
static DEVICES: LazyLock<Mutex<HashMap<DeviceId, DeviceEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(not(target_arch = "wasm32"))]
static NEXT_DEVICE_ID: LazyLock<Mutex<DeviceId>> = LazyLock::new(|| Mutex::new(1));

#[cfg(not(target_arch = "wasm32"))]
static SHADERS: LazyLock<Mutex<HashMap<ShaderId, ShaderEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(not(target_arch = "wasm32"))]
static NEXT_SHADER_ID: LazyLock<Mutex<ShaderId>> = LazyLock::new(|| Mutex::new(1));

#[cfg(not(target_arch = "wasm32"))]
static PIPELINES: LazyLock<Mutex<HashMap<PipelineId, PipelineEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(not(target_arch = "wasm32"))]
static NEXT_PIPELINE_ID: LazyLock<Mutex<PipelineId>> = LazyLock::new(|| Mutex::new(1));

#[cfg(not(target_arch = "wasm32"))]
static BUFFERS: LazyLock<Mutex<HashMap<BufferId, BufferEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(not(target_arch = "wasm32"))]
static NEXT_BUFFER_ID: LazyLock<Mutex<BufferId>> = LazyLock::new(|| Mutex::new(1));

#[cfg(not(target_arch = "wasm32"))]
static TEXTURES: LazyLock<Mutex<HashMap<TextureId, TextureEntry>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(not(target_arch = "wasm32"))]
static NEXT_TEXTURE_ID: LazyLock<Mutex<TextureId>> = LazyLock::new(|| Mutex::new(1));

static LAST_ERROR: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));

//////////////////////////////////////////////////////////////////////////////
// Global registries (wasm)
//////////////////////////////////////////////////////////////////////////////

#[cfg(target_arch = "wasm32")]
struct WindowEntry {
    canvas: web_sys::HtmlCanvasElement,
}

#[cfg(target_arch = "wasm32")]
struct SurfaceEntry {
    surface: wgpu::Surface<'static>,
    config: Option<wgpu::SurfaceConfiguration>,
    size: wgpu::Extent3d,
}

#[cfg(target_arch = "wasm32")]
struct DeviceEntry {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(target_arch = "wasm32")]
struct ShaderEntry {
    module: wgpu::ShaderModule,
    source: String,
}

#[cfg(target_arch = "wasm32")]
struct PipelineEntry {
    pipeline: wgpu::RenderPipeline,
}

#[cfg(target_arch = "wasm32")]
struct BufferEntry {
    buffer: wgpu::Buffer,
    size: wgpu::BufferAddress,
}

#[cfg(target_arch = "wasm32")]
struct TextureEntry {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: wgpu::Extent3d,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WINDOWS: RefCell<HashMap<WindowId, WindowEntry>> = RefCell::new(HashMap::new());
    static NEXT_WINDOW_ID: RefCell<WindowId> = RefCell::new(1);
    static SURFACES: RefCell<HashMap<SurfaceId, SurfaceEntry>> = RefCell::new(HashMap::new());
    static NEXT_SURFACE_ID: RefCell<SurfaceId> = RefCell::new(1);
    static DEVICES: RefCell<HashMap<DeviceId, DeviceEntry>> = RefCell::new(HashMap::new());
    static NEXT_DEVICE_ID: RefCell<DeviceId> = RefCell::new(1);
    static SHADERS: RefCell<HashMap<ShaderId, ShaderEntry>> = RefCell::new(HashMap::new());
    static NEXT_SHADER_ID: RefCell<ShaderId> = RefCell::new(1);
    static PIPELINES: RefCell<HashMap<PipelineId, PipelineEntry>> = RefCell::new(HashMap::new());
    static NEXT_PIPELINE_ID: RefCell<PipelineId> = RefCell::new(1);
    static BUFFERS: RefCell<HashMap<BufferId, BufferEntry>> = RefCell::new(HashMap::new());
    static NEXT_BUFFER_ID: RefCell<BufferId> = RefCell::new(1);
    static TEXTURES: RefCell<HashMap<TextureId, TextureEntry>> = RefCell::new(HashMap::new());
    static NEXT_TEXTURE_ID: RefCell<TextureId> = RefCell::new(1);
}

#[cfg(target_arch = "wasm32")]
static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();

#[cfg(target_arch = "wasm32")]
fn wasm_instance() -> &'static wgpu::Instance {
    INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            flags: wgpu::InstanceFlags::default(),
            backend_options: Default::default(),
            display: None,
            memory_budget_thresholds: Default::default(),
        })
    })
}

//////////////////////////////////////////////////////////////////////////////
// Registry accessor helpers (cfg-dispatched)
// These avoid cfg_if! inside expression bodies.
//////////////////////////////////////////////////////////////////////////////

#[cfg(not(target_arch = "wasm32"))]
fn dev_contains(id: DeviceId) -> bool {
    DEVICES.lock().unwrap().get(&id).is_some()
}
#[cfg(target_arch = "wasm32")]
fn dev_contains(id: DeviceId) -> bool {
    DEVICES.with(|cell| cell.borrow().get(&id).is_some())
}

#[cfg(not(target_arch = "wasm32"))]
fn surf_contains(id: SurfaceId) -> bool {
    SURFACES.lock().unwrap().get(&id).is_some()
}
#[cfg(target_arch = "wasm32")]
fn surf_contains(id: SurfaceId) -> bool {
    SURFACES.with(|cell| cell.borrow().get(&id).is_some())
}

#[cfg(not(target_arch = "wasm32"))]
fn pipe_contains(id: PipelineId) -> bool {
    PIPELINES.lock().unwrap().get(&id).is_some()
}
#[cfg(target_arch = "wasm32")]
fn pipe_contains(id: PipelineId) -> bool {
    PIPELINES.with(|cell| cell.borrow().get(&id).is_some())
}

#[cfg(not(target_arch = "wasm32"))]
fn buf_contains(id: BufferId) -> bool {
    BUFFERS.lock().unwrap().get(&id).is_some()
}
#[cfg(target_arch = "wasm32")]
fn buf_contains(id: BufferId) -> bool {
    BUFFERS.with(|cell| cell.borrow().get(&id).is_some())
}

#[cfg(not(target_arch = "wasm32"))]
fn shader_len() -> usize {
    SHADERS.lock().unwrap().len()
}
#[cfg(target_arch = "wasm32")]
fn shader_len() -> usize {
    SHADERS.with(|cell| cell.borrow().len())
}

#[cfg(not(target_arch = "wasm32"))]
fn shader_source(id: ShaderId) -> Option<String> {
    SHADERS.lock().unwrap().get(&id).map(|s| s.source.clone())
}
#[cfg(target_arch = "wasm32")]
fn shader_source(id: ShaderId) -> Option<String> {
    SHADERS.with(|cell| cell.borrow().get(&id).map(|s| s.source.clone()))
}

//////////////////////////////////////////////////////////////////////////////
// Error handling
//////////////////////////////////////////////////////////////////////////////

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
// Platform-specific helpers
//////////////////////////////////////////////////////////////////////////////

#[cfg(not(target_arch = "wasm32"))]
fn window_inner_size(window: &winit::window::Window) -> winit::dpi::PhysicalSize<u32> {
    window.inner_size()
}

#[cfg(target_arch = "wasm32")]
fn window_inner_size(_canvas: &web_sys::HtmlCanvasElement) -> winit::dpi::PhysicalSize<u32> {
    winit::dpi::PhysicalSize::new(800, 600)
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

// ---------------------------------------------------------------------------
// wgpu_create_window
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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

    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    let mut attrs = winit::window::WindowAttributes::default();
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

    let _leaked = Box::leak(Box::new(event_loop));

    WINDOWS.lock().unwrap().insert(id, WindowEntry { window });
    id
}

#[cfg(target_arch = "wasm32")]
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
    let id = NEXT_WINDOW_ID.with(|cell| {
        let mut c = cell.borrow_mut();
        let cur = *c;
        *c += 1;
        cur
    });

    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);
    canvas.set_id(&format!("lppwgpu-canvas-{}", id));

    if let Some(body) = document.body() {
        let _ = body.append_child(&canvas);
    }

    WINDOWS.with(|cell| {
        cell.borrow_mut().insert(
            id,
            WindowEntry { canvas },
        );
    });
    id
}

// ---------------------------------------------------------------------------
// wgpu_destroy_window
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
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
    let removed = WINDOWS.with(|cell| cell.borrow_mut().remove(&window_id).is_some());
    if removed {
        1
    } else {
        set_error(&format!("window {window_id} not found"));
        0
    }
}

// ---------------------------------------------------------------------------
// wgpu_create_surface
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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
        window_inner_size(&entry.window)
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

#[cfg(target_arch = "wasm32")]
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
    let size = WINDOWS.with(|cell| {
        cell.borrow()
            .get(&window_id)
            .map(|e| window_inner_size(&e.canvas))
            .ok_or_else(|| {
                set_error(&format!("window {window_id} not found"));
                0i64
            })
    });
    let size = match size {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let id = NEXT_SURFACE_ID.with(|cell| {
        let mut c = cell.borrow_mut();
        let cur = *c;
        *c += 1;
        cur
    });

    let surface = WINDOWS.with(|cell| {
        let canvas = cell.borrow().get(&window_id).map(|e| e.canvas.clone());
        let canvas = match canvas {
            Some(c) => c,
            None => {
                set_error(&format!("window {window_id} not found"));
                return None;
            }
        };
        let ptr = &canvas as *const web_sys::HtmlCanvasElement as *mut std::ffi::c_void;
        let handle = raw_window_handle::WebCanvasWindowHandle::new(
            NonNull::new(ptr).unwrap(),
        );
        let wasm_handle = WasmSurfaceHandle { handle };
        let target = wgpu::SurfaceTarget::Window(Box::new(wasm_handle));
        match wasm_instance().create_surface(target) {
            Ok(s) => Some(unsafe {
                std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(s)
            }),
            Err(e) => {
                set_error(&format!("failed to create surface: {e}"));
                None
            }
        }
    });
    let surface = match surface {
        Some(s) => s,
        None => return 0,
    };

    SURFACES.with(|cell| {
        cell.borrow_mut().insert(
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
    });

    id
}

// ---------------------------------------------------------------------------
// wgpu_destroy_surface
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
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
    let removed = SURFACES.with(|cell| cell.borrow_mut().remove(&surface_id).is_some());
    if removed {
        1
    } else {
        set_error(&format!("surface {surface_id} not found"));
        0
    }
}

// ---------------------------------------------------------------------------
// wgpu_request_adapter
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// wgpu_create_device
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
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
    let id = NEXT_DEVICE_ID.with(|cell| {
        let mut c = cell.borrow_mut();
        let cur = *c;
        *c += 1;
        cur
    });

    let instance = wasm_instance();
    wasm_bindgen_futures::spawn_local(async move {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .expect("no webgpu adapter");
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
            .expect("no webgpu device");
        DEVICES.with(|cell| {
            cell.borrow_mut().insert(
                id,
                DeviceEntry { device, queue },
            );
        });
    });

    id
}

// ---------------------------------------------------------------------------
// wgpu_get_queue
// ---------------------------------------------------------------------------

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
    if !dev_contains(device_id) {
        set_error(&format!("device {device_id} not found"));
        return 0;
    }
    device_id
}

// ---------------------------------------------------------------------------
// wgpu_configure_surface
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(target_arch = "wasm32")]
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

    let surface_ptr = SURFACES.with(|cell| {
        cell.borrow_mut()
            .get_mut(&surface_id)
            .map(|e| &mut e.surface as *mut wgpu::Surface<'static>)
    });
    let surface_ptr = match surface_ptr {
        Some(p) => p,
        None => {
            set_error(&format!("surface {surface_id} not found"));
            return 0;
        }
    };
    let surface = unsafe { &mut *surface_ptr };

    let device = DEVICES.with(|cell| {
        cell.borrow().get(&device_id).map(|e| e.device.clone())
    });
    let device = match device {
        Some(d) => d,
        None => {
            set_error(&format!("device {device_id} not found"));
            return 0;
        }
    };
    let _queue = DEVICES.with(|cell| {
        cell.borrow().get(&device_id).map(|e| e.queue.clone())
    });

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

    surface.configure(&device, &config);

    SURFACES.with(|cell| {
        cell.borrow_mut().get_mut(&surface_id).unwrap().config = Some(config);
        cell.borrow_mut().get_mut(&surface_id).unwrap().size = size;
    });

    let _ = _queue;
    1
}

// ---------------------------------------------------------------------------
// Shader compilation
// ---------------------------------------------------------------------------

fn compile_wgsl(device: &wgpu::Device, source: &str) -> Result<wgpu::ShaderModule, String> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    Ok(module)
}

// ---------------------------------------------------------------------------
// wgpu_create_shader_embedded / wgpu_create_shader_file
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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
    let device = device_entry.device.clone();
    drop(devices);

    let source = match find_embedded_shader(name_hash as u64) {
        Some(s) => s,
        None => {
            set_error(&format!("embedded shader with hash {name_hash} not found"));
            return 0;
        }
    };

    match compile_wgsl(&device, source) {
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

#[cfg(target_arch = "wasm32")]
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
    let device = DEVICES.with(|cell| {
        cell.borrow()
            .get(&device_id)
            .map(|e| e.device.clone())
            .ok_or_else(|| {
                set_error(&format!("device {device_id} not found"));
                0i64
            })
    });
    let device = match device {
        Ok(d) => d,
        Err(_) => return 0,
    };

    let source = match find_embedded_shader(name_hash as u64) {
        Some(s) => s,
        None => {
            set_error(&format!("embedded shader with hash {name_hash} not found"));
            return 0;
        }
    };

    match compile_wgsl(&device, source) {
        Ok(module) => {
            let id = NEXT_SHADER_ID.with(|cell| {
                let mut c = cell.borrow_mut();
                let cur = *c;
                *c += 1;
                cur
            });
            SHADERS.with(|cell| {
                cell.borrow_mut().insert(
                    id,
                    ShaderEntry {
                        module,
                        source: source.to_string(),
                    },
                );
            });
            id
        }
        Err(e) => {
            set_error(&e);
            0
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
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
    let device = device_entry.device.clone();
    drop(devices);

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

    match compile_wgsl(&device, &source) {
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

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn wgpu_create_shader_file(
    device_id: i64,
    _path_hash: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
    _: i64,
) -> i64 {
    clear_error();
    let _device = DEVICES.with(|cell| cell.borrow().get(&device_id).is_some());
    if !_device {
        set_error(&format!("device {device_id} not found"));
        return 0;
    }
    // On wasm, shader files are not accessible via fs::read_dir.
    // Shaders must be embedded via wgpu_create_shader_embedded.
    set_error("wgpu_create_shader_file not supported on wasm; use embedded shaders");
    0
}

// ---------------------------------------------------------------------------
// wgpu_create_render_pipeline
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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
    let device = device_entry.device.clone();
    drop(devices);

    let shaders = SHADERS.lock().unwrap();
    let shader_entry = match shaders.get(&shader_id) {
        Some(e) => e,
        None => {
            set_error(&format!("shader {shader_id} not found"));
            return 0;
        }
    };
    let module = shader_entry.module.clone();
    drop(shaders);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
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
            module: &module,
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

#[cfg(target_arch = "wasm32")]
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
    let device = DEVICES.with(|cell| {
        cell.borrow()
            .get(&device_id)
            .map(|e| e.device.clone())
            .ok_or_else(|| {
                set_error(&format!("device {device_id} not found"));
                0i64
            })
    });
    let device = match device {
        Ok(d) => d,
        Err(_) => return 0,
    };

    let module = SHADERS.with(|cell| {
        cell.borrow()
            .get(&shader_id)
            .map(|e| e.module.clone())
            .ok_or_else(|| {
                set_error(&format!("shader {shader_id} not found"));
                0i64
            })
    });
    let module = match module {
        Ok(m) => m,
        Err(_) => return 0,
    };

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
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
            module: &module,
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

    let id = NEXT_PIPELINE_ID.with(|cell| {
        let mut c = cell.borrow_mut();
        let cur = *c;
        *c += 1;
        cur
    });
    PIPELINES.with(|cell| {
        cell.borrow_mut().insert(id, PipelineEntry { pipeline });
    });
    id
}

// ---------------------------------------------------------------------------
// wgpu_create_buffer
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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
    let device = device_entry.device.clone();
    drop(devices);

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

#[cfg(target_arch = "wasm32")]
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
    let device = DEVICES.with(|cell| {
        cell.borrow()
            .get(&device_id)
            .map(|e| e.device.clone())
            .ok_or_else(|| {
                set_error(&format!("device {device_id} not found"));
                0i64
            })
    });
    let device = match device {
        Ok(d) => d,
        Err(_) => return 0,
    };

    let buffer_usage = wgpu::BufferUsages::from_bits(usage as u32).unwrap_or(
        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    );

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: buffer_usage,
        mapped_at_creation: false,
    });

    let id = NEXT_BUFFER_ID.with(|cell| {
        let mut c = cell.borrow_mut();
        let cur = *c;
        *c += 1;
        cur
    });
    BUFFERS.with(|cell| {
        cell.borrow_mut().insert(
            id,
            BufferEntry {
                buffer,
                size: size as u64,
            },
        );
    });
    id
}

// ---------------------------------------------------------------------------
// wgpu_write_buffer / wgpu_create_texture
// ---------------------------------------------------------------------------

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

#[cfg(not(target_arch = "wasm32"))]
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
    let device = device_entry.device.clone();
    drop(devices);

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

#[cfg(target_arch = "wasm32")]
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
    let device = DEVICES.with(|cell| {
        cell.borrow()
            .get(&device_id)
            .map(|e| e.device.clone())
            .ok_or_else(|| {
                set_error(&format!("device {device_id} not found"));
                0i64
            })
    });
    let device = match device {
        Ok(d) => d,
        Err(_) => return 0,
    };

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

    let id = NEXT_TEXTURE_ID.with(|cell| {
        let mut c = cell.borrow_mut();
        let cur = *c;
        *c += 1;
        cur
    });
    TEXTURES.with(|cell| {
        cell.borrow_mut().insert(
            id,
            TextureEntry {
                texture,
                view,
                size,
            },
        );
    });
    id
}

// ---------------------------------------------------------------------------
// wgpu_begin_render_pass
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
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

    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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

#[cfg(target_arch = "wasm32")]
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
    let surface_ptr = SURFACES.with(|cell| {
        cell.borrow_mut()
            .get_mut(&surface_id)
            .map(|e| &mut e.surface as *mut wgpu::Surface<'static>)
            .ok_or_else(|| {
                set_error(&format!("surface {surface_id} not found"));
                0i64
            })
    });
    let surface_ptr = match surface_ptr {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let surface = unsafe { &mut *surface_ptr };

    let (device, queue) = match DEVICES.with(|cell| {
        cell.borrow()
            .get(&device_id)
            .map(|e| (e.device.clone(), e.queue.clone()))
            .ok_or_else(|| {
                set_error(&format!("device {device_id} not found"));
                0i64
            })
    }) {
        Ok((d, q)) => (d, q),
        Err(_) => return 0,
    };

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

    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

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

// ---------------------------------------------------------------------------
// wgpu_set_pipeline / wgpu_set_vertex_buffer / wgpu_draw / wgpu_end_render_pass
// ---------------------------------------------------------------------------

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
    if !pipe_contains(pipeline_id) {
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
    if !buf_contains(buffer_id) {
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

// ---------------------------------------------------------------------------
// wgpu_submit / wgpu_present / wgpu_poll_events
// ---------------------------------------------------------------------------

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
    if !dev_contains(device_id) {
        set_error(&format!("device {device_id} not found"));
        return 0;
    }
    device_id
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
    if !surf_contains(surface_id) {
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

// ---------------------------------------------------------------------------
// wgpu_shader_list / wgpu_shader_name / wgpu_last_error
// ---------------------------------------------------------------------------

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
    let count = shader_len();
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
    match shader_source(shader_id) {
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
