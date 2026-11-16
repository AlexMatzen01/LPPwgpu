# LPPwgpu

wgpu graphics package for LPPO. Enables WebGPU rendering from LPPO programs
via a C-ABI FFI layer (libloading).

## Installation

Add the repository to LPPO:

```sh
lppo pkg repo add LPPwgpu https://github.com/AlexMatzen01/LPPwgpu git
```

Install the package:

```sh
lppo pkg install lppwgpu
```

## Usage

```lppo
use lppwgpu;

// Initialize wgpu
wgpu_init();

// Create a window (800x600)
let win = wgpu_create_window(0, 800, 600);

// Create a surface for the window
let surf = wgpu_create_surface(win);

// Create a device + queue
let dev = wgpu_create_device();

// Configure the surface
wgpu_configure_surface(surf, dev, 800, 600);

// Create a shader from an embedded shader
let shader = wgpu_create_shader_embedded(dev, hash("basic"));

// Or load a shader from a file
let shader = wgpu_create_shader_file(dev, hash("shaders/my_shader.wgsl"));

// Create a render pipeline
let pipeline = wgpu_create_render_pipeline(dev, shader, 0, 0, 32);

// Main render loop
while true {
    wgpu_begin_render_pass(dev, surf, 0, 0, 0, 255);
    wgpu_set_pipeline(pipeline);
    wgpu_draw(0, 3, 0, 0);
    wgpu_end_render_pass();
    wgpu_submit(dev);
    wgpu_present(surf);
    wgpu_poll_events();
}
```

## Shader sources

WGSL shaders can be supplied two ways:

1. **Embedded** — Add `(&str, &str)` entries to the `EMBEDDED_SHADERS` array in
   `src/lib.rs` and reference them by name via `wgpu_create_shader_embedded`.
   Example shaders are provided in `examples/shaders/`.

2. **From file** — Place `.wgsl` files in `shaders/` (or the current directory)
   and load them via `wgpu_create_shader_file`, passing the FNV-1a hash of the
   file path.

## WebAssembly build

LPPwgpu compiles to `wasm32-unknown-unknown` and runs in the browser via WebGPU.
A pre-made `index.html` is included that loads `lppo_wgpu.wasm` and drives the
render loop.

### Build for wasm

```sh
cargo build --release --target wasm32-unknown-unknown --features wasm
```

This produces:

- `target/wasm32-unknown-unknown/release/lppo_wgpu.wasm`
- `index.html` (static web shell)

### Serve

```sh
cd target/wasm32-unknown-unknown/release
python3 -m http.server 8080
```

Then open `http://localhost:8080/index.html`.

### Ports

The package declares port `8080` in `package.json`. LPPO uses this when running
packages that expose network services.

## FFI convention

All exported functions follow the LPPO native calling convention:

```c
i64 func(i64 a0, i64 a1, i64 a2, i64 a3, i64 a4, i64 a5, i64 a6, i64 a7);
```

LPPO pads missing arguments with 0. `Str` arguments are hashed to `i64` using
FNV-1a (the same algorithm used in `index.html`).

## License

MIT
