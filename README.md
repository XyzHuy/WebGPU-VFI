# VFI WebGPU Demo

Browser deployment for this frame interpolation model, built with React, Vite,
ONNX Runtime Web, WebGPU compute shaders, WebCodecs, and MP4 muxing in the
browser.

The app runs the model as a split WebGPU pipeline:

1. `frame_interpolation_encoder_fp32.onnx` encodes reusable per-frame features.
2. `frame_interpolation_motion_fp32.onnx` predicts shift weights and visibility.
3. `src/apply_shift_webgpu.js` runs the fused directional warp on WebGPU.
4. Stage 2 uses the default hybrid hot Conv path:
   `stage2_pre.onnx -> custom Conv3x3 + bias + PReLU -> stage2_post.onnx`.

This deployment is intended for interactive browser inference and Vercel-style
static hosting. It is not expected to match the inference time of the original
PyTorch/CUDA model in the repository. ONNX Runtime Web and browser WebGPU add
extra session, synchronization, memory, and shader-dispatch overhead, and the
refiner still performs large FP32 convolution work.

## Requirements

- Chrome or another browser with WebGPU enabled
- Node.js 18+
- Exported model files in `public/models`

The current browser path is FP32-only.

## GPU Compatibility

The app performs a browser-side compatibility gate before loading the model:

- requires a Chromium-family desktop browser
- requires a secure context (`https://` or `localhost`)
- requests a high-performance WebGPU adapter
- rejects fallback/software adapters such as SwiftShader, llvmpipe, lavapipe,
  WARP, and other CPU/software paths
- currently expects an NVIDIA adapter because this demo is tuned around that
  deployment target

If the page rejects the adapter, ask users to enable browser hardware
acceleration and inspect `chrome://gpu`. On Windows, also set the browser to
use the discrete GPU in Settings -> System -> Display -> Graphics or the NVIDIA
Control Panel.

Linux NVIDIA users can launch Chrome directly from a terminal:

```bash
APP_URL="https://web-gpu-vfi.vercel.app/"
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"

__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__VK_LAYER_NV_optimus=NVIDIA_only \
DRI_PRIME=1 \
VK_DRIVER_FILES="$NVIDIA_ICD" \
VK_ICD_FILENAMES="$NVIDIA_ICD" \
google-chrome \
  --user-data-dir=/tmp/vfi-webgpu-chrome-nvidia \
  --enable-unsafe-webgpu \
  --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan,WebGPUDeveloperFeatures \
  --enable-dawn-features=allow_unsafe_apis \
  --use-angle=vulkan \
  --ignore-gpu-blocklist \
  --disable-software-rasterizer \
  --enable-gpu-rasterization \
  --enable-zero-copy \
  "$APP_URL"
```

Replace `google-chrome` with `google-chrome-stable`, `chromium`,
`microsoft-edge`, or `brave-browser` if needed. If your NVIDIA ICD file lives
elsewhere, update `NVIDIA_ICD` or remove the two `VK_*` lines.

Windows users can launch Chrome from PowerShell:

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Chrome = "$Env:ProgramFiles\Google\Chrome\Application\chrome.exe"

Start-Process $Chrome -ArgumentList @(
  "--user-data-dir=$Env:TEMP\vfi-webgpu-chrome",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  $AppUrl
)
```

For Edge or Brave, replace `$Chrome` with the installed browser path.
Replace `https://YOUR-VERCEL-APP.vercel.app` with the real Vercel URL after
deployment.

## Export Models

From the repository root:

```bash
python3 Export_ONNX.py --validate
```

By default this exports fixed `1x3x720x1280` FP32 artifacts to
`public/models`, including the hybrid hot Conv Stage 2 files:

- `frame_interpolation_encoder_fp32.onnx`
- `frame_interpolation_motion_fp32.onnx`
- `frame_interpolation_stage2_fp32.onnx`
- `frame_interpolation_stage2_hotconv_upconv2_0.json`
- `frame_interpolation_stage2_hotconv_upconv2_0_pre.onnx`
- `frame_interpolation_stage2_hotconv_upconv2_0_post.onnx`
- `frame_interpolation_stage2_hotconv_upconv2_0_weight.bin`
- `frame_interpolation_stage2_hotconv_upconv2_0_bias.bin`
- `frame_interpolation_stage2_hotconv_upconv2_0_prelu.bin`

`Export_ONNX.py` owns the hot Conv split step; there is no separate split script
inside this WebGPU app.

## Run Locally

```bash
npm install
npm run dev
```

The app supports pair interpolation and video interpolation at x2, x4, x8, and
x16. Video mode decodes frames with WebCodecs and writes a silent H.264 MP4 in
the browser.

## Build

```bash
npm run build
```

The Vite output in `dist` can be served as static files. Keep the ONNX, binary
hot Conv artifacts, ORT WASM files, and demo videos under `public` so they are
copied into the build.

## Profiling

Open the app with `?profile=1` to enable WebGPU timestamp profiling. This is
useful for finding expensive kernels, but it should be disabled for normal
latency measurements.

## Project Files

- `src/App.jsx` - UI, model loading, pair/video workflows, progress and metrics
- `src/inference_pipeline.js` - encoder cache, motion graph, custom warp, Stage 2 orchestration
- `src/apply_shift_webgpu.js` - fused WebGPU directional warp shader
- `src/conv3x3_prelu_webgpu.js` - custom Stage 2 hot Conv shader
- `src/video_encoder.js` - browser H.264 encoding and MP4 muxing
- `scripts/analyze_stage2_onnx.py` - optional ONNX Conv cost analysis
