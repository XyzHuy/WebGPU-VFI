# Frame Interpolation WebGPU

Run frame and video interpolation directly in a Chromium browser with WebGPU.

This is the browser deployment of
[XyzHuy/UET-Deep-Learning-2025_Frame-Interpolation](https://github.com/XyzHuy/UET-Deep-Learning-2025_Frame-Interpolation).
It uses React, Vite, ONNX Runtime Web, custom WebGPU compute shaders, WebCodecs,
and in-browser MP4 encoding.

> This demo is optimized for convenience and portability, not for matching the
> PyTorch/CUDA inference speed of the original repository. Browser WebGPU,
> ONNX Runtime Web sessions, and FP32 convolution overhead make it slower than
> the native Torch path.

## Quick Start

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. For Vercel, deploy as a normal Vite static
site:

| Setting | Value |
| --- | --- |
| Framework | Vite |
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Environment Variables | none |
## Browser Requirements

Use a desktop Chromium-family browser:

- Google Chrome
- Microsoft Edge
- Brave
- Chromium

The app checks WebGPU before loading the model. It rejects software adapters
such as SwiftShader, llvmpipe, lavapipe, WARP, CPU fallback, and other
non-hardware paths.

If the page rejects your GPU:

1. Turn on browser hardware acceleration.
2. Open `chrome://gpu` and confirm WebGPU is using the real GPU.
3. Launch the browser with one of the commands below.

The commands include:

- `--no-first-run` to skip first-run sign-in/onboarding pages
- `--no-default-browser-check` to skip "make this default browser" prompts
- a fixed `--user-data-dir` so the browser remembers the skipped setup screen
- `--disable-software-rasterizer` so SwiftShader/software fallback is rejected

If a browser still shows one setup page, click the "stay signed out" or skip
option once. As long as you keep the same profile directory, it should not ask
again.

## Linux NVIDIA Commands
Chrome:

```bash

APP_URL="https://web-gpu-vfi.vercel.app/"
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only DRI_PRIME=1 \
VK_DRIVER_FILES="$NVIDIA_ICD" VK_ICD_FILENAMES="$NVIDIA_ICD" \
google-chrome \
  --user-data-dir=/tmp/vfi-webgpu-chrome \
  --no-first-run \
  --no-default-browser-check \
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

Chromium:

```bash

APP_URL="https://web-gpu-vfi.vercel.app/"
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only DRI_PRIME=1 \
VK_DRIVER_FILES="$NVIDIA_ICD" VK_ICD_FILENAMES="$NVIDIA_ICD" \
chromium \
  --user-data-dir=/tmp/vfi-webgpu-chromium \
  --no-first-run \
  --no-default-browser-check \
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

Microsoft Edge:

```bash

APP_URL="https://web-gpu-vfi.vercel.app/"
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only DRI_PRIME=1 \
VK_DRIVER_FILES="$NVIDIA_ICD" VK_ICD_FILENAMES="$NVIDIA_ICD" \
microsoft-edge \
  --user-data-dir=/tmp/vfi-webgpu-edge \
  --no-first-run \
  --no-default-browser-check \
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

Brave:

```bash
APP_URL="https://web-gpu-vfi.vercel.app/"
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only DRI_PRIME=1 \
VK_DRIVER_FILES="$NVIDIA_ICD" VK_ICD_FILENAMES="$NVIDIA_ICD" \
brave-browser \
  --user-data-dir=/tmp/vfi-webgpu-brave \
  --no-first-run \
  --no-default-browser-check \
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

If your NVIDIA ICD file is not at `/usr/share/vulkan/icd.d/nvidia_icd.json`,
update `NVIDIA_ICD` or remove the two `VK_*` variables.

## Windows Commands

Before launching, set the browser to use the discrete GPU in Windows Settings:

`Settings -> System -> Display -> Graphics`

Chrome:

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Browser = "$Env:ProgramFiles\Google\Chrome\Application\chrome.exe"
Start-Process $Browser -ArgumentList @(
  "--user-data-dir=$Env:TEMP\vfi-webgpu-chrome",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  $AppUrl
)
```

Microsoft Edge:

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Browser = "${Env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
Start-Process $Browser -ArgumentList @(
  "--user-data-dir=$Env:TEMP\vfi-webgpu-edge",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  $AppUrl
)
```

Brave:

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Browser = "$Env:ProgramFiles\BraveSoftware\Brave-Browser\Application\brave.exe"
Start-Process $Browser -ArgumentList @(
  "--user-data-dir=$Env:TEMP\vfi-webgpu-brave",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  $AppUrl
)
```

Chromium:

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Browser = "$Env:LOCALAPPDATA\Chromium\Application\chrome.exe"
Start-Process $Browser -ArgumentList @(
  "--user-data-dir=$Env:TEMP\vfi-webgpu-chromium",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  $AppUrl
)
```

If your browser is installed somewhere else, update `$Browser`.

## Model Files

The deployed app expects these files under `public/models`:

- `frame_interpolation_encoder_fp32.onnx`
- `frame_interpolation_motion_fp32.onnx`
- `frame_interpolation_stage2_fp32.onnx`
- `frame_interpolation_stage2_hotconv_upconv2_0.json`
- `frame_interpolation_stage2_hotconv_upconv2_0_pre.onnx`
- `frame_interpolation_stage2_hotconv_upconv2_0_post.onnx`
- `frame_interpolation_stage2_hotconv_upconv2_0_weight.bin`
- `frame_interpolation_stage2_hotconv_upconv2_0_bias.bin`
- `frame_interpolation_stage2_hotconv_upconv2_0_prelu.bin`

Generate them from the original repository:

```bash
python3 Export_ONNX.py --validate
```

`Export_ONNX.py` exports both the split ONNX pipeline and the default hybrid
hot Conv artifacts.

## What Runs In The Browser

1. Per-frame encoder ONNX caches reusable features.
2. Motion ONNX predicts shift weights and visibility.
3. A custom WebGPU shader performs directional `apply_shift` warping.
4. Stage 2 runs through the default hybrid hot Conv path.
5. Video mode decodes frames with WebCodecs and writes a silent H.264 MP4.

Open the app with `?profile=1` to enable WebGPU timestamp profiling. Turn this
off for normal latency measurements.
