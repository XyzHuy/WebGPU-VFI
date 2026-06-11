# Frame Interpolation WebGPU

<p>
  <img alt="React" src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=111" />
  <img alt="Vite" src="https://img.shields.io/badge/Vite-6-646CFF?logo=vite&logoColor=fff" />
  <img alt="WebGPU" src="https://img.shields.io/badge/WebGPU-Chromium-4285F4?logo=googlechrome&logoColor=fff" />
  <img alt="ONNX Runtime" src="https://img.shields.io/badge/ONNX_Runtime-Web-005CED?logo=onnx&logoColor=fff" />
  <img alt="Video only" src="https://img.shields.io/badge/Inference-video_only-1f7a4d" />
</p>

Browser deployment for
[XyzHuy/UET-Deep-Learning-2025_Frame-Interpolation](https://github.com/XyzHuy/UET-Deep-Learning-2025_Frame-Interpolation).
The app runs video frame interpolation directly in a desktop Chromium browser
using WebGPU, ONNX Runtime Web, custom WebGPU compute shaders, WebCodecs, and
in-browser MP4 encoding.

The app interpolates input videos to a higher frame
rate, defaults to `x4`, and normalizes frames to the model's fixed `1280 x 720`
16:9 input using contain padding.

> This deployment is optimized for portability and a browser-based demo flow.
> It is not expected to match the PyTorch/CUDA speed of the original repository.
## Video Demo (Comparison)
### Original Video (6 FPS) - On One Puch Man

https://github.com/user-attachments/assets/4b13bd06-0757-44c1-aabf-ba682f54e1a7



### Interpolated Video (96 FPS) - On One Puch Man


https://github.com/user-attachments/assets/8073f3aa-6ff9-48ad-a7bb-818471eab434


## Browser And GPU Requirements

The app is designed for desktop Chromium-family browsers with a hardware NVIDIA
WebGPU adapter. It rejects software adapters such as SwiftShader, llvmpipe,
lavapipe, WARP, CPU fallback, and other non-hardware paths.

| Browser | Status |
| --- | --- |
| ![Google Chrome](https://img.shields.io/badge/Google_Chrome-4285F4?logo=googlechrome&logoColor=white) | Recommended |
| ![Microsoft Edge](https://img.shields.io/badge/Microsoft_Edge-0078D7?logo=microsoftedge&logoColor=white) | Supported |
| ![Brave](https://img.shields.io/badge/Brave-FB542B?logo=brave&logoColor=white) | Supported |
| ![Chromium](https://img.shields.io/badge/Chromium-4285F4?logo=chromium&logoColor=white) | Supported |

## Live app deploy in Vercel
https://web-gpu-vfi.vercel.app/

> **Note:** Desktop PCs with NVIDIA GPUs can usually run the demo directly. NVIDIA laptops may require launching the browser with the provided WebGPU/NVIDIA script. See [Launch Browser With NVIDIA WebGPU](#launch-browser-with-nvidia-webgpu).
## Launch Browser With NVIDIA WebGPU

Open the block for your OS/browser and copy the command into a terminal.

<details>
<summary><strong>Linux NVIDIA commands</strong></summary>

If your NVIDIA ICD file is not at `/usr/share/vulkan/icd.d/nvidia_icd.json`,
update `NVIDIA_ICD` in the command before running it.

#### Google Chrome

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

#### Microsoft Edge

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

#### Brave

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

#### Chromium

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

</details>

<details>
<summary><strong>Windows NVIDIA commands</strong></summary>

Open PowerShell and use the block for your browser.
#### Google Chrome

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Profile = "$Env:TEMP\vfi-webgpu-chrome"

Start-Process "chrome.exe" -ArgumentList @(
  "--user-data-dir=$Profile",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  "--force_high_performance_gpu",
  "--new-window",
  $AppUrl
)
```

#### Microsoft Edge

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Profile = "$Env:TEMP\vfi-webgpu-edge"

Start-Process "msedge.exe" -ArgumentList @(
  "--user-data-dir=$Profile",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  "--enable-features=UseSkiaRenderer",
  "--force_high_performance_gpu",
  "--new-window",
  $AppUrl
)
```

#### Brave

```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Profile = "$Env:TEMP\vfi-webgpu-brave"

Start-Process "brave.exe" -ArgumentList @(
  "--user-data-dir=$Profile",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  "--enable-features=UseSkiaRenderer",
  "--force_high_performance_gpu",
  "--new-window",
  $AppUrl
)
```

#### Chromium
It's still chrome.exe 
```powershell
$AppUrl = "https://web-gpu-vfi.vercel.app/"
$Profile = "$Env:TEMP\vfi-webgpu-chromium"

Start-Process "chrome.exe" -ArgumentList @(
  "--user-data-dir=$Profile",
  "--no-first-run",
  "--no-default-browser-check",
  "--enable-unsafe-webgpu",
  "--ignore-gpu-blocklist",
  "--disable-software-rasterizer",
  "--enable-gpu-rasterization",
  "--enable-features=UseSkiaRenderer",
  "--force_high_performance_gpu",
  "--new-window",
  $AppUrl
)
```

</details>

<details>
<summary><strong>Linux dev shortcut script</strong></summary>

For local development, the repository also includes a convenience wrapper around
the Linux NVIDIA flags:

```bash
./scripts/launch-chrome-nvidia.sh "http://localhost:5174/"
```

Use `VFI_WEBGPU_CHROME_BIN` to choose another Chromium-family browser:

| Browser | Command |
| --- | --- |
| Chrome | `./scripts/launch-chrome-nvidia.sh "http://localhost:5174/"` |
| Chromium | `VFI_WEBGPU_CHROME_BIN=chromium ./scripts/launch-chrome-nvidia.sh "http://localhost:5174/"` |
| Edge | `VFI_WEBGPU_CHROME_BIN=microsoft-edge ./scripts/launch-chrome-nvidia.sh "http://localhost:5174/"` |
| Brave | `VFI_WEBGPU_CHROME_BIN=brave-browser ./scripts/launch-chrome-nvidia.sh "http://localhost:5174/"` |

Optional environment variables:

| Variable | Default |
| --- | --- |
| `VFI_WEBGPU_CHROME_BIN` | `google-chrome` |
| `VFI_WEBGPU_CHROME_PROFILE` | `/tmp/vfi-webgpu-chrome-nvidia` |
| `VFI_WEBGPU_NVIDIA_ICD` | `/usr/share/vulkan/icd.d/nvidia_icd.json` |

</details>

### macOS
 Not yet support

## Local Development

Install dependencies and start Vite:

```bash
npm install
npm run dev
```

Then open the local URL printed by Vite with the same GPU-aware browser flow.
For the configured Vite port on Linux NVIDIA:

```bash
./scripts/launch-chrome-nvidia.sh "http://localhost:5174/"
```

If Vite chooses another port, pass that exact URL to the script.

Useful scripts:

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the local Vite dev server |
| `npm run build` | Build the static production app |
| `npm run preview` | Preview the production build locally |

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


## Technical Notes

- Input frames are resized to `1280 x 720` using 16:9 contain padding.
- Video interpolation defaults to `x4`; the UI also exposes `x2`, `x8`, and
  `x16`.
- ONNX Runtime Web runs the model pieces on WebGPU. 
- The app intentionally rejects software/fallback adapters 


