#!/usr/bin/env bash
set -euo pipefail

APP_URL="${1:-http://localhost:5174/}"
PROFILE_DIR="${VFI_WEBGPU_CHROME_PROFILE:-/tmp/vfi-webgpu-chrome-nvidia}"
NVIDIA_ICD="${VFI_WEBGPU_NVIDIA_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
CHROME_BIN="${VFI_WEBGPU_CHROME_BIN:-google-chrome}"

if ! command -v "${CHROME_BIN}" >/dev/null 2>&1; then
  echo "Browser not found: ${CHROME_BIN}" >&2
  echo "Set VFI_WEBGPU_CHROME_BIN=google-chrome-stable, chromium, microsoft-edge, or brave-browser." >&2
  exit 1
fi

mkdir -p "${PROFILE_DIR}"

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export DRI_PRIME=1
if [[ -f "${NVIDIA_ICD}" ]]; then
  export VK_DRIVER_FILES="${NVIDIA_ICD}"
  export VK_ICD_FILENAMES="${NVIDIA_ICD}"
fi

CHROME_FLAGS=(
  "--user-data-dir=${PROFILE_DIR}"
  "--enable-unsafe-webgpu"
  "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan,WebGPUDeveloperFeatures"
  "--enable-dawn-features=allow_unsafe_apis"
  "--use-angle=vulkan"
  "--ignore-gpu-blocklist"
  "--disable-software-rasterizer"
  "--enable-gpu-rasterization"
  "--enable-zero-copy"
)

exec "${CHROME_BIN}" "${CHROME_FLAGS[@]}" "${APP_URL}"
