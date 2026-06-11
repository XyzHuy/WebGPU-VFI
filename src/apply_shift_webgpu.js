const WORKGROUP_SIZE = 256;
const NUM_DIRECTIONS = 9;
const NUM_SCALES = 6;

const APPLY_SHIFT_WGSL = `
struct Params {
  batch: u32,
  channels: u32,
  height: u32,
  width: u32,
  weight_channels: u32,
  weight_height: u32,
  weight_width: u32,
  padding: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> img: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct BilinearCoords {
  y0: u32,
  x0: u32,
  y1: u32,
  x1: u32,
  fy: f32,
  fx: f32,
};

fn direction_delta(direction: u32, scale: u32) -> vec2<i32> {
  switch direction {
    case 0u: { return vec2<i32>(0, 0); }
    case 1u: { return vec2<i32>(0, -i32(scale)); }
    case 2u: { return vec2<i32>(0, i32(scale)); }
    case 3u: { return vec2<i32>(-i32(scale), 0); }
    case 4u: { return vec2<i32>(i32(scale), 0); }
    case 5u: { return vec2<i32>(-i32(scale), -i32(scale)); }
    case 6u: { return vec2<i32>(-i32(scale), i32(scale)); }
    case 7u: { return vec2<i32>(i32(scale), -i32(scale)); }
    default: { return vec2<i32>(i32(scale), i32(scale)); }
  }
}

fn weight_coords(y: u32, x: u32) -> BilinearCoords {
  let source_y = (f32(y) + 0.5) * f32(params.weight_height) / f32(params.height) - 0.5;
  let source_x = (f32(x) + 0.5) * f32(params.weight_width) / f32(params.width) - 0.5;
  let y0 = i32(floor(source_y));
  let x0 = i32(floor(source_x));
  let y1 = y0 + 1;
  let x1 = x0 + 1;
  let fy = source_y - f32(y0);
  let fx = source_x - f32(x0);
  let cy0 = u32(clamp(y0, 0, i32(params.weight_height) - 1));
  let cx0 = u32(clamp(x0, 0, i32(params.weight_width) - 1));
  let cy1 = u32(clamp(y1, 0, i32(params.weight_height) - 1));
  let cx1 = u32(clamp(x1, 0, i32(params.weight_width) - 1));
  return BilinearCoords(cy0, cx0, cy1, cx1, fy, fx);
}

fn sample_weight(b: u32, channel: u32, coords: BilinearCoords) -> f32 {
  let weight_plane = params.weight_height * params.weight_width;
  let base = (b * params.weight_channels + channel) * weight_plane;
  let top = mix(
    weights[base + coords.y0 * params.weight_width + coords.x0],
    weights[base + coords.y0 * params.weight_width + coords.x1],
    coords.fx,
  );
  let bottom = mix(
    weights[base + coords.y1 * params.weight_width + coords.x0],
    weights[base + coords.y1 * params.weight_width + coords.x1],
    coords.fx,
  );
  return mix(top, bottom, coords.fy);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.batch * params.height * params.width;
  let linear = gid.x;
  if linear >= total {
    return;
  }

  let width = params.width;
  let height = params.height;
  let channels = params.channels;
  let plane_size = height * width;
  let channel_stride = plane_size;
  let batch_stride = channels * plane_size;

  let x = linear % width;
  let y = (linear / width) % height;
  let b = linear / plane_size;
  let coords = weight_coords(y, x);

  var acc = vec3<f32>(0.0);
  for (var scale_index: u32 = 0u; scale_index < ${NUM_SCALES}u; scale_index++) {
    let scale = 1u << scale_index;
    for (var direction: u32 = 0u; direction < ${NUM_DIRECTIONS}u; direction++) {
      let delta = direction_delta(direction, scale);
      let src_y = i32(y) - delta.y;
      let src_x = i32(x) - delta.x;
      if src_y < 0 || src_y >= i32(height) || src_x < 0 || src_x >= i32(width) {
        continue;
      }

      let weight_channel = scale_index * ${NUM_DIRECTIONS}u + direction;
      let img_offset = b * batch_stride + u32(src_y) * width + u32(src_x);
      let rgb = vec3<f32>(
        f32(img[img_offset]),
        f32(img[img_offset + channel_stride]),
        f32(img[img_offset + 2u * channel_stride]),
      );
      acc += rgb * sample_weight(b, weight_channel, coords);
    }
  }

  let out_offset = b * batch_stride + y * width + x;
  let normalized = acc / ${NUM_SCALES}.0;
  out[out_offset] = normalized.x;
  out[out_offset + channel_stride] = normalized.y;
  out[out_offset + 2u * channel_stride] = normalized.z;
}
`;

function tensorByteLength(dims) {
  return dims.reduce((acc, dim) => acc * dim, 1) * 4;
}

function readTensorShape(tensor) {
  const [batch, channels, height, width] = tensor.dims;
  return { batch, channels, height, width };
}

export class ApplyShiftWebGpu {
  constructor(device, numScales = 6) {
    if (numScales !== NUM_SCALES) {
      throw new Error(`apply_shift shader expects ${NUM_SCALES} scales, got ${numScales}`);
    }
    this.device = device;
    this.numScales = numScales;
    this.pipeline = null;
    this.uniformBuffer = null;
    this.bindGroupLayout = null;
  }

  async initialize() {
    if (this.pipeline) {
      return;
    }

    const shaderModule = this.device.createShaderModule({
      label: "apply_shift",
      code: APPLY_SHIFT_WGSL,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: "apply_shift_layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: "apply_shift_pipeline_layout",
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = await this.device.createComputePipelineAsync({
      label: "apply_shift_pipeline",
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "apply_shift_uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  createDispatch(imgTensor, weightsTensor, outputLabel) {
    if (!this.pipeline) {
      throw new Error("ApplyShiftWebGpu.initialize() must be called first");
    }

    const imgShape = readTensorShape(imgTensor);
    const weightsShape = readTensorShape(weightsTensor);
    if (imgShape.batch !== weightsShape.batch) {
      throw new Error("apply_shift batch mismatch between image and weights");
    }
    if (imgShape.channels !== 3) {
      throw new Error(`apply_shift expects RGB input, got ${imgShape.channels} channels`);
    }

    const outputBytes = tensorByteLength(imgTensor.dims);
    const outputBuffer = this.device.createBuffer({
      label: outputLabel,
      size: outputBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const bindGroup = this.device.createBindGroup({
      label: "apply_shift_bind_group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: imgTensor.gpuBuffer } },
        { binding: 2, resource: { buffer: weightsTensor.gpuBuffer } },
        { binding: 3, resource: { buffer: outputBuffer } },
      ],
    });

    const total = imgShape.batch * imgShape.height * imgShape.width;
    const workgroups = Math.ceil(total / WORKGROUP_SIZE);

    return {
      bindGroup,
      workgroups,
      gpuBuffer: outputBuffer,
      dims: imgTensor.dims,
      byteLength: outputBytes,
    };
  }

  runPair(img0Tensor, weights0Tensor, img1Tensor, weights1Tensor) {
    const first = this.createDispatch(img0Tensor, weights0Tensor, "apply_shift_output_0");
    const second = this.createDispatch(img1Tensor, weights1Tensor, "apply_shift_output_1");
    const imgShape = readTensorShape(img0Tensor);
    const weightsShape = readTensorShape(weights0Tensor);

    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Uint32Array([
        imgShape.batch,
        imgShape.channels,
        imgShape.height,
        imgShape.width,
        weightsShape.channels,
        weightsShape.height,
        weightsShape.width,
        0,
      ]),
    );

    const encoder = this.device.createCommandEncoder({ label: "apply_shift_pair_encoder" });
    const pass = encoder.beginComputePass({ label: "apply_shift_pair_pass" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, first.bindGroup);
    pass.dispatchWorkgroups(first.workgroups);
    pass.setBindGroup(0, second.bindGroup);
    pass.dispatchWorkgroups(second.workgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    return [first, second];
  }

  dispose() {
    this.uniformBuffer?.destroy();
    this.uniformBuffer = null;
    this.pipeline = null;
    this.bindGroupLayout = null;
  }
}

export function createGpuTensor(
  ort,
  device,
  data,
  dims,
  label = "tensor",
) {
  const byteLength = tensorByteLength(dims);
  const gpuBuffer = device.createBuffer({
    label,
    size: byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(gpuBuffer, 0, data);
  return ort.Tensor.fromGpuBuffer(gpuBuffer, {
    dataType: "float32",
    dims,
    dispose: () => gpuBuffer.destroy(),
  });
}

export function wrapGpuBufferAsTensor(
  ort,
  gpuBuffer,
  dims,
  dispose,
) {
  return ort.Tensor.fromGpuBuffer(gpuBuffer, {
    dataType: "float32",
    dims,
    dispose,
  });
}

export async function tensorToCpuData(tensor, releaseGpuData = false) {
  if (tensor.location === "cpu" || tensor.location === "cpu-pinned") {
    return tensor.data;
  }
  return tensor.getData(releaseGpuData);
}

export async function copyGpuTensorToCpu(device, tensor) {
  if (tensor.location !== "gpu-buffer") {
    return tensor.data;
  }

  const byteLength = tensorByteLength(tensor.dims);
  const readback = device.createBuffer({
    label: "tensor_readback",
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "tensor_readback_encoder" });
    encoder.copyBufferToBuffer(tensor.gpuBuffer, 0, readback, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Float32Array(readback.getMappedRange().slice(0));
  } finally {
    if (readback.mapState === "mapped") {
      readback.unmap();
    }
    readback.destroy();
  }
}

export async function ensureGpuTensor(ort, device, tensor, label) {
  if (tensor.location === "gpu-buffer") {
    try {
      if (tensor.gpuBuffer) {
        return { tensor, dispose: null };
      }
    } catch {
      // Fall through and re-upload from CPU if ORT did not expose the buffer handle.
    }
    const data = await tensor.getData();
    const gpuTensor = createGpuTensor(ort, device, data, tensor.dims, label);
    return {
      tensor: gpuTensor,
      dispose: () => gpuTensor.dispose?.(),
    };
  }

  const gpuTensor = createGpuTensor(ort, device, tensor.data, tensor.dims, label);
  return {
    tensor: gpuTensor,
    dispose: () => gpuTensor.dispose?.(),
  };
}
