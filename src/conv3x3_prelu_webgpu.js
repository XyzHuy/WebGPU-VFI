const WORKGROUP_SIZE = 256;

const CONV3X3_PRELU_WGSL = `
struct Params {
  batch: u32,
  in_channels: u32,
  out_channels: u32,
  in_height: u32,
  in_width: u32,
  out_height: u32,
  out_width: u32,
  stride: u32,
  pad: u32,
  out_channel_groups4: u32,
  total: u32,
  padding: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(2) var<storage, read> weight_tensor: array<f32>;
@group(0) @binding(3) var<storage, read> bias_tensor: array<f32>;
@group(0) @binding(4) var<storage, read> prelu_slope: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_tensor: array<f32>;

fn weight_at(out_channel: u32, in_channel: u32, ky: u32, kx: u32) -> f32 {
  if (out_channel >= params.out_channels) {
    return 0.0;
  }
  let kernel_offset = ky * 3u + kx;
  let index = ((out_channel * params.in_channels + in_channel) * 9u) + kernel_offset;
  return weight_tensor[index];
}

fn bias_at(out_channel: u32) -> f32 {
  if (out_channel >= params.out_channels) {
    return 0.0;
  }
  return bias_tensor[out_channel];
}

fn slope_at(out_channel: u32) -> f32 {
  if (out_channel >= params.out_channels) {
    return 0.0;
  }
  return prelu_slope[out_channel];
}

fn store_output(batch: u32, out_channel: u32, y: u32, x: u32, value: f32) {
  if (out_channel >= params.out_channels) {
    return;
  }
  let plane = params.out_height * params.out_width;
  let offset = (batch * params.out_channels + out_channel) * plane + y * params.out_width + x;
  output_tensor[offset] = value;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let linear = gid.x;
  if (linear >= params.total) {
    return;
  }

  let out_plane = params.out_height * params.out_width;
  let x = linear % params.out_width;
  let y = (linear / params.out_width) % params.out_height;
  let out_group = (linear / out_plane) % params.out_channel_groups4;
  let batch = linear / (out_plane * params.out_channel_groups4);
  let out_channel0 = out_group * 4u;

  var acc = vec4<f32>(
    bias_at(out_channel0),
    bias_at(out_channel0 + 1u),
    bias_at(out_channel0 + 2u),
    bias_at(out_channel0 + 3u),
  );

  for (var in_channel = 0u; in_channel < params.in_channels; in_channel++) {
    for (var ky = 0u; ky < 3u; ky++) {
      let in_y = i32(y * params.stride + ky) - i32(params.pad);
      if (in_y < 0 || in_y >= i32(params.in_height)) {
        continue;
      }
      for (var kx = 0u; kx < 3u; kx++) {
        let in_x = i32(x * params.stride + kx) - i32(params.pad);
        if (in_x < 0 || in_x >= i32(params.in_width)) {
          continue;
        }

        let in_plane = params.in_height * params.in_width;
        let input_offset =
          (batch * params.in_channels + in_channel) * in_plane +
          u32(in_y) * params.in_width +
          u32(in_x);
        let input_value = input_tensor[input_offset];
        let weights = vec4<f32>(
          weight_at(out_channel0, in_channel, ky, kx),
          weight_at(out_channel0 + 1u, in_channel, ky, kx),
          weight_at(out_channel0 + 2u, in_channel, ky, kx),
          weight_at(out_channel0 + 3u, in_channel, ky, kx),
        );
        acc += input_value * weights;
      }
    }
  }

  let slopes = vec4<f32>(
    slope_at(out_channel0),
    slope_at(out_channel0 + 1u),
    slope_at(out_channel0 + 2u),
    slope_at(out_channel0 + 3u),
  );
  let activated = select(acc * slopes, acc, acc >= vec4<f32>(0.0));

  store_output(batch, out_channel0, y, x, activated.x);
  store_output(batch, out_channel0 + 1u, y, x, activated.y);
  store_output(batch, out_channel0 + 2u, y, x, activated.z);
  store_output(batch, out_channel0 + 3u, y, x, activated.w);
}
`;

export class Conv3x3PreluWebGpu {
  constructor(device) {
    this.device = device;
    this.pipeline = null;
    this.bindGroupLayout = null;
    this.uniformBuffer = null;
  }

  async initialize() {
    if (this.pipeline) {
      return;
    }

    const shaderModule = this.device.createShaderModule({
      label: "conv3x3_prelu",
      code: CONV3X3_PRELU_WGSL,
    });

    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: "conv3x3_prelu_layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: "conv3x3_prelu_pipeline_layout",
      bindGroupLayouts: [this.bindGroupLayout],
    });

    this.pipeline = await this.device.createComputePipelineAsync({
      label: "conv3x3_prelu_pipeline",
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "conv3x3_prelu_uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  createOutputBuffer(shape, label = "conv3x3_prelu_output") {
    return this.device.createBuffer({
      label,
      size: outputElementCount(shape) * Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  run({ input, weight, bias, slope, output, shape, label = "conv3x3_prelu_pass" }) {
    if (!this.pipeline) {
      throw new Error("Conv3x3PreluWebGpu.initialize() must be called first");
    }
    validateShape(shape);

    const outGroups4 = Math.ceil(shape.outChannels / 4);
    const total = shape.batch * outGroups4 * shape.outHeight * shape.outWidth;
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      new Uint32Array([
        shape.batch,
        shape.inChannels,
        shape.outChannels,
        shape.inHeight,
        shape.inWidth,
        shape.outHeight,
        shape.outWidth,
        shape.stride,
        shape.pad,
        outGroups4,
        total,
        0,
      ]),
    );

    const bindGroup = this.device.createBindGroup({
      label: "conv3x3_prelu_bind_group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: weight } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: slope } },
        { binding: 5, resource: { buffer: output } },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: `${label}_encoder` });
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(total / WORKGROUP_SIZE));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  dispose() {
    this.uniformBuffer?.destroy();
    this.uniformBuffer = null;
    this.bindGroupLayout = null;
    this.pipeline = null;
  }
}

export function outputElementCount(shape) {
  return shape.batch * shape.outChannels * shape.outHeight * shape.outWidth;
}

export function createStorageBuffer(device, data, label) {
  const buffer = device.createBuffer({
    label,
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export function createEmptyStorageBuffer(device, elements, label) {
  return device.createBuffer({
    label,
    size: elements * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

function validateShape(shape) {
  const required = [
    "batch",
    "inChannels",
    "outChannels",
    "inHeight",
    "inWidth",
    "outHeight",
    "outWidth",
    "stride",
    "pad",
  ];
  for (const key of required) {
    if (!Number.isInteger(shape[key]) || shape[key] <= 0) {
      throw new Error(`Invalid Conv3x3 shape field '${key}': ${shape[key]}`);
    }
  }
}
