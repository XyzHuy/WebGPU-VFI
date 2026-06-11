import * as ort from "onnxruntime-web/webgpu";
import {
  ApplyShiftWebGpu,
  createGpuTensor,
  ensureGpuTensor,
  wrapGpuBufferAsTensor,
} from "./apply_shift_webgpu.js";
import {
  Conv3x3PreluWebGpu,
  createEmptyStorageBuffer,
  createStorageBuffer,
  outputElementCount,
} from "./conv3x3_prelu_webgpu.js";

const FRAME_ENCODER_OUTPUTS = [
  "flow",
  "ctx_c1",
  "ctx_c2",
  "ctx_c3",
];

const MOTION_OUTPUTS = [
  "weights_fwd_shift",
  "weights_bwd_shift",
  "weights_fwd_refiner",
  "weights_bwd_refiner",
  "visibility_full",
  "visibility_refiner",
];

const gpuOutputPreferences = (names) =>
  Object.fromEntries(names.map((name) => [name, "gpu-buffer"]));

export const GPU_PROFILING_ENABLED =
  typeof window !== "undefined" &&
  new URLSearchParams(window.location.search).get("profile") === "1";

let gpuProfileEvents = [];
let collectGpuProfile = false;

if (GPU_PROFILING_ENABLED) {
  ort.env.webgpu.profiling = {
    mode: "default",
    ondata: (event) => {
      if (collectGpuProfile) {
        gpuProfileEvents.push(event);
      }
    },
  };
}

function createGpuSessionOptions(outputNames) {
  return {
    executionProviders: [{
      name: "webgpu",
      preferredLayout: "NCHW",
      validationMode: "wgpuOnly",
    }],
    graphOptimizationLevel: "all",
    preferredOutputLocation: gpuOutputPreferences(outputNames),
  };
}

function createStage2SessionOptions() {
  return {
    executionProviders: [{
      name: "webgpu",
      preferredLayout: "NCHW",
      validationMode: "wgpuOnly",
    }],
    graphOptimizationLevel: "all",
    preferredOutputLocation: { pred: "gpu-buffer" },
  };
}

function createHotConvPreSessionOptions(outputNames) {
  return {
    executionProviders: [{
      name: "webgpu",
      preferredLayout: "NCHW",
      validationMode: "wgpuOnly",
    }],
    graphOptimizationLevel: "all",
    preferredOutputLocation: gpuOutputPreferences(outputNames),
  };
}

function beginGpuProfile() {
  if (!GPU_PROFILING_ENABLED) {
    return;
  }
  gpuProfileEvents = [];
  collectGpuProfile = true;
}

async function finishGpuProfile(device) {
  if (!GPU_PROFILING_ENABLED) {
    return null;
  }

  await device.queue.onSubmittedWorkDone();
  await new Promise((resolve) => setTimeout(resolve, 0));
  collectGpuProfile = false;

  const grouped = new Map();
  let totalGpuMs = 0;
  for (const event of gpuProfileEvents) {
    const durationMs = (event.endTime - event.startTime) / 1e6;
    totalGpuMs += durationMs;
    const key = `${event.kernelType}:${event.programName}`;
    const current = grouped.get(key) ?? { name: key, ms: 0, calls: 0 };
    current.ms += durationMs;
    current.calls += 1;
    grouped.set(key, current);
  }

  const topKernels = [...grouped.values()]
    .sort((left, right) => right.ms - left.ms)
    .slice(0, 5);
  const profile = {
    kernelCount: gpuProfileEvents.length,
    totalGpuMs,
    topKernels,
  };
  console.table(topKernels);
  return profile;
}

function buildStage2Feed(inputNames, motion, frame0, frame1, warpedImg0, warpedImg1) {
  const feed = {
    img0: frame0.image,
    img1: frame1.image,
    warped_img0: warpedImg0,
    warped_img1: warpedImg1,
    ctx0_c1: frame0.features.ctx_c1,
    ctx0_c2: frame0.features.ctx_c2,
    ctx0_c3: frame0.features.ctx_c3,
    ctx1_c1: frame1.features.ctx_c1,
    ctx1_c2: frame1.features.ctx_c2,
    ctx1_c3: frame1.features.ctx_c3,
  };

  for (const name of inputNames) {
    if (name in feed) {
      continue;
    }
    if (name in motion) {
      feed[name] = motion[name];
      continue;
    }
    throw new Error(`Missing stage2 input mapping for '${name}'`);
  }

  return feed;
}

export class FusedApplyShiftPipeline {
  constructor({ encoderPath, motionPath, stage2Path, stage2HotConvPath = "", height, width, numScales = 6 }) {
    this.encoderPath = encoderPath;
    this.motionPath = motionPath;
    this.stage2Path = stage2Path;
    this.stage2HotConvPath = stage2HotConvPath;
    this.height = height;
    this.width = width;
    this.numScales = numScales;
    this.encoderSession = null;
    this.motionSession = null;
    this.stage2Session = null;
    this.stage2HotConv = null;
    this.applyShift = null;
    this.device = null;
    this.loadPromise = null;
    this.lastTimings = null;
  }

  async load(adapter) {
    if (this.stage2Session || this.stage2HotConv) {
      return;
    }
    if (this.loadPromise) {
      return this.loadPromise;
    }

    this.loadPromise = (async () => {
      ort.env.webgpu.adapter = adapter;

      // ORT only exposes env.webgpu.device after the first WebGPU session initializes JSEP.
      this.encoderSession = await ort.InferenceSession.create(
        this.encoderPath,
        createGpuSessionOptions(FRAME_ENCODER_OUTPUTS),
      );

      this.device = ort.env.webgpu.device;
      if (!this.device) {
        throw new Error("WebGPU device is not available after ORT session initialization");
      }
      this.applyShift = new ApplyShiftWebGpu(this.device, this.numScales);
      await this.applyShift.initialize();

      this.motionSession = await ort.InferenceSession.create(
        this.motionPath,
        createGpuSessionOptions(MOTION_OUTPUTS),
      );
      if (this.stage2HotConvPath) {
        try {
          this.stage2HotConv = new HotConvStage2(this.device, this.stage2HotConvPath);
          await this.stage2HotConv.load();
        } catch (error) {
          console.warn(`Hot Conv stage2 unavailable, falling back to ONNX stage2: ${error.message}`);
          await this.stage2HotConv?.release();
          this.stage2HotConv = null;
        }
      }
      if (!this.stage2HotConv) {
        this.stage2Session = await ort.InferenceSession.create(
          this.stage2Path,
          createStage2SessionOptions(),
        );
      }
    })();

    try {
      await this.loadPromise;
    } catch (error) {
      await this.release();
      throw error;
    }
  }

  async run(img0Data, img1Data) {
    if (!this.stage2Session && !this.stage2HotConv) {
      throw new Error("FusedApplyShiftPipeline.load() must be called before run()");
    }

    beginGpuProfile();
    let frame0 = null;
    let frame1 = null;
    try {
      frame0 = await this.encodeFrame(img0Data, "frame0");
      frame1 = await this.encodeFrame(img1Data, "frame1");
      const pred = await this.runEncodedPair(frame0, frame1);
      const encoderMs = frame0.timings.encoderMs + frame1.timings.encoderMs;
      this.lastTimings = {
        ...this.lastTimings,
        uploadMs: frame0.timings.uploadMs + frame1.timings.uploadMs,
        encoderMs,
        stage1Ms: encoderMs + this.lastTimings.motionMs,
      };
      return pred;
    } finally {
      frame0?.dispose();
      frame1?.dispose();
    }
  }

  async encodeFrame(data, label = "frame") {
    const dims = [1, 3, this.height, this.width];
    const uploadStarted = performance.now();
    const image = createGpuTensor(ort, this.device, data, dims, `${label}_gpu`);
    const uploadMs = performance.now() - uploadStarted;

    try {
      return await this.encodeTensor(image, { uploadMs, ownsImage: true });
    } catch (error) {
      image.dispose?.();
      throw error;
    }
  }

  async encodeTensor(image, { uploadMs = 0, ownsImage = true } = {}) {
    if (!this.encoderSession) {
      throw new Error("FusedApplyShiftPipeline.load() must be called before encodeTensor()");
    }

    const encoderStarted = performance.now();
    const results = await this.encoderSession.run({ img: image });
    const encoderMs = performance.now() - encoderStarted;
    const features = Object.fromEntries(
      FRAME_ENCODER_OUTPUTS.map((name) => [name, results[name]]),
    );
    let disposed = false;

    return {
      image,
      features,
      timings: { uploadMs, encoderMs },
      dispose() {
        if (disposed) return;
        disposed = true;
        for (const tensor of Object.values(features)) {
          tensor?.dispose?.();
        }
        if (ownsImage) {
          image?.dispose?.();
        }
      },
    };
  }

  async runEncodedPair(frame0, frame1) {
    if ((!this.stage2Session && !this.stage2HotConv) || !this.motionSession) {
      throw new Error("FusedApplyShiftPipeline.load() must be called before runEncodedPair()");
    }

    if (!collectGpuProfile) {
      beginGpuProfile();
    }
    let motionResults = null;
    let weightsFwd = null;
    let weightsBwd = null;
    let warpedImg0 = null;
    let warpedImg1 = null;

    try {
      const motionStarted = performance.now();
      motionResults = await this.motionSession.run({
        flow0: frame0.features.flow,
        flow1: frame1.features.flow,
      });
      const motionMs = performance.now() - motionStarted;
      const motion = Object.fromEntries(
        MOTION_OUTPUTS.map((name) => [name, motionResults[name]]),
      );

      weightsFwd = await ensureGpuTensor(
        ort,
        this.device,
        motion.weights_fwd_shift,
        "weights_fwd_shift_gpu",
      );
      weightsBwd = await ensureGpuTensor(
        ort,
        this.device,
        motion.weights_bwd_shift,
        "weights_bwd_shift_gpu",
      );

      const shiftStarted = performance.now();
      const [warped0, warped1] = this.applyShift.runPair(
        frame0.image,
        weightsFwd.tensor,
        frame1.image,
        weightsBwd.tensor,
      );
      if (GPU_PROFILING_ENABLED) {
        await this.device.queue.onSubmittedWorkDone();
      }
      const shiftMs = performance.now() - shiftStarted;

      warpedImg0 = wrapGpuBufferAsTensor(
        ort,
        warped0.gpuBuffer,
        warped0.dims,
        () => warped0.gpuBuffer.destroy(),
      );
      warpedImg1 = wrapGpuBufferAsTensor(
        ort,
        warped1.gpuBuffer,
        warped1.dims,
        () => warped1.gpuBuffer.destroy(),
      );

      const stage2Started = performance.now();
      const stage2Feed = buildStage2Feed(
        this.stage2HotConv?.inputNames ?? this.stage2Session.inputNames,
        motion,
        frame0,
        frame1,
        warpedImg0,
        warpedImg1,
      );
      const stage2Results = this.stage2HotConv
        ? await this.stage2HotConv.run(stage2Feed)
        : await this.stage2Session.run(stage2Feed);
      const stage2Ms = performance.now() - stage2Started;
      const stage2SplitTimings = this.stage2HotConv?.lastTimings ?? {};
      const gpuProfile = await finishGpuProfile(this.device);
      this.lastTimings = {
        uploadMs: 0,
        encoderMs: 0,
        motionMs,
        stage1Ms: motionMs,
        shiftMs,
        stage2Ms,
        stage2Runtime: this.stage2HotConv ? "hot-conv-split" : "ort",
        ...stage2SplitTimings,
        gpuProfile,
      };

      return stage2Results.pred;
    } finally {
      collectGpuProfile = false;
      warpedImg0?.dispose?.();
      warpedImg1?.dispose?.();
      weightsFwd?.dispose?.();
      weightsBwd?.dispose?.();
      for (const tensor of Object.values(motionResults ?? {})) {
        tensor?.dispose?.();
      }
    }
  }

  async generateIntermediateTensors(
    frame0,
    frame1,
    fpsMultiplier = 2,
    { signal, onGenerated } = {},
  ) {
    if (!Number.isInteger(fpsMultiplier) || fpsMultiplier < 2 ||
        (fpsMultiplier & (fpsMultiplier - 1)) !== 0) {
      throw new Error("fpsMultiplier must be a power of two greater than or equal to 2");
    }
    throwIfAborted(signal);

    const midTensor = await this.runEncodedPair(frame0, frame1);
    let midFrame = null;
    let left = [];
    let right = [];
    try {
      throwIfAborted(signal);
      await onGenerated?.();
      if (fpsMultiplier === 2) {
        return [midTensor];
      }

      // Keep the prediction tensor alive for output while caching its encoder features on GPU.
      midFrame = await this.encodeTensor(midTensor, { ownsImage: false });
      const half = fpsMultiplier / 2;
      left = await this.generateIntermediateTensors(
        frame0,
        midFrame,
        half,
        { signal, onGenerated },
      );
      right = await this.generateIntermediateTensors(
        midFrame,
        frame1,
        half,
        { signal, onGenerated },
      );
      return [...left, midTensor, ...right];
    } catch (error) {
      for (const tensor of [...left, ...right]) {
        tensor.dispose?.();
      }
      midTensor.dispose?.();
      throw error;
    } finally {
      midFrame?.dispose();
    }
  }

  async generateMidpointPreview(
    frame0,
    frame1,
    fpsMultiplier = 2,
    { signal, onGenerated } = {},
  ) {
    if (!Number.isInteger(fpsMultiplier) || fpsMultiplier < 2 ||
        (fpsMultiplier & (fpsMultiplier - 1)) !== 0) {
      throw new Error("fpsMultiplier must be a power of two greater than or equal to 2");
    }
    throwIfAborted(signal);
    const midTensor = await this.runEncodedPair(frame0, frame1);
    let midFrame = null;
    let leftPreview = null;
    let rightPreview = null;
    try {
      throwIfAborted(signal);
      await onGenerated?.();
      if (fpsMultiplier === 2) {
        return midTensor;
      }

      midFrame = await this.encodeTensor(midTensor, { ownsImage: false });
      const half = fpsMultiplier / 2;
      leftPreview = await this.generateMidpointPreview(
        frame0,
        midFrame,
        half,
        { signal, onGenerated },
      );
      leftPreview.dispose?.();
      leftPreview = null;
      rightPreview = await this.generateMidpointPreview(
        midFrame,
        frame1,
        half,
        { signal, onGenerated },
      );
      rightPreview.dispose?.();
      rightPreview = null;
      return midTensor;
    } catch (error) {
      midTensor.dispose?.();
      throw error;
    } finally {
      leftPreview?.dispose?.();
      rightPreview?.dispose?.();
      midFrame?.dispose();
    }
  }

  async streamIntermediateTensors(
    frame0,
    frame1,
    fpsMultiplier,
    onTensor,
    { signal, onGenerated } = {},
  ) {
    if (!Number.isInteger(fpsMultiplier) || fpsMultiplier < 2 ||
        (fpsMultiplier & (fpsMultiplier - 1)) !== 0) {
      throw new Error("fpsMultiplier must be a power of two greater than or equal to 2");
    }
    throwIfAborted(signal);

    const midTensor = await this.runEncodedPair(frame0, frame1);
    let midFrame = null;
    try {
      throwIfAborted(signal);
      await onGenerated?.();
      if (fpsMultiplier === 2) {
        await onTensor(midTensor);
        return;
      }

      midFrame = await this.encodeTensor(midTensor, { ownsImage: false });
      const half = fpsMultiplier / 2;
      await this.streamIntermediateTensors(
        frame0,
        midFrame,
        half,
        onTensor,
        { signal, onGenerated },
      );
      throwIfAborted(signal);
      await onTensor(midTensor);
      await this.streamIntermediateTensors(
        midFrame,
        frame1,
        half,
        onTensor,
        { signal, onGenerated },
      );
    } finally {
      midFrame?.dispose();
      midTensor.dispose?.();
    }
  }

  async runVideoSequence(
    frameDataIterable,
    fpsMultiplier,
    onFrame,
    { signal, onProgress } = {},
  ) {
    let previous = null;
    let intervalIndex = 0;
    let generatedFrames = 0;
    let outputFrames = 0;
    try {
      for await (const frameData of frameDataIterable) {
        throwIfAborted(signal);
        const current = await this.encodeFrame(frameData, `video_output_${intervalIndex + 1}`);
        try {
          if (previous) {
            await onFrame({
              tensor: previous.image,
              kind: "source",
              intervalIndex,
              outputIndex: outputFrames,
            });
            outputFrames += 1;
            await this.streamIntermediateTensors(
              previous,
              current,
              fpsMultiplier,
              async (tensor) => {
                await onFrame({
                  tensor,
                  kind: "intermediate",
                  intervalIndex,
                  outputIndex: outputFrames,
                });
                outputFrames += 1;
              },
              {
                signal,
                onGenerated: async () => {
                  generatedFrames += 1;
                  await onProgress?.({
                    intervalIndex,
                    generatedFrames,
                    outputFrames,
                  });
                },
              },
            );
            previous.dispose();
            intervalIndex += 1;
          }
          previous = current;
        } catch (error) {
          current.dispose();
          throw error;
        }
      }

      if (previous) {
        throwIfAborted(signal);
        await onFrame({
          tensor: previous.image,
          kind: "source",
          intervalIndex,
          outputIndex: outputFrames,
          final: true,
        });
        outputFrames += 1;
      }
      return { generatedFrames, outputFrames, completedPairs: intervalIndex };
    } finally {
      previous?.dispose();
    }
  }

  async runAdjacentSequence(
    frameDataIterable,
    fpsMultiplier,
    onInterval,
    { signal, onProgress, retainIntermediates = true } = {},
  ) {
    let previous = null;
    let intervalIndex = 0;
    let generatedFrames = 0;
    try {
      for await (const frameData of frameDataIterable) {
        throwIfAborted(signal);
        const current = await this.encodeFrame(frameData, `video_frame_${intervalIndex + 1}`);
        try {
          if (previous) {
            const generationOptions = {
              signal,
              onGenerated: async () => {
                generatedFrames += 1;
                await onProgress?.({
                  intervalIndex,
                  generatedFrames,
                });
              },
            };
            const intermediates = retainIntermediates
              ? await this.generateIntermediateTensors(
                  previous,
                  current,
                  fpsMultiplier,
                  generationOptions,
                )
              : [
                  await this.generateMidpointPreview(
                    previous,
                    current,
                    fpsMultiplier,
                    generationOptions,
                  ),
                ];
            try {
              await onInterval({
                index: intervalIndex,
                frame: previous.image,
                intermediates,
              });
            } finally {
              for (const tensor of intermediates) {
                tensor.dispose?.();
              }
            }
            previous.dispose();
            intervalIndex += 1;
          }
          previous = current;
        } catch (error) {
          current.dispose();
          throw error;
        }
      }

      if (previous) {
        throwIfAborted(signal);
        await onInterval({
          index: intervalIndex,
          frame: previous.image,
          intermediates: [],
          final: true,
        });
      }
    } finally {
      previous?.dispose();
    }
  }

  async release() {
    const sessions = [
      this.encoderSession,
      this.motionSession,
      this.stage2Session,
    ].filter(Boolean);
    this.encoderSession = null;
    this.motionSession = null;
    this.stage2Session = null;
    this.loadPromise = null;
    this.applyShift?.dispose();
    this.applyShift = null;
    await this.stage2HotConv?.release();
    this.stage2HotConv = null;
    this.device = null;
    await Promise.allSettled(sessions.map((session) => session.release()));
  }
}

class HotConvStage2 {
  constructor(device, manifestPath) {
    this.device = device;
    this.manifestPath = manifestPath;
    this.manifest = null;
    this.preSession = null;
    this.postSession = null;
    this.kernel = null;
    this.weightBuffer = null;
    this.biasBuffer = null;
    this.slopeBuffer = null;
    this.shape = null;
    this.inputNames = null;
    this.lastTimings = null;
  }

  async load() {
    const manifestUrl = new URL(this.manifestPath, window.location.href);
    const response = await fetch(manifestUrl);
    if (!response.ok) {
      throw new Error(`Missing hot Conv manifest: ${this.manifestPath}`);
    }
    this.manifest = await response.json();

    const preUrl = new URL(this.manifest.preModel, manifestUrl).pathname;
    const postUrl = new URL(this.manifest.postModel, manifestUrl).pathname;
    const preOutputs = [this.manifest.inputName, this.manifest.mergedName];
    this.preSession = await ort.InferenceSession.create(
      preUrl,
      createHotConvPreSessionOptions(preOutputs),
    );
    this.postSession = await ort.InferenceSession.create(
      postUrl,
      createStage2SessionOptions(),
    );

    this.kernel = new Conv3x3PreluWebGpu(this.device);
    await this.kernel.initialize();
    this.shape = hotConvShapeFromManifest(this.manifest);
    const [weight, bias, slope] = await Promise.all([
      fetchFloat32(new URL(this.manifest.weight, manifestUrl)),
      fetchFloat32(new URL(this.manifest.bias, manifestUrl)),
      fetchFloat32(new URL(this.manifest.prelu, manifestUrl)),
    ]);
    this.weightBuffer = createStorageBuffer(this.device, weight, "stage2_hotconv_weight");
    this.biasBuffer = createStorageBuffer(this.device, bias, "stage2_hotconv_bias");
    this.slopeBuffer = createStorageBuffer(this.device, slope, "stage2_hotconv_prelu");
    this.inputNames = this.preSession.inputNames;
  }

  async run(feed) {
    let preResults = null;
    let convInput = null;
    let customOutput = null;
    let outputTensor = null;
    try {
      const preStarted = performance.now();
      preResults = await this.preSession.run(feed);
      const stage2PreMs = performance.now() - preStarted;
      convInput = await ensureGpuTensor(
        ort,
        this.device,
        preResults[this.manifest.inputName],
        "stage2_hotconv_input",
      );
      customOutput = createEmptyStorageBuffer(
        this.device,
        outputElementCount(this.shape),
        "stage2_hotconv_output",
      );
      const hotConvStarted = performance.now();
      this.kernel.run({
        input: convInput.tensor.gpuBuffer,
        weight: this.weightBuffer,
        bias: this.biasBuffer,
        slope: this.slopeBuffer,
        output: customOutput,
        shape: this.shape,
        label: "stage2_hotconv_upconv2_0",
      });
      const stage2HotConvSubmitMs = performance.now() - hotConvStarted;
      const outputBuffer = customOutput;
      outputTensor = wrapGpuBufferAsTensor(
        ort,
        outputBuffer,
        this.manifest.outputShape,
        () => outputBuffer.destroy(),
      );
      customOutput = null;
      const postStarted = performance.now();
      const results = await this.postSession.run({
        [this.manifest.outputName]: outputTensor,
        [this.manifest.mergedName]: preResults[this.manifest.mergedName],
      });
      const stage2PostMs = performance.now() - postStarted;
      this.lastTimings = {
        stage2PreMs,
        stage2HotConvSubmitMs,
        stage2PostMs,
      };
      return results;
    } finally {
      outputTensor?.dispose?.();
      if (customOutput) {
        customOutput.destroy();
      }
      convInput?.dispose?.();
      for (const tensor of Object.values(preResults ?? {})) {
        tensor?.dispose?.();
      }
    }
  }

  async release() {
    this.weightBuffer?.destroy();
    this.biasBuffer?.destroy();
    this.slopeBuffer?.destroy();
    this.weightBuffer = null;
    this.biasBuffer = null;
    this.slopeBuffer = null;
    this.kernel?.dispose();
    this.kernel = null;
    const sessions = [this.preSession, this.postSession].filter(Boolean);
    this.preSession = null;
    this.postSession = null;
    await Promise.allSettled(sessions.map((session) => session.release()));
  }
}

function hotConvShapeFromManifest(manifest) {
  const input = manifest.inputShape;
  const output = manifest.outputShape;
  if (!input || !output) {
    throw new Error("Hot Conv manifest is missing input/output shapes");
  }
  return {
    batch: input[0],
    inChannels: input[1],
    outChannels: output[1],
    inHeight: input[2],
    inWidth: input[3],
    outHeight: output[2],
    outWidth: output[3],
    stride: manifest.stride?.[0] ?? 1,
    pad: manifest.pad?.[0] ?? 1,
  };
}

async function fetchFloat32(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url.pathname}: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  return new Float32Array(buffer);
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    throw new DOMException("Video inference cancelled", "AbortError");
  }
}

export async function probeModelUrl(url) {
  const response = await fetch(url, { method: "HEAD" });
  return response.ok;
}

export async function probeFusedModels(encoderPath, motionPath, stage2Path) {
  const [encoderReady, motionReady, stage2Ready] = await Promise.all([
    probeModelUrl(encoderPath),
    probeModelUrl(motionPath),
    probeModelUrl(stage2Path),
  ]);
  return {
    encoderReady,
    motionReady,
    stage2Ready,
    ready: encoderReady && motionReady && stage2Ready,
  };
}
