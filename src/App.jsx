import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as ort from "onnxruntime-web/webgpu";
import {
  ALL_FORMATS,
  BlobSource,
  CanvasSink,
  Input,
  UrlSource,
} from "mediabunny";
import {
  AlertCircle,
  Cpu,
  Download,
  Film,
  ImagePlus,
  Loader2,
  Play,
  Square,
  Upload,
} from "lucide-react";
import { copyGpuTensorToCpu, tensorToCpuData } from "./apply_shift_webgpu.js";
import {
  FusedApplyShiftPipeline,
  GPU_PROFILING_ENABLED,
  probeFusedModels,
} from "./inference_pipeline.js";
import { createMp4CanvasEncoder } from "./video_encoder.js";
import "./styles.css";

const MODEL_SIZE = {
  width: 1280,
  height: 720,
};
const ENCODER_PATH = "/models/frame_interpolation_encoder_fp32.onnx";
const MOTION_PATH = "/models/frame_interpolation_motion_fp32.onnx";
const STAGE2_PATH = "/models/frame_interpolation_stage2_fp32.onnx";
const STAGE2_HOTCONV_PATH = "/models/frame_interpolation_stage2_hotconv_upconv2_0.json";
const ORT_WASM_PATH = "/ort/";
const SOFTWARE_GPU_MARKERS = [
  "swiftshader",
  "llvmpipe",
  "lavapipe",
  "software",
  "fallback",
  "cpu",
  "mesa offscreen",
  "warp",
];
const TOY_VIDEOS = [
  {
    name: "one_punch_6fps.mp4",
    label: "One Punch - 1280x720, 6 FPS",
    url: "/toy-videos/one_punch_6fps.mp4",
  },
  {
    name: "rdr2_6fps.mp4",
    label: "RDR2 - 1920x1080, 6 FPS",
    url: "/toy-videos/rdr2_6fps.mp4",
  },
  {
    name: "wuwa_6fps.mp4",
    label: "Wuthering Waves - 1920x1080, 6 FPS",
    url: "/toy-videos/wuwa_6fps.mp4",
  },
];

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = ORT_WASM_PATH;
ort.env.logLevel = "warning";
ort.env.webgpu.powerPreference = "high-performance";
ort.env.webgpu.forceFallbackAdapter = false;

function App() {
  const [frame0File, setFrame0File] = useState(null);
  const [frame1File, setFrame1File] = useState(null);
  const [mode, setMode] = useState("pair");
  const [videoFile, setVideoFile] = useState(null);
  const [selectedToyVideo, setSelectedToyVideo] = useState("");
  const [videoMultiplier, setVideoMultiplier] = useState(2);
  const [videoMetrics, setVideoMetrics] = useState(null);
  const [videoProgress, setVideoProgress] = useState(null);
  const [videoOutputBlob, setVideoOutputBlob] = useState(null);
  const [status, setStatus] = useState("Ready");
  const [busy, setBusy] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [compatibilityIssue, setCompatibilityIssue] = useState(null);
  const [resizeMode, setResizeMode] = useState("contain");
  const [outputUrl, setOutputUrl] = useState("");

  const frame0CanvasRef = useRef(null);
  const frame1CanvasRef = useRef(null);
  const outputCanvasRef = useRef(null);
  const sessionPromiseRef = useRef(null);
  const runtimeRef = useRef(null);
  const warmupDoneRef = useRef(false);
  const videoAbortRef = useRef(null);
  const uploadedVideoUrl = useObjectUrl(videoFile);
  const videoOutputUrl = useObjectUrl(videoOutputBlob);
  const selectedToyVideoUrl =
    TOY_VIDEOS.find((video) => video.name === selectedToyVideo)?.url ?? "";
  const videoInputUrl = selectedToyVideoUrl || uploadedVideoUrl;

  const canRun = frame0File && frame1File && !busy;
  const canRunVideo = (videoFile || selectedToyVideo) && !busy;
  const statusTone = useMemo(() => {
    const lower = status.toLowerCase();
    if (lower.includes("failed") || lower.includes("unavailable") || lower.includes("error")) return "bad";
    if (lower.includes("ready") || lower.includes("done") || lower.includes("loaded")) return "good";
    return "neutral";
  }, [status]);

  useEffect(() => {
    let cancelled = false;
    let activeRuntime = null;

    setBusy(true);
    const startupPromise = loadSession()
      .then(async (runtime) => {
        activeRuntime = runtime;
        if (!cancelled) {
          await warmupSession(runtime);
        }
        return runtime;
      })
      .then((runtime) => {
        if (!cancelled) {
          setStatus("Fused apply_shift pipeline loaded (FP32)");
        }
        return runtime;
      })
      .catch((error) => {
        if (!cancelled) {
          setCompatibilityIssue(error.compatibility ?? null);
          setStatus(`Model load failed: ${error.message}`);
        }
        return activeRuntime;
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });

    return () => {
      cancelled = true;
      void startupPromise.then((runtime) => runtime?.pipeline.release());
    };
  }, []);

  async function loadSession() {
    if (runtimeRef.current) {
      return runtimeRef.current;
    }
    if (sessionPromiseRef.current) return sessionPromiseRef.current;

    sessionPromiseRef.current = (async () => {
      setCompatibilityIssue(null);
      setStatus("Checking WebGPU compatibility");
      const { adapter, label, info } = await requestCompatibleAdapter();
      ort.env.webgpu.adapter = adapter;

      setStatus(`Loading model on ${label}`);
      const started = performance.now();
      const runtime = await createRuntime(adapter);
      runtimeRef.current = runtime;
      setMetrics((current) => ({
        ...current,
        loadMs: performance.now() - started,
        provider: label,
        adapterInfo: info,
        precision: "FP32",
      }));
      return runtime;
    })();

    try {
      return await sessionPromiseRef.current;
    } catch (error) {
      sessionPromiseRef.current = null;
      throw error;
    }
  }

  async function runFrameInference() {
    if (!frame0File || !frame1File) {
      setStatus("Select two frames");
      return;
    }

    setBusy(true);
    setOutputUrl("");
    setMetrics((current) => ({
      loadMs: current?.loadMs,
      warmupMs: current?.warmupMs,
      provider: current?.provider,
      precision: current?.precision,
    }));
    setStatus("Preparing frames");

    let predTensor = null;
    try {
      const runtime = await loadSession();
      if (!warmupDoneRef.current) {
        await warmupSession(runtime);
      }
      const prepStarted = performance.now();
      const img0 = await imageFileToTensor(
        frame0File,
        frame0CanvasRef.current,
        resizeMode,
      );
      const img1 = await imageFileToTensor(
        frame1File,
        frame1CanvasRef.current,
        resizeMode,
      );
      const prepMs = performance.now() - prepStarted;

      setStatus("Running fused WebGPU inference");
      const inferenceStarted = performance.now();
      predTensor = await runInference(runtime, img0, img1);
      const inferenceMs = performance.now() - inferenceStarted;
      const pipelineTimings = runtime.pipeline.lastTimings;

      const drawStarted = performance.now();
      const predData = await readPrediction(runtime, predTensor);
      drawTensorToCanvas(predData, outputCanvasRef.current);
      const nextUrl = outputCanvasRef.current.toDataURL("image/png");
      setOutputUrl(nextUrl);
      setMetrics((current) => ({
        ...current,
        prepMs,
        inferenceMs,
        ...pipelineTimings,
        drawMs: performance.now() - drawStarted,
      }));
      setStatus("Frame inference done");
    } catch (error) {
      setStatus(`WebGPU inference failed: ${error.message}`);
    } finally {
      predTensor?.dispose?.();
      setBusy(false);
    }
  }

  async function runVideoInference() {
    const toyVideo = TOY_VIDEOS.find((video) => video.name === selectedToyVideo);
    if (!videoFile && !toyVideo) {
      setStatus("Select a video");
      return;
    }

    const controller = new AbortController();
    videoAbortRef.current = controller;
    setBusy(true);
    setOutputUrl("");
    setVideoMetrics(null);
    setVideoProgress(null);
    setVideoOutputBlob(null);
    setStatus("Reading video metadata");

    let source = null;
    let encoder = null;
    try {
      const runtime = await loadSession();
      if (!warmupDoneRef.current) {
        await warmupSession(runtime);
      }

      source = await createVideoFrameSource(
        toyVideo?.url || videoFile,
        frame0CanvasRef.current,
        resizeMode,
      );
      const availablePairs = Math.max(0, source.frameCount - 1);
      if (availablePairs === 0) {
        throw new Error("Video must contain at least two frames");
      }

      const requestedPairs = availablePairs;
      const sourceFrames = source.frameCount;
      const generatedTarget = availablePairs * (videoMultiplier - 1);
      const selectedOutputFrames = availablePairs * videoMultiplier + 1;
      const outputFps = source.fps * videoMultiplier;
      const estimatedIntervalMs = estimateIntervalMs(metrics, videoMultiplier);
      const started = performance.now();

      setVideoMetrics({
        durationSeconds: source.duration,
        sourceFps: source.fps,
        availablePairs,
        requestedPairs,
        sourceFrames,
        generatedFrames: 0,
        generatedTarget,
        completedPairs: 0,
        encodedFrames: 0,
        selectedOutputFrames,
        outputFps,
        elapsedMs: 0,
        etaMs: availablePairs * estimatedIntervalMs,
      });
      setVideoProgress({
        progress: 0,
        completed: 0,
        total: selectedOutputFrames,
        message: "Preparing MP4 encoder",
      });
      setStatus(`Video inference x${videoMultiplier}: 0/${requestedPairs} pairs`);

      encoder = await createMp4CanvasEncoder(outputCanvasRef.current, {
        fps: outputFps,
      });
      setVideoProgress((current) => ({
        ...current,
        message: "Interpolating and encoding",
      }));

      await runtime.pipeline.runVideoSequence(
        source.frames(),
        videoMultiplier,
        async ({ tensor, intervalIndex, outputIndex, final }) => {
          const frameData = await copyGpuTensorToCpu(runtime.pipeline.device, tensor);
          drawTensorToCanvas(frameData, outputCanvasRef.current);
          await encoder.addFrame();

          const elapsedMs = performance.now() - started;
          const encodedFrames = outputIndex + 1;
          const completedPairs = final
            ? requestedPairs
            : Math.min(requestedPairs, intervalIndex);
          setVideoMetrics((current) => ({
            ...current,
            completedPairs,
            encodedFrames,
            elapsedMs,
          }));
          setVideoProgress({
            progress: encodedFrames / selectedOutputFrames,
            completed: encodedFrames,
            total: selectedOutputFrames,
            message: "Interpolating and encoding",
          });
        },
        {
          signal: controller.signal,
          onProgress: ({ generatedFrames }) => {
            const elapsedMs = performance.now() - started;
            const averageGeneratedMs = elapsedMs / generatedFrames;
            setVideoMetrics((current) => ({
              ...current,
              generatedFrames,
              elapsedMs,
              etaMs: averageGeneratedMs * (generatedTarget - generatedFrames),
            }));
            const completedPairs = Math.floor(generatedFrames / (videoMultiplier - 1));
            setStatus(
              `Video inference x${videoMultiplier}: ${completedPairs}/${requestedPairs} pairs`,
            );
          },
        },
      );

      setVideoProgress({
        progress: 1,
        completed: selectedOutputFrames,
        total: selectedOutputFrames,
        message: "Finalizing MP4",
      });
      const outputBlob = await encoder.finalize();
      encoder = null;
      const elapsedMs = performance.now() - started;
      setVideoOutputBlob(outputBlob);
      setVideoMetrics((current) => ({
        ...current,
        completedPairs: requestedPairs,
        generatedFrames: generatedTarget,
        encodedFrames: selectedOutputFrames,
        elapsedMs,
        averagePairMs: elapsedMs / requestedPairs,
        etaMs: 0,
      }));
      setVideoProgress({
        progress: 1,
        completed: selectedOutputFrames,
        total: selectedOutputFrames,
        message: "MP4 ready",
      });
      setStatus(`Full interpolated MP4 ready (x${videoMultiplier})`);
    } catch (error) {
      await encoder?.cancel();
      if (error.name === "AbortError") {
        setStatus("Video inference stopped");
        setVideoProgress((current) => current && ({
          ...current,
          message: "Stopped",
        }));
      } else {
        setStatus(`Video inference failed: ${error.message}`);
        setVideoProgress((current) => current && ({
          ...current,
          message: "Failed",
        }));
      }
    } finally {
      source?.dispose();
      videoAbortRef.current = null;
      setBusy(false);
    }
  }

  function stopVideoInference() {
    videoAbortRef.current?.abort();
  }

  async function warmupSession(runtime) {
    if (warmupDoneRef.current) return;

    setStatus("Warming up WebGPU pipeline");
    const planeSize = MODEL_SIZE.width * MODEL_SIZE.height;
    const zeros = new Float32Array(planeSize * 3);
    const started = performance.now();
    const warmupOutput = await runInference(runtime, zeros, zeros);
    warmupOutput?.dispose?.();
    warmupDoneRef.current = true;
    setMetrics((current) => ({
      ...current,
      ...runtime.pipeline.lastTimings,
      warmupMs: performance.now() - started,
    }));
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Frame Interpolation WebGPU</h1>
          <p>Browser-only frame and video inference with the fixed 720p FP32 ONNX graph.</p>
        </div>
        <div className={`status-pill ${statusTone}`}>
          {busy ? <Loader2 className="spin" size={16} /> : <Cpu size={16} />}
          <span>{status}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="mode-switch">
            <button
              className={mode === "pair" ? "active" : ""}
              type="button"
              onClick={() => setMode("pair")}
              disabled={busy}
            >
              <ImagePlus size={15} />
              Frame pair
            </button>
            <button
              className={mode === "video" ? "active" : ""}
              type="button"
              onClick={() => setMode("video")}
              disabled={busy}
            >
              <Film size={15} />
              Video
            </button>
          </div>

          {mode === "pair" ? (
            <>
          <div className="section-title">
            <ImagePlus size={16} />
            <span>Frame pair</span>
          </div>

          <FileDrop label={frame0File?.name || "Frame 0"} accept="image/*" onChange={setFrame0File} />
          <FileDrop label={frame1File?.name || "Frame 1"} accept="image/*" onChange={setFrame1File} />

          <Field label="Resize">
            <select value={resizeMode} onChange={(event) => setResizeMode(event.target.value)}>
              <option value="contain">contain 16:9</option>
              <option value="cover">cover crop</option>
              <option value="stretch">stretch</option>
            </select>
          </Field>

          <div className="metric-table">
            <div className="metric-row static">
              <span>Model</span>
              <strong>720p fused</strong>
              <span>{metrics?.precision || "-"}</span>
            </div>
            <div className="metric-row static">
              <span>Input</span>
              <strong>1x3x720x1280</strong>
              <span>{metrics?.precision || "-"}</span>
            </div>
            <Metric label="Load" value={formatMs(metrics?.loadMs)} tail={metrics?.provider || "nvidia"} />
            <Metric label="Warmup" value={formatMs(metrics?.warmupMs)} tail="gpu" />
            <Metric label="Prep" value={formatMs(metrics?.prepMs)} tail="canvas" />
            <Metric label="Infer" value={formatMs(metrics?.inferenceMs)} tail="gpu" />
            <Metric label="Upload" value={formatMs(metrics?.uploadMs)} tail="gpu" />
            <Metric label="Encoder" value={formatMs(metrics?.encoderMs)} tail="ort" />
            <Metric label="Motion" value={formatMs(metrics?.motionMs)} tail="ort" />
            <Metric label="Encode + motion" value={formatMs(metrics?.stage1Ms)} tail="ort" />
            <Metric label="Shift" value={formatMs(metrics?.shiftMs)} tail="encode" />
            <Metric label="Stage 2" value={formatMs(metrics?.stage2Ms)} tail={metrics?.stage2Runtime || "ort"} />
            {metrics?.stage2PreMs !== undefined && (
              <>
                <Metric label="Stage2 pre" value={formatMs(metrics.stage2PreMs)} tail="ort" />
                <Metric
                  label="Hot conv"
                  value={formatMs(metrics.stage2HotConvSubmitMs)}
                  tail="submit"
                />
                <Metric label="Stage2 post" value={formatMs(metrics.stage2PostMs)} tail="ort+wait" />
              </>
            )}
            {GPU_PROFILING_ENABLED && (
              <>
                <Metric label="GPU total" value={formatMs(metrics?.gpuProfile?.totalGpuMs)} tail="timestamps" />
                <Metric
                  label="GPU kernels"
                  value={metrics?.gpuProfile?.kernelCount ?? "-"}
                  tail="dispatches"
                />
                {(metrics?.gpuProfile?.topKernels ?? []).slice(0, 3).map((kernel, index) => (
                  <Metric
                    key={kernel.name}
                    label={`Hot ${index + 1}`}
                    value={formatMs(kernel.ms)}
                    tail={`${shortKernelName(kernel.name)} x${kernel.calls}`}
                  />
                ))}
              </>
            )}
            <Metric label="Draw" value={formatMs(metrics?.drawMs)} tail="png" />
          </div>

          <button className="primary dark" onClick={runFrameInference} disabled={!canRun}>
            <Play size={16} />
            Run frame inference
          </button>

          {outputUrl && (
            <a className="download full" href={outputUrl} download="interpolated_frame.png">
              <Download size={16} />
              Download PNG
            </a>
          )}
            </>
          ) : (
            <>
              <div className="section-title">
                <Film size={16} />
                <span>Video interpolation output</span>
              </div>

              <Field label="Toy video">
                <select
                  value={selectedToyVideo}
                  onChange={(event) => {
                    const name = event.target.value;
                    setSelectedToyVideo(name);
                    if (name) {
                      setVideoFile(null);
                    }
                  }}
                >
                  <option value="">uploaded file</option>
                  {TOY_VIDEOS.map((video) => (
                    <option key={video.name} value={video.name}>
                      {video.label}
                    </option>
                  ))}
                </select>
              </Field>

              <FileDrop
                label={videoFile?.name || "Input video"}
                accept="video/*"
                onChange={(file) => {
                  setVideoFile(file);
                  if (file) setSelectedToyVideo("");
                }}
              />

              <Field label="Interpolation multiplier">
                <select
                  value={videoMultiplier}
                  onChange={(event) => setVideoMultiplier(Number(event.target.value))}
                >
                  {[2, 4, 8, 16].map((factor) => (
                    <option key={factor} value={factor}>x{factor}</option>
                  ))}
                </select>
              </Field>

              <Field label="Resize">
                <select value={resizeMode} onChange={(event) => setResizeMode(event.target.value)}>
                  <option value="contain">contain 16:9</option>
                  <option value="cover">cover crop</option>
                  <option value="stretch">stretch</option>
                </select>
              </Field>

              <div className="metric-table">
                <Metric
                  label="Duration"
                  value={formatDuration(videoMetrics?.durationSeconds * 1000)}
                  tail={
                    videoMetrics
                      ? `${formatFps(videoMetrics.sourceFps)} fps source`
                      : "native fps"
                  }
                />
                <Metric
                  label="Pairs"
                  value={
                    videoMetrics
                      ? `${videoMetrics.completedPairs}/${videoMetrics.requestedPairs}`
                      : "-"
                  }
                  tail={`${videoMetrics?.sourceFrames ?? "-"} source frames`}
                />
                <Metric
                  label="Generated"
                  value={
                    videoMetrics
                      ? `${videoMetrics.generatedFrames}/${videoMetrics.generatedTarget}`
                      : "-"
                  }
                  tail={`x${videoMultiplier}`}
                />
                <Metric
                  label="Encoded"
                  value={
                    videoMetrics
                      ? `${videoMetrics.encodedFrames}/${videoMetrics.selectedOutputFrames}`
                      : "-"
                  }
                  tail="MP4 frames"
                />
                <Metric
                  label="Output"
                  value={videoMetrics?.selectedOutputFrames ?? "-"}
                  tail={
                    videoMetrics
                      ? `${formatFps(videoMetrics.outputFps)} fps`
                      : `native x${videoMultiplier}`
                  }
                />
                <Metric
                  label="Elapsed"
                  value={formatDuration(videoMetrics?.elapsedMs)}
                  tail="end-to-end"
                />
                <Metric
                  label="Average"
                  value={formatDuration(videoMetrics?.averagePairMs)}
                  tail="per source pair"
                />
                <Metric
                  label="ETA"
                  value={formatDuration(videoMetrics?.etaMs)}
                  tail="full video"
                />
              </div>

              {videoProgress && (
                <div className="progress-box">
                  <div className="progress-meta">
                    <span>{videoProgress.message}</span>
                    <strong>{Math.round(videoProgress.progress * 100)}%</strong>
                  </div>
                  <div className="progress-track">
                    <div
                      style={{
                        width: `${Math.min(100, Math.max(0, videoProgress.progress * 100))}%`,
                      }}
                    />
                  </div>
                  <div className="progress-count">
                    <span>Output frames</span>
                    <span>{videoProgress.completed}/{videoProgress.total}</span>
                  </div>
                </div>
              )}

              {videoAbortRef.current ? (
                <button className="primary stop" onClick={stopVideoInference}>
                  <Square size={15} />
                  Stop
                </button>
              ) : (
                <button
                  className="primary dark"
                  onClick={runVideoInference}
                  disabled={!canRunVideo}
                >
                  <Play size={16} />
                  Run video inference
                </button>
              )}

              {videoOutputUrl && (
                <a
                  className="download full"
                  href={videoOutputUrl}
                  download={`interpolated_x${videoMultiplier}.mp4`}
                >
                  <Download size={16} />
                  Download MP4
                </a>
              )}

              <div className="hint">
                <AlertCircle size={15} />
                <span>
                  Encodes H.264 MP4 in the browser. Output is currently silent; source audio is
                  not copied.
                </span>
              </div>
            </>
          )}

          {compatibilityIssue && (
            <GpuCompatibilityNotice issue={compatibilityIssue} />
          )}

          {mode === "pair" && (
          <div className="hint">
            <AlertCircle size={15} />
            <span>Use Chromium with hardware acceleration. The app rejects SwiftShader and other software adapters.</span>
          </div>
          )}
        </aside>

        <section className="stage">
          <div className={mode === "video" ? "frame-stage video-stage" : "frame-stage"}>
            {mode === "video" && videoInputUrl && (
              <VideoPreview title="Input video" src={videoInputUrl} />
            )}
            <CanvasPreview
              title={mode === "video" ? "Decoded source frame" : "Frame 0"}
              canvasRef={frame0CanvasRef}
            />
            <CanvasPreview
              title={mode === "video" ? `Encoding preview x${videoMultiplier}` : "Interpolated"}
              canvasRef={outputCanvasRef}
            />
            {mode === "video" && videoOutputUrl && (
              <VideoPreview title="Interpolated MP4" src={videoOutputUrl} />
            )}
            {mode === "pair" && (
              <CanvasPreview title="Frame 1" canvasRef={frame1CanvasRef} />
            )}
          </div>
        </section>
      </section>
    </main>
  );
}

async function createRuntime(adapter) {
  const availability = await probeFusedModels(ENCODER_PATH, MOTION_PATH, STAGE2_PATH);
  if (!availability.ready) {
    const missing = [
      !availability.encoderReady ? ENCODER_PATH : null,
      !availability.motionReady ? MOTION_PATH : null,
      !availability.stage2Ready ? STAGE2_PATH : null,
    ].filter(Boolean);
    throw new Error(`Missing split ONNX model: ${missing.join(", ")}`);
  }

  const pipeline = new FusedApplyShiftPipeline({
    encoderPath: ENCODER_PATH,
    motionPath: MOTION_PATH,
    stage2Path: STAGE2_PATH,
    stage2HotConvPath: STAGE2_HOTCONV_PATH,
    height: MODEL_SIZE.height,
    width: MODEL_SIZE.width,
  });
  await pipeline.load(adapter);
  return { pipeline };
}

async function runInference(runtime, img0, img1) {
  return runtime.pipeline.run(img0, img1);
}

async function readPrediction(runtime, predTensor) {
  if (predTensor.location === "gpu-buffer") {
    return copyGpuTensorToCpu(runtime.pipeline.device, predTensor);
  }
  return tensorToCpuData(predTensor, true);
}

function GpuCompatibilityNotice({ issue }) {
  const pageUrl = issue.pageUrl || "https://YOUR-VERCEL-APP.vercel.app";
  const adapterLabel = issue.adapterInfo ? formatAdapterLabel(issue.adapterInfo) : "";
  const linuxCommand = `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only DRI_PRIME=1 google-chrome --enable-unsafe-webgpu --ignore-gpu-blocklist --disable-software-rasterizer --use-angle=vulkan "${pageUrl}"`;
  const windowsCommand = `Start-Process "$Env:ProgramFiles\\Google\\Chrome\\Application\\chrome.exe" -ArgumentList "--enable-unsafe-webgpu --ignore-gpu-blocklist --disable-software-rasterizer ${pageUrl}"`;
  return (
    <div className="compatibility-panel">
      <div className="section-title">
        <AlertCircle size={16} />
        <span>GPU required</span>
      </div>
      <p>{issue.reason}</p>
      {adapterLabel && (
        <p className="compatibility-detail">Detected adapter: {adapterLabel}</p>
      )}
      <p className="compatibility-detail">Detected browser: {issue.browserLabel}</p>
      <ol>
        <li>Open the app in Chrome, Edge, Brave, or Chromium on desktop.</li>
        <li>Enable browser hardware acceleration and check <code>chrome://gpu</code>.</li>
        <li>Launch the browser with hardware WebGPU flags if the adapter is still rejected.</li>
      </ol>
      <span className="command-label">Linux NVIDIA</span>
      <code className="command-snippet">{linuxCommand}</code>
      <span className="command-label">Windows PowerShell</span>
      <code className="command-snippet">{windowsCommand}</code>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function FileDrop({ label, accept, onChange }) {
  return (
    <label className="file-drop">
      <Upload size={17} />
      <span>{label}</span>
      <input type="file" accept={accept} onChange={(event) => onChange(event.target.files?.[0] ?? null)} />
    </label>
  );
}

function CanvasPreview({ title, canvasRef }) {
  return (
    <div className="preview">
      <div className="preview-title">{title}</div>
      <canvas ref={canvasRef} width={MODEL_SIZE.width} height={MODEL_SIZE.height} />
    </div>
  );
}

function VideoPreview({ title, src }) {
  return (
    <div className="preview">
      <div className="preview-title">{title}</div>
      <video src={src} controls playsInline preload="metadata" />
    </div>
  );
}

function Metric({ label, value, tail }) {
  return (
    <div className="metric-row static">
      <span>{label}</span>
      <strong>{value}</strong>
      <span>{tail}</span>
    </div>
  );
}

function makeCompatibilityIssue(reason, details = {}) {
  return {
    reason,
    browserLabel: detectBrowserLabel(),
    pageUrl: typeof window !== "undefined" ? window.location.href : "",
    ...details,
  };
}

function throwCompatibility(reason, details = {}) {
  const error = new Error(reason);
  error.compatibility = makeCompatibilityIssue(reason, details);
  throw error;
}

async function requestCompatibleAdapter() {
  if (!isChromiumLikeBrowser()) {
    throwCompatibility("Use a Chromium browser such as Chrome, Edge, Brave, or Chromium.");
  }

  if (!window.isSecureContext) {
    throwCompatibility("WebGPU requires HTTPS or localhost.");
  }

  if (!("gpu" in navigator)) {
    throwCompatibility("WebGPU is unavailable in this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
    forceFallbackAdapter: false,
  });

  if (!adapter) {
    throwCompatibility("No high-performance WebGPU adapter found.");
  }

  const info = await readAdapterInfo(adapter);
  const label = formatAdapterLabel(info);
  const lower = label.toLowerCase();

  if (info.isFallbackAdapter) {
    throwCompatibility(`Rejected fallback WebGPU adapter: ${label}.`, { adapterInfo: info });
  }

  if (isSoftwareAdapter(lower)) {
    throwCompatibility(`Rejected software WebGPU adapter: ${label}.`, { adapterInfo: info });
  }

  if (!isNvidiaAdapter(lower)) {
    throwCompatibility(`Expected NVIDIA WebGPU adapter, got: ${label}.`, { adapterInfo: info });
  }

  return { adapter, label, info };
}

async function readAdapterInfo(adapter) {
  if (adapter.info) return adapter.info;
  if (typeof adapter.requestAdapterInfo === "function") return adapter.requestAdapterInfo();
  return {};
}

function formatAdapterLabel(info) {
  const parts = [info.vendor, info.architecture, info.device, info.description]
    .map((part) => String(part || "").trim())
    .filter(Boolean);
  return parts.length > 0 ? parts.join(" / ") : "unknown adapter";
}

function isSoftwareAdapter(label) {
  return SOFTWARE_GPU_MARKERS.some((needle) => label.includes(needle));
}

function isNvidiaAdapter(label) {
  return label.includes("nvidia") || label.includes("10de") || label.includes("0x10de");
}

function isChromiumLikeBrowser() {
  const brands = navigator.userAgentData?.brands ?? [];
  if (brands.some(({ brand }) => /chromium|chrome|edge|brave|opera/i.test(brand))) {
    return true;
  }
  return /Chrome|Chromium|Edg|OPR|Brave/i.test(navigator.userAgent);
}

function detectBrowserLabel() {
  const brands = navigator.userAgentData?.brands
    ?.map(({ brand, version }) => `${brand} ${version}`)
    .join(", ");
  if (brands) return brands;
  return navigator.userAgent;
}

async function imageFileToTensor(file, canvas, resizeMode) {
  const bitmap = await createImageBitmap(file);
  drawBitmapToCanvas(bitmap, canvas, resizeMode);
  bitmap.close?.();
  return canvasToNchw(canvas);
}

async function createVideoFrameSource(source, canvas, resizeMode) {
  const input = new Input({
    source: typeof source === "string" ? new UrlSource(source) : new BlobSource(source),
    formats: ALL_FORMATS,
  });
  try {
    const videoTrack = await input.getPrimaryVideoTrack();
    if (!videoTrack) throw new Error("Input does not contain a video track");
    if (!(await videoTrack.canDecode())) {
      throw new Error("Browser cannot decode this video codec");
    }

    const [duration, stats] = await Promise.all([
      videoTrack.computeDuration(),
      videoTrack.computePacketStats(),
    ]);
    if (!Number.isFinite(duration) || duration <= 0) {
      throw new Error("Could not determine the video duration");
    }
    if (!Number.isFinite(stats.averagePacketRate) || stats.averagePacketRate <= 0) {
      throw new Error("Could not determine the source frame rate");
    }
    if (!Number.isInteger(stats.packetCount) || stats.packetCount <= 0) {
      throw new Error("Input video contains no decodable frames");
    }

    const sink = new CanvasSink(videoTrack, {
      width: MODEL_SIZE.width,
      height: MODEL_SIZE.height,
      fit: resizeMode === "stretch" ? "fill" : resizeMode,
    });
    return {
      duration,
      fps: stats.averagePacketRate,
      frameCount: stats.packetCount,
      async *frames() {
        let decodedFrames = 0;
        for await (const decoded of sink.canvases()) {
          drawBitmapToCanvas(decoded.canvas, canvas, "stretch");
          decodedFrames += 1;
          yield canvasToNchw(canvas);
        }
        if (decodedFrames !== stats.packetCount) {
          throw new Error(
            `Decoded ${decodedFrames} frames, but the container reports ${stats.packetCount}`,
          );
        }
      },
      dispose() {
        input.dispose();
      },
    };
  } catch (error) {
    input.dispose();
    throw error;
  }
}

function drawBitmapToCanvas(bitmap, canvas, resizeMode) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, MODEL_SIZE.width, MODEL_SIZE.height);
  ctx.fillStyle = "rgb(0, 0, 0)";
  ctx.fillRect(0, 0, MODEL_SIZE.width, MODEL_SIZE.height);

  const sourceWidth = bitmap.videoWidth || bitmap.width;
  const sourceHeight = bitmap.videoHeight || bitmap.height;
  if (resizeMode === "stretch") {
    ctx.drawImage(bitmap, 0, 0, MODEL_SIZE.width, MODEL_SIZE.height);
    return;
  }

  const imageRatio = sourceWidth / sourceHeight;
  const targetRatio = MODEL_SIZE.width / MODEL_SIZE.height;
  const scale =
    resizeMode === "cover"
      ? Math.max(MODEL_SIZE.width / sourceWidth, MODEL_SIZE.height / sourceHeight)
      : Math.min(MODEL_SIZE.width / sourceWidth, MODEL_SIZE.height / sourceHeight);
  const drawWidth = sourceWidth * scale;
  const drawHeight = sourceHeight * scale;
  const dx = (MODEL_SIZE.width - drawWidth) / 2;
  const dy = (MODEL_SIZE.height - drawHeight) / 2;

  if (Number.isFinite(imageRatio) && Number.isFinite(targetRatio)) {
    ctx.drawImage(bitmap, dx, dy, drawWidth, drawHeight);
  }
}

function canvasToNchw(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const image = ctx.getImageData(0, 0, MODEL_SIZE.width, MODEL_SIZE.height).data;
  const planeSize = MODEL_SIZE.width * MODEL_SIZE.height;
  const tensor = new Float32Array(planeSize * 3);

  for (let i = 0, pixel = 0; pixel < planeSize; pixel += 1, i += 4) {
    tensor[pixel] = image[i] / 255;
    tensor[planeSize + pixel] = image[i + 1] / 255;
    tensor[planeSize * 2 + pixel] = image[i + 2] / 255;
  }

  return tensor;
}

function drawTensorToCanvas(data, canvas) {
  const ctx = canvas.getContext("2d");
  const planeSize = MODEL_SIZE.width * MODEL_SIZE.height;
  const image = ctx.createImageData(MODEL_SIZE.width, MODEL_SIZE.height);

  for (let pixel = 0, i = 0; pixel < planeSize; pixel += 1, i += 4) {
    image.data[i] = floatToByte(data[pixel]);
    image.data[i + 1] = floatToByte(data[planeSize + pixel]);
    image.data[i + 2] = floatToByte(data[planeSize * 2 + pixel]);
    image.data[i + 3] = 255;
  }

  ctx.putImageData(image, 0, 0);
}

function floatToByte(value) {
  return Math.max(0, Math.min(255, Math.round(value * 255)));
}

function formatMs(value) {
  if (!Number.isFinite(value)) return "-";
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${value.toFixed(0)}ms`;
}

function formatDuration(value) {
  if (!Number.isFinite(value)) return "-";
  if (value < 1000) return `${Math.round(value)}ms`;
  const totalSeconds = Math.max(0, Math.round(value / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

function formatFps(value) {
  if (!Number.isFinite(value)) return "-";
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function estimateIntervalMs(pipelineMetrics, multiplier) {
  const encoderPerFrame = pipelineMetrics?.encoderMs
    ? pipelineMetrics.encoderMs / 2
    : 410;
  const pairHeadAndRefiner =
    (pipelineMetrics?.motionMs ?? 40) +
    (pipelineMetrics?.shiftMs ?? 1) +
    (pipelineMetrics?.stage2Ms ?? 820);
  return (multiplier / 2) * encoderPerFrame + (multiplier - 1) * pairHeadAndRefiner;
}

function shortKernelName(name) {
  const parts = String(name).split(":");
  return parts.at(-1) || name;
}

function useObjectUrl(blob) {
  const [url, setUrl] = useState("");

  useEffect(() => {
    if (!blob) {
      setUrl("");
      return undefined;
    }
    const nextUrl = URL.createObjectURL(blob);
    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [blob]);

  return url;
}

createRoot(document.getElementById("root")).render(<App />);
