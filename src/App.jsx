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
  BadgeCheck,
  Cpu,
  Download,
  Film,
  Gauge,
  Loader2,
  MonitorCheck,
  Play,
  Settings2,
  Square,
  Upload,
} from "lucide-react";
import { copyGpuTensorToCpu } from "./apply_shift_webgpu.js";
import {
  FusedApplyShiftPipeline,
  probeFusedModels,
} from "./inference_pipeline.js";
import { createMp4CanvasEncoder } from "./video_encoder.js";
import "./styles.css";

const MODEL_SIZE = {
  width: 1280,
  height: 720,
};
const DEFAULT_VIDEO_MULTIPLIER = 4;
const VIDEO_RESIZE_MODE = "contain";
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
  const [videoFile, setVideoFile] = useState(null);
  const [selectedToyVideo, setSelectedToyVideo] = useState("");
  const [videoMultiplier, setVideoMultiplier] = useState(DEFAULT_VIDEO_MULTIPLIER);
  const [videoMetrics, setVideoMetrics] = useState(null);
  const [videoProgress, setVideoProgress] = useState(null);
  const [videoOutputBlob, setVideoOutputBlob] = useState(null);
  const [status, setStatus] = useState("Ready");
  const [busy, setBusy] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [compatibilityIssue, setCompatibilityIssue] = useState(null);

  const frame0CanvasRef = useRef(null);
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

  async function runVideoInference() {
    const toyVideo = TOY_VIDEOS.find((video) => video.name === selectedToyVideo);
    if (!videoFile && !toyVideo) {
      setStatus("Select a video");
      return;
    }

    const controller = new AbortController();
    videoAbortRef.current = controller;
    setBusy(true);
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
        audioTrack: source.audioTrack,
        signal: controller.signal,
      });
      setVideoMetrics((current) => ({
        ...current,
        audio: encoder.audio,
      }));
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
      const { blob: outputBlob, audio } = await encoder.finalize();
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
        audio,
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
          <div className="eyebrow">
            <Film size={15} />
            <span>WebGPU video interpolation</span>
          </div>
          <h1>Frame Interpolation WebGPU</h1>
          <p>Upscale source FPS directly in a desktop Chromium browser with the fixed 720p FP32 ONNX graph.</p>
        </div>
        <div className={`status-pill ${statusTone}`}>
          {busy ? <Loader2 className="spin" size={16} /> : <Cpu size={16} />}
          <span>{status}</span>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="section-title">
            <Upload size={16} />
            <span>Input</span>
          </div>

          <Field label="Sample video">
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
              <option value="">Use uploaded file</option>
              {TOY_VIDEOS.map((video) => (
                <option key={video.name} value={video.name}>
                  {video.label}
                </option>
              ))}
            </select>
          </Field>

          <FileDrop
            label={videoFile?.name || "Upload video"}
            accept="video/*"
            onChange={(file) => {
              setVideoFile(file);
              if (file) setSelectedToyVideo("");
            }}
          />

          <div className="section-title">
            <Settings2 size={16} />
            <span>Output</span>
          </div>

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

          <div className="format-lock">
            <MonitorCheck size={16} />
            <div>
              <strong>1280 x 720</strong>
              <span>16:9 contain, black padded when needed</span>
            </div>
          </div>

          <div className="section-title">
            <Gauge size={16} />
            <span>Progress</span>
          </div>

          <div className="metric-table">
            <Metric label="Model" value="720p fused" tail={metrics?.precision || "FP32"} />
            <Metric label="Adapter" value={metrics?.provider || "-"} tail="WebGPU" />
            <Metric label="Load" value={formatMs(metrics?.loadMs)} tail="startup" />
            <Metric label="Warmup" value={formatMs(metrics?.warmupMs)} tail="gpu" />
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
              label="Audio"
              value={videoMetrics?.audio?.label ?? (videoMetrics ? "Checking" : "-")}
              tail={
                videoMetrics?.audio?.packetCount
                  ? `${videoMetrics.audio.packetCount} packets`
                  : "MP4"
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
              Encodes H.264 MP4 in the browser and copies compatible input audio into
              the output. Files without a supported audio track are exported silent.
            </span>
          </div>

          {compatibilityIssue && (
            <GpuCompatibilityNotice issue={compatibilityIssue} />
          )}
        </aside>

        <section className="stage">
          <div className="frame-stage video-stage">
            {videoInputUrl && (
              <VideoPreview title="Input video" src={videoInputUrl} />
            )}
            <CanvasPreview
              title="Decoded source frame"
              canvasRef={frame0CanvasRef}
            />
            <CanvasPreview
              title={`Encoding preview x${videoMultiplier}`}
              canvasRef={outputCanvasRef}
            />
            {videoOutputUrl && (
              <VideoPreview title="Interpolated MP4" src={videoOutputUrl} />
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

function GpuCompatibilityNotice({ issue }) {
  const pageUrl = issue.pageUrl || "https://YOUR-VERCEL-APP.vercel.app";
  const adapterLabel = issue.adapterInfo ? formatAdapterLabel(issue.adapterInfo) : "";
  const environment = detectClientEnvironment();
  const launchGuide = getLaunchGuide(environment, pageUrl);

  return (
    <div className="compatibility-panel">
      <div className="section-title">
        <BadgeCheck size={16} />
        <span>Hardware GPU required</span>
      </div>
      <p>{issue.reason}</p>
      {adapterLabel && (
        <p className="compatibility-detail">Detected adapter: {adapterLabel}</p>
      )}
      <p className="compatibility-detail">
        Detected: {environment.browserLabel} on {environment.osLabel}
      </p>
      <ol>
        {launchGuide.steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ol>
      <span className="command-label">{launchGuide.commandLabel}</span>
      <code className="command-snippet">{launchGuide.command}</code>
      {issue.browserLabel && (
        <p className="compatibility-detail">Raw browser data: {issue.browserLabel}</p>
      )}
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

function detectClientEnvironment() {
  const brands = navigator.userAgentData?.brands ?? [];
  const brandNames = brands.map(({ brand }) => brand);
  const userAgent = navigator.userAgent || "";
  const platform = navigator.userAgentData?.platform || navigator.platform || "";
  const platformText = `${platform} ${userAgent}`.toLowerCase();
  const browser = detectBrowserKey(brandNames, userAgent);
  const os = detectOsKey(platformText);

  return {
    browser,
    browserLabel: browserLabels[browser] || "Chromium browser",
    os,
    osLabel: osLabels[os] || "desktop OS",
  };
}

function detectBrowserKey(brandNames, userAgent) {
  const brands = brandNames.join(" ").toLowerCase();
  if (brands.includes("brave") || /brave/i.test(userAgent)) return "brave";
  if (brands.includes("edge") || /edg\//i.test(userAgent)) return "edge";
  if (brands.includes("google chrome") || /chrome/i.test(userAgent)) return "chrome";
  if (brands.includes("chromium") || /chromium/i.test(userAgent)) return "chromium";
  return "chromium";
}

function detectOsKey(platformText) {
  if (platformText.includes("android")) return "android";
  if (platformText.includes("iphone") || platformText.includes("ipad")) return "ios";
  if (platformText.includes("win")) return "windows";
  if (platformText.includes("mac")) return "macos";
  if (platformText.includes("linux") || platformText.includes("x11")) return "linux";
  return "unknown";
}

const browserLabels = {
  brave: "Brave",
  chrome: "Google Chrome",
  chromium: "Chromium",
  edge: "Microsoft Edge",
};

const osLabels = {
  android: "Android",
  ios: "iOS",
  linux: "Linux",
  macos: "macOS",
  unknown: "unknown OS",
  windows: "Windows",
};

function getLaunchGuide(environment, pageUrl) {
  const gpuPage = browserGpuPage(environment.browser);
  const steps = [
    "Turn on hardware acceleration in the browser settings.",
    `Open ${gpuPage} and confirm WebGPU is using the NVIDIA adapter.`,
    "Restart the browser with the command below if it still selects software rendering.",
  ];

  if (environment.os === "linux") {
    return {
      steps,
      commandLabel: `Linux NVIDIA - ${environment.browserLabel}`,
      command: linuxLaunchCommand(environment.browser, pageUrl),
    };
  }

  if (environment.os === "windows") {
    return {
      steps,
      commandLabel: `Windows PowerShell - ${environment.browserLabel}`,
      command: windowsLaunchCommand(environment.browser, pageUrl),
    };
  }

  if (environment.os === "macos") {
    return {
      steps: [
        "This build currently requires an NVIDIA WebGPU adapter.",
        "macOS Apple GPUs can expose WebGPU, but this app will still reject them until the NVIDIA-only check is relaxed.",
        "Use a Linux or Windows NVIDIA machine for this model build.",
      ],
      commandLabel: "macOS note",
      command: "macOS Apple GPUs are not supported by the current NVIDIA-only adapter check.",
    };
  }

  return {
    steps: [
      "Use a desktop Chromium browser on Linux or Windows with an NVIDIA GPU.",
      `Turn on hardware acceleration and check ${gpuPage}.`,
      "For Linux NVIDIA, use the command below instead of launching the browser normally.",
    ],
    commandLabel: "Linux NVIDIA fallback",
    command: linuxLaunchCommand("chrome", pageUrl),
  };
}

function browserGpuPage(browser) {
  return {
    brave: "brave://gpu",
    chrome: "chrome://gpu",
    chromium: "chrome://gpu",
    edge: "edge://gpu",
  }[browser] || "chrome://gpu";
}

function linuxLaunchCommand(browser, pageUrl) {
  const bin = {
    brave: "brave-browser",
    chrome: "google-chrome",
    chromium: "chromium",
    edge: "microsoft-edge",
  }[browser] || "google-chrome";
  const profile = {
    brave: "vfi-webgpu-brave",
    chrome: "vfi-webgpu-chrome",
    chromium: "vfi-webgpu-chromium",
    edge: "vfi-webgpu-edge",
  }[browser] || "vfi-webgpu-chrome";

  return `APP_URL="${pageUrl}"
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"

__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only DRI_PRIME=1 \\
VK_DRIVER_FILES="$NVIDIA_ICD" VK_ICD_FILENAMES="$NVIDIA_ICD" \\
${bin} \\
  --user-data-dir=/tmp/${profile} \\
  --no-first-run \\
  --no-default-browser-check \\
  --enable-unsafe-webgpu \\
  --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan,WebGPUDeveloperFeatures \\
  --enable-dawn-features=allow_unsafe_apis \\
  --use-angle=vulkan \\
  --ignore-gpu-blocklist \\
  --disable-software-rasterizer \\
  --enable-gpu-rasterization \\
  --enable-zero-copy \\
  "$APP_URL"`;
}

function windowsLaunchCommand(browser, pageUrl) {
  const config = {
    brave: {
      exe: "$Env:ProgramFiles\\BraveSoftware\\Brave-Browser\\Application\\brave.exe",
      profile: "vfi-webgpu-brave",
    },
    chrome: {
      exe: "$Env:ProgramFiles\\Google\\Chrome\\Application\\chrome.exe",
      profile: "vfi-webgpu-chrome",
    },
    chromium: {
      exe: "$Env:LOCALAPPDATA\\Chromium\\Application\\chrome.exe",
      profile: "vfi-webgpu-chromium",
    },
    edge: {
      exe: "${Env:ProgramFiles(x86)}\\Microsoft\\Edge\\Application\\msedge.exe",
      profile: "vfi-webgpu-edge",
    },
  }[browser] || {
    exe: "$Env:ProgramFiles\\Google\\Chrome\\Application\\chrome.exe",
    profile: "vfi-webgpu-chrome",
  };

  return `$AppUrl = "${pageUrl}"
$Profile = "$Env:TEMP\\${config.profile}"
$Browser = "${config.exe}"
Start-Process $Browser -ArgumentList @(
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
)`;
}

async function createVideoFrameSource(source, canvas) {
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

    const audioTrack =
      (await videoTrack.getPrimaryPairableAudioTrack()) ??
      (await input.getPrimaryAudioTrack());
    const sink = new CanvasSink(videoTrack, {
      width: MODEL_SIZE.width,
      height: MODEL_SIZE.height,
      fit: VIDEO_RESIZE_MODE,
    });
    return {
      duration,
      fps: stats.averagePacketRate,
      frameCount: stats.packetCount,
      audioTrack,
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
