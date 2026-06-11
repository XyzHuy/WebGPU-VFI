import {
  BufferTarget,
  CanvasSource,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
} from "mediabunny";

export async function createMp4CanvasEncoder(canvas, { fps }) {
  if (!("VideoEncoder" in window)) {
    throw new Error("WebCodecs VideoEncoder is unavailable in this browser");
  }
  if (!Number.isFinite(fps) || fps <= 0) {
    throw new Error(`Invalid output FPS: ${fps}`);
  }

  const target = new BufferTarget();
  const output = new Output({
    format: new Mp4OutputFormat({ fastStart: "in-memory" }),
    target,
  });
  const source = new CanvasSource(canvas, {
    codec: "avc",
    fullCodecString: "avc1.42001f",
    bitrate: QUALITY_HIGH,
    keyFrameInterval: 2,
    latencyMode: "quality",
    hardwareAcceleration: "no-preference",
  });
  output.addVideoTrack(source);
  await output.start();

  const frameDuration = 1 / fps;
  let frameIndex = 0;
  let closed = false;

  return {
    get frameCount() {
      return frameIndex;
    },
    async addFrame() {
      if (closed) throw new Error("MP4 encoder is already closed");
      const timestamp = frameIndex * frameDuration;
      const keyFrame = frameIndex % Math.max(1, Math.round(fps * 2)) === 0;
      await source.add(timestamp, frameDuration, { keyFrame });
      frameIndex += 1;
    },
    async finalize() {
      if (closed) throw new Error("MP4 encoder is already closed");
      closed = true;
      source.close();
      await output.finalize();
      if (!target.buffer) {
        throw new Error("MP4 muxer returned an empty output");
      }
      return new Blob([target.buffer], { type: "video/mp4" });
    },
    async cancel() {
      if (closed) return;
      closed = true;
      source.close();
      await output.cancel();
    },
  };
}
