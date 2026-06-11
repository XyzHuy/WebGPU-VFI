import {
  BufferTarget,
  CanvasSource,
  EncodedAudioPacketSource,
  EncodedPacketSink,
  Mp4OutputFormat,
  Output,
  QUALITY_HIGH,
} from "mediabunny";

export async function createMp4CanvasEncoder(canvas, { fps, audioTrack = null, signal = null }) {
  if (!("VideoEncoder" in window)) {
    throw new Error("WebCodecs VideoEncoder is unavailable in this browser");
  }
  if (!Number.isFinite(fps) || fps <= 0) {
    throw new Error(`Invalid output FPS: ${fps}`);
  }

  const format = new Mp4OutputFormat({ fastStart: "in-memory" });
  const target = new BufferTarget();
  const output = new Output({
    format,
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

  const audio = await prepareAudioCopy(format, audioTrack);
  if (audio.source) {
    output.addAudioTrack(audio.source);
  }

  await output.start();
  const audioCopyPromise = audio.source
    ? copyAudioPackets(audioTrack, audio.source, signal)
      .then((result) => ({ result, error: null }))
      .catch((error) => ({ result: null, error }))
    : Promise.resolve({ result: { packetCount: 0 }, error: null });

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
      try {
        const { result: audioResult, error: audioError } = await audioCopyPromise;
        if (audioError) {
          throw audioError;
        }
        source.close();
        await output.finalize();
        if (!target.buffer) {
          throw new Error("MP4 muxer returned an empty output");
        }
        return {
          blob: new Blob([target.buffer], { type: "video/mp4" }),
          audio: {
            ...publicAudioInfo(audio),
            packetCount: audioResult?.packetCount ?? 0,
          },
        };
      } catch (error) {
        source.close();
        await output.cancel().catch(() => {});
        throw error;
      }
    },
    async cancel() {
      if (closed) return;
      closed = true;
      source.close();
      await output.cancel();
      await audioCopyPromise;
    },
    audio: publicAudioInfo(audio),
  };
}

async function prepareAudioCopy(format, audioTrack) {
  if (!audioTrack) {
    return {
      copied: false,
      reason: "none",
      label: "No input audio",
      source: null,
    };
  }

  const codec = await audioTrack.getCodec();
  if (!codec) {
    return {
      copied: false,
      reason: "unknown-codec",
      label: "Audio codec unknown",
      source: null,
    };
  }

  if (!format.getSupportedAudioCodecs().includes(codec)) {
    return {
      copied: false,
      reason: "unsupported-codec",
      codec,
      label: `Unsupported audio: ${codec}`,
      source: null,
    };
  }

  return {
    copied: true,
    reason: "copied",
    codec,
    label: `Copied ${codec}`,
    source: new EncodedAudioPacketSource(codec),
  };
}

async function copyAudioPackets(audioTrack, audioSource, signal) {
  const sink = new EncodedPacketSink(audioTrack);
  const decoderConfig = await audioTrack.getDecoderConfig();
  const meta = { decoderConfig: decoderConfig ?? undefined };
  const firstTimestamp = await audioTrack.getFirstTimestamp();
  const timestampBase = Number.isFinite(firstTimestamp) ? firstTimestamp : 0;
  let packetCount = 0;

  try {
    for await (const packet of sink.packets()) {
      if (signal?.aborted) break;

      const timestamp = Math.max(0, packet.timestamp - timestampBase);
      const copiedPacket = packet.clone({ timestamp });
      await audioSource.add(copiedPacket, meta);
      packetCount += 1;
    }
  } finally {
    audioSource.close();
  }

  return { packetCount };
}

function publicAudioInfo(audio) {
  const { source: _source, ...info } = audio;
  return info;
}
