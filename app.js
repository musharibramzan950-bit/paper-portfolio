// Hand-scrubbed hero video.
// MediaPipe HandLandmarker tracks hand openness (fingertip→wrist distance
// normalized by palm size); openness 0–1 maps directly onto the video
// timeline. Fist = frame 0, open palm = last frame.

import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const scrubVideo = document.getElementById("scrubVideo");
const videoFallback = document.getElementById("videoFallback");
const camVideo = document.getElementById("camVideo");
const camCanvas = document.getElementById("camCanvas");
const startBtn = document.getElementById("startBtn");
const hudStatus = document.getElementById("hudStatus");
const recDot = document.getElementById("recDot");
const meterFill = document.getElementById("meterFill");
const opennessLabel = document.getElementById("opennessLabel");

const ctx = camCanvas.getContext("2d");

// Openness calibration: mean(tip→wrist)/palmSize for a fist sits near the
// low bound, a fully spread palm near the high bound. Auto-widened at
// runtime so it adapts to any hand.
let calMin = 1.25;
let calMax = 1.95;
let smoothed = 0; // EMA of openness
const ALPHA = 0.22;

scrubVideo.addEventListener("error", () => {
  videoFallback.hidden = false;
});
// Decode the first frame so the poster isn't blank.
scrubVideo.addEventListener("loadeddata", () => {
  scrubVideo.currentTime = 0;
});

// Keyboard fallback: arrow keys scrub without a camera.
window.addEventListener("keydown", (e) => {
  if (!scrubVideo.duration) return;
  const step = scrubVideo.duration / 40;
  if (e.key === "ArrowRight") scrubVideo.currentTime = Math.min(scrubVideo.duration, scrubVideo.currentTime + step);
  if (e.key === "ArrowLeft") scrubVideo.currentTime = Math.max(0, scrubVideo.currentTime - step);
});

let landmarker = null;

async function loadLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "warming up…";
  try {
    hudStatus.textContent = "HAND TRACKER · LOADING";
    const [stream] = await Promise.all([
      navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, facingMode: "user" },
      }),
      loadLandmarker(),
    ]);
    camVideo.srcObject = stream;
    await camVideo.play();
    camCanvas.width = camVideo.videoWidth;
    camCanvas.height = camVideo.videoHeight;
    startBtn.remove();
    recDot.classList.add("live");
    hudStatus.textContent = "HAND TRACKER · LIVE";
    requestAnimationFrame(track);
  } catch (err) {
    hudStatus.textContent = "HAND TRACKER · BLOCKED";
    startBtn.disabled = false;
    startBtn.textContent = "camera blocked — retry";
    console.error(err);
  }
});

const TIPS = [8, 12, 16, 20]; // index, middle, ring, pinky fingertips
const WRIST = 0;
const MIDDLE_MCP = 9;

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0));
}

function handOpenness(lm) {
  const palm = dist(lm[WRIST], lm[MIDDLE_MCP]);
  if (palm < 1e-6) return null;
  const mean =
    TIPS.reduce((s, i) => s + dist(lm[i], lm[WRIST]), 0) / TIPS.length / palm;
  // widen calibration toward observed extremes
  calMin = Math.min(calMin, mean + 0.02);
  calMax = Math.max(calMax, mean - 0.02);
  return Math.min(1, Math.max(0, (mean - calMin) / (calMax - calMin)));
}

let lastVideoTime = -1;

function track() {
  if (camVideo.currentTime !== lastVideoTime) {
    lastVideoTime = camVideo.currentTime;
    const result = landmarker.detectForVideo(camVideo, performance.now());
    ctx.clearRect(0, 0, camCanvas.width, camCanvas.height);

    const lm = result.landmarks?.[0];
    if (lm) {
      const raw = handOpenness(lm);
      if (raw !== null) {
        smoothed += ALPHA * (raw - smoothed);
        drawHand(lm);
        applyScrub(smoothed);
      }
    }
  }
  requestAnimationFrame(track);
}

function applyScrub(openness) {
  meterFill.style.width = `${(openness * 100).toFixed(0)}%`;
  opennessLabel.textContent = `${(openness * 100).toFixed(0)}%`;
  if (scrubVideo.duration && !scrubVideo.seeking) {
    scrubVideo.currentTime = openness * (scrubVideo.duration - 0.05);
  }
}

const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [5, 9], [9, 10], [10, 11], [11, 12],  // middle
  [9, 13], [13, 14], [14, 15], [15, 16],// ring
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17], // pinky + palm edge
];

function drawHand(lm) {
  const w = camCanvas.width;
  const h = camCanvas.height;
  ctx.strokeStyle = "rgba(231, 183, 164, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [a, b] of BONES) {
    ctx.moveTo(lm[a].x * w, lm[a].y * h);
    ctx.lineTo(lm[b].x * w, lm[b].y * h);
  }
  ctx.stroke();
  ctx.fillStyle = "#f5efe2";
  for (const p of lm) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}
