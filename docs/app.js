// webcam -> yolo11n onnx -> draw box

const NAMES = ["like", "dislike", "palm"];
const COLORS = ["lime", "red", "deepskyblue"];
const SIZE = 416;
const CONF = 0.5;
const IOU = 0.45;

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const msg = document.getElementById("msg");

// letterbox input
const inp = document.createElement("canvas");
inp.width = inp.height = SIZE;
const ictx = inp.getContext("2d", { willReadFrequently: true });

let session;

function preprocess () {
  const vw = video.videoWidth, vh = video.videoHeight;
  const r = Math.min(SIZE / vw, SIZE / vh);
  const nw = Math.round(vw * r), nh = Math.round(vh * r);
  const dx = (SIZE - nw) / 2, dy = (SIZE - nh) / 2;

  ictx.fillStyle = "rgb(114,114,114)";
  ictx.fillRect(0, 0, SIZE, SIZE);
  ictx.drawImage(video, dx, dy, nw, nh);

  const px = ictx.getImageData(0, 0, SIZE, SIZE).data;
  const n = SIZE * SIZE;
  const data = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    data[i] = px[i * 4] / 255;
    data[i + n] = px[i * 4 + 1] / 255;
    data[i + 2 * n] = px[i * 4 + 2] / 255;
  }
  return { tensor: new ort.Tensor("float32", data, [1, 3, SIZE, SIZE]), r, dx, dy };
}

function iou (a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (area(a) + area(b) - inter);
}

// output0 : [1, 7, N] = cx cy w h + 3 class score
function postprocess (out, r, dx, dy) {
  const d = out.data, N = out.dims[2];
  let boxes = [];
  for (let i = 0; i < N; i++) {
    let best = 0, cls = 0;
    for (let c = 0; c < NAMES.length; c++) {
      const s = d[(4 + c) * N + i];
      if (s > best) { best = s; cls = c; }
    }
    if (best < CONF) continue;

    const cx = d[i], cy = d[N + i], w = d[2 * N + i], h = d[3 * N + i];
    boxes.push({
      box: [(cx - w / 2 - dx) / r, (cy - h / 2 - dy) / r, (cx + w / 2 - dx) / r, (cy + h / 2 - dy) / r],
      score: best, cls,
    });
  }

  // nms
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => k.cls !== b.cls || iou(k.box, b.box) < IOU)) keep.push(b);
  }
  return keep;
}

function draw (dets) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.font = "18px sans-serif";
  for (const { box, score, cls } of dets) {
    const [x1, y1, x2, y2] = box;
    ctx.strokeStyle = ctx.fillStyle = COLORS[cls];
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillText(`${NAMES[cls]} ${score.toFixed(2)}`, x1 + 2, y1 > 20 ? y1 - 6 : y1 + 18);
  }
}

async function loop () {
  const { tensor, r, dx, dy } = preprocess();
  const res = await session.run({ images: tensor });
  const dets = postprocess(res.output0, r, dx, dy);
  draw(dets);
  // console.log(dets);
  requestAnimationFrame(loop);
}

async function main () {
  session = await ort.InferenceSession.create("yolo11n_best.onnx", { executionProviders: ["wasm"] });
  msg.textContent = "allow camera";

  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  msg.textContent = "running";
  loop();
}

main().catch((e) => { msg.textContent = "error: " + e.message; console.error(e); });
