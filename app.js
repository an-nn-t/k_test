// app.js - YOLOのみ / YOLO->CNN両対応版
// onnxruntime-web (ort) が読み込まれていることを前提

let yoloSession = null;
let cnnSession = null;
let currentImage = null;
let detections = [];

const YOLO_SIZE = 640;   // YOLO入力サイズ（モデルに合わせて変更）
const CNN_SIZE = 224;    // CNN入力サイズ（モデルに合わせて変更）
const YOLO_CONF_THRESH = 0.4; // スコア閾値

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  await loadEmbeddedModels();
});

function setupEventListeners(){
  document.getElementById('imageInput').addEventListener('change', handleImageUpload);
  document.getElementById('processBtn').addEventListener('click', processImage);

  document.getElementById('modeSelect').addEventListener('change', () => {
    // UIの変更などあればここに
  });
}

async function loadEmbeddedModels(){
  try {
    showLoading(true);
    document.getElementById('fileList').textContent = 'モデルを読み込み中…';

    const [yoloBuf, cnnBuf] = await Promise.all([
      fetch('models/yolo.onnx').then(r => { if(!r.ok) throw new Error('yolo.onnx を取得できませんでした'); return r.arrayBuffer(); }),
      fetch('models/cnn.onnx').then(r => { if(!r.ok) throw new Error('cnn.onnx を取得できませんでした'); return r.arrayBuffer(); })
    ]);

    yoloSession = await ort.InferenceSession.create(yoloBuf);
    cnnSession = await ort.InferenceSession.create(cnnBuf);

    document.getElementById('fileList').textContent = 'モデル読み込み完了';
    document.getElementById('processBtn').disabled = false;
  } catch (err) {
    console.error(err);
    showError('モデルの読み込みに失敗しました: ' + err.message);
    document.getElementById('fileList').textContent = 'モデル読み込み失敗';
  } finally {
    showLoading(false);
  }
}

function handleImageUpload(ev){
  const file = ev.target.files?.[0];
  if(!file) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    drawImageOnCanvas(img);
    document.getElementById('fileList').textContent = `選択: ${file.name} (${img.width}x${img.height})`;
  };
  img.onerror = () => showError('画像を読み込めませんでした');
  img.src = url;
}

async function processImage(){
  clearResultsUI();
  if(!currentImage || !yoloSession){
    showError('画像またはモデルが準備できていません');
    return;
  }

  try {
    showLoading(true);
    const mode = document.getElementById('modeSelect').value;

    // 1) YOLO推論
    detections = await runYOLO(currentImage);
    // YOLOだけの描画（クラスやスコアがあれば表示）
    drawDetections(currentImage, detections, { showYOLOLabel: true });

    // 2) YOLO->CNN（オプション）
    if(mode === 'yolo-cnn' && cnnSession && detections.length > 0){
      const predictions = [];
      for(const det of detections){
        try {
          const inputTensor = preprocessForCNN(currentImage, det);
          const cnnInputName = (cnnSession.inputNames && cnnSession.inputNames[0]) ? cnnSession.inputNames[0] : 'input';
          const feeds = {};
          feeds[cnnInputName] = inputTensor;
          const out = await cnnSession.run(feeds);
          const outTensor = out[Object.keys(out)[0]];
          const predicted = argMax(outTensor.data);
          det.cnnClass = predicted;
          det.cnnScores = outTensor.data;
          predictions.push(det);
        } catch (e) {
          console.warn('CNN推論で失敗:', e);
        }
      }
      // CNNの情報を上書きで描画（ラベルを追加）
      drawDetections(currentImage, detections, { showYOLOLabel: true, showCNNLabel: true });

      // エラー解析表示
      if(document.getElementById('errorAnalysisCheck').checked){
        showErrorAnalysis(detections);
      } else {
        document.getElementById('errorAnalysisSection').style.display = 'none';
      }

      // 結果summary表示
      displaySummary(detections);
    } else {
      // YOLOのみモード summary
      displaySummary(detections);
    }
  } catch (err) {
    console.error(err);
    showError('処理中にエラーが発生しました: ' + err.message);
  } finally {
    showLoading(false);
  }
}

/* ----------------------------
   YOLO 前処理 / 推論 / 出力解析
   ---------------------------- */

function preprocessForYOLO(img){
  // 640x640 にリサイズして [1,3,640,640] CHW, normalized 0..1
  const canvas = document.createElement('canvas');
  canvas.width = YOLO_SIZE;
  canvas.height = YOLO_SIZE;
  const ctx = canvas.getContext('2d');
  // 画像のアスペクト比はそのままにする場合は letterbox を実装する必要あり。
  // ここでは単純に stretch（必要なら letterbox 実装に変更）
  ctx.drawImage(img, 0, 0, YOLO_SIZE, YOLO_SIZE);
  const imageData = ctx.getImageData(0,0,YOLO_SIZE,YOLO_SIZE).data;

  const data = new Float32Array(3 * YOLO_SIZE * YOLO_SIZE);
  let idx = 0;
  // CHW: R then G then B
  for(let c=0;c<3;c++){
    for(let h=0;h<YOLO_SIZE;h++){
      for(let w=0;w<YOLO_SIZE;w++){
        const pixelIdx = (h * YOLO_SIZE + w) * 4;
        data[idx++] = imageData[pixelIdx + c] / 255.0;
      }
    }
  }
  return new ort.Tensor('float32', data, [1,3,YOLO_SIZE,YOLO_SIZE]);
}

async function runYOLO(img){
  const tensor = preprocessForYOLO(img);
  const yoloInputName = (yoloSession.inputNames && yoloSession.inputNames[0]) ? yoloSession.inputNames[0] : 'images';
  const feeds = {};
  feeds[yoloInputName] = tensor;
  const outputMap = await yoloSession.run(feeds);
  const outTensor = outputMap[Object.keys(outputMap)[0]];
  return parseYOLOOutput(outTensor, img.width, img.height);
}

/**
 * parseYOLOOutput
 *  - 汎用的にいくつかのフォーマットに対応
 *  - 想定: 各ボックスは (cx, cy, w, h, conf, class_scores...)
 *  - cx/cy/w/h は [0,1] の正規化値（もし絶対値なら調整が必要）
 */
function parseYOLOOutput(tensor, imgW, imgH){
  const d = tensor.data;
  const dims = tensor.dims;

  let boxes = [];
  // ケース1: [1, N, E] または [N, E]
  if(dims.length === 3){
    const N = dims[1];
    const E = dims[2];
    for(let i=0;i<N;i++){
      const base = i * E;
      const cx = d[base + 0];
      const cy = d[base + 1];
      const w = d[base + 2];
      const h = d[base + 3];
      const conf = d[base + 4];
      // class scores
      let maxClass = -1;
      let maxScore = -Infinity;
      for(let c=5;c<E;c++){
        const score = d[base + c];
        if(score > maxScore){ maxScore = score; maxClass = c-5; }
      }
      const score = conf * (maxScore === -Infinity ? 1.0 : maxScore);
      if(score >= YOLO_CONF_THRESH){
        const absW = w * imgW;
        const absH = h * imgH;
        const absCx = cx * imgW;
        const absCy = cy * imgH;
        const x = absCx - absW/2;
        const y = absCy - absH/2;
        boxes.push({
          box: [x, y, absW, absH],
          class: maxClass,
          score: score
        });
      }
    }
  } else if(dims.length === 2){
    // [N, E]
    const N = dims[0], E = dims[1];
    for(let i=0;i<N;i++){
      const base = i * E;
      const cx = d[base + 0];
      const cy = d[base + 1];
      const w = d[base + 2];
      const h = d[base + 3];
      const conf = d[base + 4];
      let maxClass = -1;
      let maxScore = -Infinity;
      for(let c=5;c<E;c++){
        const score = d[base + c];
        if(score > maxScore){ maxScore = score; maxClass = c-5; }
      }
      const score = conf * (maxScore === -Infinity ? 1.0 : maxScore);
      if(score >= YOLO_CONF_THRESH){
        const absW = w * imgW;
        const absH = h * imgH;
        const absCx = cx * imgW;
        const absCy = cy * imgH;
        const x = absCx - absW/2;
        const y = absCy - absH/2;
        boxes.push({
          box: [x, y, absW, absH],
          class: maxClass,
          score: score
        });
      }
    }
  } else {
    console.warn('UNKNOWN YOLO output dims:', dims);
  }

  // 左上が負になる場合や範囲外にならないようclamp
  boxes = boxes.map(b => {
    const [x,y,w,h] = b.box;
    const nx = Math.max(0, Math.min(x, imgW-1));
    const ny = Math.max(0, Math.min(y, imgH-1));
    const nw = Math.max(1, Math.min(w, imgW - nx));
    const nh = Math.max(1, Math.min(h, imgH - ny));
    return {...b, box: [nx, ny, nw, nh]};
  });

  // 簡易NMS（中心距離ベースの重複除去）
  boxes.sort((a,b)=>a.box[0]-b.box[0]);
  const filtered = [];
  for(const b of boxes){
    let dup = false;
    const cx = b.box[0] + b.box[2]/2;
    for(const e of filtered){
      const ecx = e.box[0] + e.box[2]/2;
      if(Math.abs(cx-ecx) < Math.min(b.box[2], e.box[2]) * 0.5){
        dup = true; break;
      }
    }
    if(!dup) filtered.push(b);
  }
  return filtered;
}

/* ----------------------------
   CNN 前処理
   - 入力 [1,3,224,224] を期待
   ---------------------------- */
function preprocessForCNN(image, det){
  const [x,y,w,h] = det.box.map(v=>Math.round(v));
  // 切り出し用canvas
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = Math.max(1,w);
  cropCanvas.height = Math.max(1,h);
  const cropCtx = cropCanvas.getContext('2d');
  cropCtx.drawImage(image, x, y, w, h, 0, 0, cropCanvas.width, cropCanvas.height);

  // リサイズ用canvas
  const cnnCanvas = document.createElement('canvas');
  cnnCanvas.width = CNN_SIZE;
  cnnCanvas.height = CNN_SIZE;
  const cnnCtx = cnnCanvas.getContext('2d');
  cnnCtx.drawImage(cropCanvas, 0, 0, CNN_SIZE, CNN_SIZE);

  const imageData = cnnCtx.getImageData(0,0,CNN_SIZE,CNN_SIZE).data;
  // normalize (mean/std) - モデルに合わせたいがここでは一般的なImageNetのmean/stdを使用
  const mean = [0.485, 0.456, 0.406];
  const std  = [0.229, 0.224, 0.225];

  const tensorData = new Float32Array(3 * CNN_SIZE * CNN_SIZE);
  let pixelCount = CNN_SIZE * CNN_SIZE;
  for(let i=0;i<pixelCount;i++){
    const r = imageData[i*4] / 255;
    const g = imageData[i*4 + 1] / 255;
    const b = imageData[i*4 + 2] / 255;
    tensorData[i] = (r - mean[0]) / std[0];                                     // R channel
    tensorData[i + pixelCount] = (g - mean[1]) / std[1];                        // G channel
    tensorData[i + 2*pixelCount] = (b - mean[2]) / std[2];                      // B channel
  }
  return new ort.Tensor('float32', tensorData, [1,3,CNN_SIZE,CNN_SIZE]);
}

/* ----------------------------
   描画・UI関連
   ---------------------------- */

function drawImageOnCanvas(img){
  const canvas = document.getElementById('imageCanvas');
  const ctx = canvas.getContext('2d');
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0);
}

function drawDetections(img, dets, opts = { showYOLOLabel:false, showCNNLabel:false }){
  drawImageOnCanvas(img);
  const canvas = document.getElementById('imageCanvas');
  const ctx = canvas.getContext('2d');
  ctx.lineWidth = 2;
  for(const d of dets){
    const [x,y,w,h] = d.box;
    ctx.strokeStyle = 'red';
    ctx.strokeRect(x,y,w,h);

    // 背景ボックス
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    const labelYOLO = opts.showYOLOLabel ? `YOLO:${d.class ?? '-'} ${(d.score? (d.score*100).toFixed(1): '-') }%` : '';
    const labelCNN = (opts.showCNNLabel && d.cnnClass !== undefined) ? `CNN:${d.cnnClass}` : '';
    const label = [labelYOLO, labelCNN].filter(Boolean).join('  ');
    if(label){
      ctx.font = '16px sans-serif';
      const tw = ctx.measureText(label).width;
      const lx = x;
      const ly = Math.max(16, y - 6);
      ctx.fillRect(lx - 2, ly - 14, tw + 6, 18);
      ctx.fillStyle = 'white';
      ctx.fillText(label, lx + 2, ly);
    }
  }
}

function displaySummary(dets){
  const numbers = dets.map(d => d.cnnClass !== undefined ? d.cnnClass : (d.class ?? '-'));
  const avgConf = dets.length > 0 ? (dets.reduce((s,d)=>s+(d.score||0),0) / dets.length) : 0;
  document.getElementById('predictedNumbers').textContent = numbers.join(', ') || '-';
  document.getElementById('bboxCount').textContent = dets.length;
  document.getElementById('avgConfidence').textContent = dets.length ? (avgConf*100).toFixed(1) + '%' : '-';
}

function showErrorAnalysis(dets){
  const container = document.getElementById('bboxImagesContainer');
  container.innerHTML = '';
  for(const [i,d] of dets.entries()){
    // 切り出しサムネイル
    const c = document.createElement('canvas');
    const [x,y,w,h] = d.box.map(v=>Math.round(v));
    c.width = Math.min(200, Math.max(1,w));
    c.height = Math.min(200, Math.max(1,h));
    const ctx = c.getContext('2d');
    ctx.drawImage(currentImage, x, y, w, h, 0, 0, c.width, c.height);

    const item = document.createElement('div');
    item.className = 'bbox-item';
    const title = document.createElement('div');
    title.textContent = `BBox ${i+1} ${d.cnnClass !== undefined ? '-> CNN:'+d.cnnClass : ''}`;
    title.style.fontWeight = '600';
    const imgEl = document.createElement('img');
    imgEl.className = 'stage-image';
    imgEl.src = c.toDataURL();

    item.appendChild(title);
    item.appendChild(imgEl);
    container.appendChild(item);
  }
  document.getElementById('errorAnalysisSection').style.display = dets.length ? 'block' : 'none';
}

function clearResultsUI(){
  document.getElementById('predictedNumbers').textContent = '-';
  document.getElementById('bboxCount').textContent = '-';
  document.getElementById('avgConfidence').textContent = '-';
  document.getElementById('errorMessage').textContent = '';
  document.getElementById('errorAnalysisSection').style.display = 'none';
  document.getElementById('bboxImagesContainer').innerHTML = '';
}

function showError(msg){
  const el = document.getElementById('errorMessage');
  el.textContent = msg;
  setTimeout(()=>{ el.textContent = ''; }, 6000);
}

function showLoading(on){
  document.querySelector('.loading').classList.toggle('active', on);
}

/* util */
function argMax(arr){
  let im = 0;
  let mv = arr[0];
  for(let i=1;i<arr.length;i++){
    if(arr[i] > mv){ mv = arr[i]; im = i; }
  }
  return im;
}
