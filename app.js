let yoloSession = null;
let cnnSession = null;
let currentImage = null;

// --- 初期化 ---
document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    await loadModels();
});

async function loadModels() {
    try {
        setStatus("モデルを読み込み中...");
        const [yoloBuffer, cnnBuffer] = await Promise.all([
            fetch('models/yolo.onnx').then(res => res.arrayBuffer()),
            fetch('models/cnn.onnx').then(res => res.arrayBuffer())
        ]);
        yoloSession = await ort.InferenceSession.create(yoloBuffer);
        cnnSession = await ort.InferenceSession.create(cnnBuffer);
        setStatus("モデル読み込み完了");
    } catch (err) {
        setStatus("モデル読み込み失敗: " + err.message);
    }
}

function setupEventListeners() {
    document.getElementById('imageInput').addEventListener('change', handleImageUpload);
    document.getElementById('processBtn').addEventListener('click', processImage);
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const img = new Image();
    img.onload = () => {
        currentImage = img;
        drawImageOnCanvas(img);
        document.getElementById('processBtn').style.display = 'inline-block';
    };
    img.src = URL.createObjectURL(file);
}

async function processImage() {
    if (!currentImage || !yoloSession) return;
    setStatus("YOLO推論中...");
    const yoloResults = await runYOLO(currentImage);

    // YOLOの検出結果を描画
    drawDetections(currentImage, yoloResults, "YOLO");

    // CNNで各bboxを再分類
    if (cnnSession) {
        setStatus("CNN推論中...");
        for (let det of yoloResults) {
            const inputTensor = preprocessForCNN(currentImage, det);
            const feeds = { input: inputTensor }; // CNNの入力名はモデルに合わせる
            const results = await cnnSession.run(feeds);
            const output = results[Object.keys(results)[0]];
            const predictedClass = argMax(output.data);
            det.cnnClass = predictedClass;
        }
        drawDetections(currentImage, yoloResults, "CNN");
    }
    setStatus("推論完了");
}

// --- YOLO 前処理 ---
function preprocessForYOLO(image) {
    const size = 640; // YOLO入力サイズに合わせる
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, size, size);

    const imageData = ctx.getImageData(0, 0, size, size);
    const { data } = imageData;
    const input = new Float32Array(3 * size * size);

    for (let i = 0; i < size * size; i++) {
        input[i] = data[i * 4] / 255.0;                       // R
        input[i + size * size] = data[i * 4 + 1] / 255.0;     // G
        input[i + 2 * size * size] = data[i * 4 + 2] / 255.0; // B
    }

    return new ort.Tensor('float32', input, [1, 3, size, size]);
}

// --- YOLO 推論実行 ---
async function runYOLO(image) {
    const tensor = preprocessForYOLO(image);
    const feeds = { images: tensor }; // 入力名はモデルに合わせる（例: "images"）
    const results = await yoloSession.run(feeds);

    const output = results[Object.keys(results)[0]];
    console.log("YOLO raw output:", output.data);

    return parseYOLOOutput(output, image.width, image.height);
}

// --- YOLO 出力解釈 ---
// YOLO出力が [x1, y1, x2, y2, conf, class] 形式の場合
function parseYOLOOutput(output, imgWidth, imgHeight) {
    const data = output.data;
    const numBoxes = data.length / 6;
    let detections = [];

    for (let i = 0; i < numBoxes; i++) {
        const x1 = data[i * 6 + 0];
        const y1 = data[i * 6 + 1];
        const x2 = data[i * 6 + 2];
        const y2 = data[i * 6 + 3];
        const score = data[i * 6 + 4];
        const cls = data[i * 6 + 5];

        if (score > 0.7) { // 閾値を高めに
            detections.push({
                x: x1,
                y: y1,
                width: x2 - x1,
                height: y2 - y1,
                score: score,
                class: cls
            });
        }
    }

    // NMSで重複除去
    detections = nonMaxSuppression(detections, 0.45);

    return detections;
}



// --- CNN用の前処理 (224x224 RGB) ---
function preprocessForCNN(image, det) {
    const targetSize = 224;
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = det.width;
    cropCanvas.height = det.height;
    const cropCtx = cropCanvas.getContext('2d');
    cropCtx.drawImage(image, det.x, det.y, det.width, det.height, 0, 0, det.width, det.height);

    const cnnCanvas = document.createElement('canvas');
    cnnCanvas.width = targetSize;
    cnnCanvas.height = targetSize;
    const cnnCtx = cnnCanvas.getContext('2d');
    cnnCtx.drawImage(cropCanvas, 0, 0, targetSize, targetSize);

    const imageData = cnnCtx.getImageData(0, 0, targetSize, targetSize);
    const { data } = imageData;
    const input = new Float32Array(3 * targetSize * targetSize);

    for (let i = 0; i < targetSize * targetSize; i++) {
        input[i] = data[i * 4] / 255.0;
        input[i + targetSize * targetSize] = data[i * 4 + 1] / 255.0;
        input[i + 2 * targetSize * targetSize] = data[i * 4 + 2] / 255.0;
    }
    return new ort.Tensor('float32', input, [1, 3, targetSize, targetSize]);
}

// --- 描画 (YOLO or CNNの結果ラベルを選択) ---
function drawDetections(image, detections, mode) {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    for (let det of detections) {
        ctx.strokeStyle = "red";
        ctx.lineWidth = 2;
        ctx.strokeRect(det.x, det.y, det.width, det.height);

        let label = "";
        if (mode === "YOLO") {
            label = `YOLO: ${det.class} (${det.score.toFixed(2)})`;
        } else if (mode === "CNN") {
            label = `CNN: ${det.cnnClass}`;
        }

        ctx.fillStyle = "yellow";
        ctx.font = "14px Arial";
        ctx.fillText(label, det.x, det.y > 10 ? det.y - 5 : det.y + 15);
    }
}

// --- 補助関数 ---
function argMax(arr) {
    return arr.indexOf(Math.max(...arr));
}

function drawImageOnCanvas(img) {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
}

function setStatus(msg) {
    document.getElementById('status').innerText = msg;
}
