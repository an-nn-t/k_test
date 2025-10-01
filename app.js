// --- グローバル変数 ---
let yoloSession = null;
let cnnSession = null;
let currentImage = null;
let detections = [];

// --- 初期化 ---
document.addEventListener('DOMContentLoaded', async () => {
    setupEventListeners();
    await loadEmbeddedModels(); // ページロード時にモデルを自動ロード
});

/**
 * モデルを fetch で埋め込み読み込み
 */
async function loadEmbeddedModels() {
    try {
        console.log('Loading embedded ONNX models...');
        showLoading(true);

        const [yoloBuffer, cnnBuffer] = await Promise.all([
            fetch('models/final_yolo.onnx').then(res => res.arrayBuffer()),
            fetch('models/final_cnn.onnx').then(res => res.arrayBuffer())
        ]);

        yoloSession = await ort.InferenceSession.create(yoloBuffer);
        cnnSession = await ort.InferenceSession.create(cnnBuffer);

        console.log('Models loaded successfully');
        document.getElementById('processBtn').style.display = 'block';
        showLoading(false);
    } catch (error) {
        console.error('Failed to load embedded models:', error);
        showError('モデルの読み込みに失敗しました: ' + error.message);
        showLoading(false);
    }
}

/**
 * UIイベント設定
 */
function setupEventListeners() {
    const imageInput = document.getElementById('imageInput');
    const processBtn = document.getElementById('processBtn');

    imageInput.addEventListener('change', handleImageUpload);
    processBtn.addEventListener('click', processImage);
}

/**
 * 画像アップロード処理
 */
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
            currentImage = img;
            drawImageOnCanvas(img);
            document.getElementById('fileList').textContent = `アップロードされた画像: ${file.name}`;
            document.getElementById('processBtn').style.display = 'block';
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

/**
 * Canvasに画像を描画
 */
function drawImageOnCanvas(img) {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
}

/**
 * 画像処理開始
 */
async function processImage() {
    if (!currentImage || !yoloSession || !cnnSession) {
        showError('画像またはモデルが準備できていません。');
        return;
    }

    try {
        showLoading(true);

        // 1. YOLOで推論
        detections = await runYOLO(currentImage);

        // 2. CNNで数字を分類
        const results = await runCNNOnDetections(currentImage, detections);

        // 3. 結果表示
        displayResults(results);

        showLoading(false);
    } catch (error) {
        console.error('Error during processing:', error);
        showError('処理中にエラーが発生しました: ' + error.message);
        showLoading(false);
    }
}

/**
 * YOLO推論処理
 */
async function runYOLO(img) {
    const inputTensor = preprocessImage(img, 640, 640);
    const feeds = { images: inputTensor };
    const output = await yoloSession.run(feeds);

    const outputTensor = output[Object.keys(output)[0]];
    return parseYOLOOutput(outputTensor, img.width, img.height);
}

/**
 * YOLO前処理
 */
function preprocessImage(img, targetWidth, targetHeight) {
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
    const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);

    const data = Float32Array.from(imageData.data).filter((_, i) => i % 4 !== 3); // RGBA → RGB
    for (let i = 0; i < data.length; i++) data[i] /= 255.0;

    return new ort.Tensor('float32', data, [1, 3, targetHeight, targetWidth]);
}

/**
 * YOLO出力をパース
 */
function parseYOLOOutput(tensor, imgWidth, imgHeight) {
    const data = tensor.data;
    const numDetections = data.length / 85; // YOLOv5形式: 85要素
    const detections = [];

    for (let i = 0; i < numDetections; i++) {
        const offset = i * 85;
        const x = data[offset];
        const y = data[offset + 1];
        const w = data[offset + 2];
        const h = data[offset + 3];
        const conf = data[offset + 4];

        if (conf > 0.5) {
            detections.push({
                x: (x - w / 2) * imgWidth / 640,
                y: (y - h / 2) * imgHeight / 640,
                width: w * imgWidth / 640,
                height: h * imgHeight / 640,
                confidence: conf
            });
        }
    }
    return detections;
}

/**
 * CNNで切り出した領域を数字分類
 */
async function runCNNOnDetections(img, detections) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const results = [];

    for (const det of detections) {
        canvas.width = det.width;
        canvas.height = det.height;
        ctx.drawImage(img, det.x, det.y, det.width, det.height, 0, 0, det.width, det.height);

        const imageData = ctx.getImageData(0, 0, det.width, det.height);
        const resized = preprocessImageData(imageData, 28, 28);
        const inputTensor = new ort.Tensor('float32', resized, [1, 1, 28, 28]);

        const output = await cnnSession.run({ input: inputTensor });
        const prediction = argMax(Object.values(output)[0].data);

        results.push({ bbox: det, prediction });
    }

    return results;
}

/**
 * CNN用の前処理
 */
function preprocessImageData(imageData, targetWidth, targetHeight) {
    const canvas = document.createElement('canvas');
    canvas.width = targetWidth;
    canvas.height = targetHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(
        createImageFromData(imageData),
        0, 0, targetWidth, targetHeight
    );
    const resizedData = ctx.getImageData(0, 0, targetWidth, targetHeight).data;

    return Float32Array.from(resizedData.filter((_, i) => i % 4 === 0)).map(v => v / 255.0);
}

function createImageFromData(imageData) {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    const img = new Image();
    img.src = canvas.toDataURL();
    return img;
}

function argMax(array) {
    return array.indexOf(Math.max(...array));
}

/**
 * 結果を表示
 */
function displayResults(results) {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    drawImageOnCanvas(currentImage);

    let totalConfidence = 0;

    results.forEach(result => {
        const { bbox, prediction } = result;
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

        ctx.fillStyle = 'red';
        ctx.font = '16px Arial';
        ctx.fillText(prediction, bbox.x, bbox.y > 10 ? bbox.y - 5 : bbox.y + 15);

        totalConfidence += bbox.confidence;
    });

    document.getElementById('predictedNumbers').textContent =
        results.map(r => r.prediction).join(', ');
    document.getElementById('bboxCount').textContent = results.length;
    document.getElementById('avgConfidence').textContent =
        results.length > 0 ? (totalConfidence / results.length).toFixed(2) : '-';
}

/**
 * エラーメッセージ表示
 */
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
}

/**
 * ローディング表示
 */
function showLoading(isLoading) {
    document.querySelector('.loading').classList.toggle('active', isLoading);
}
