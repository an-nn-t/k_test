let yoloSession = null;
let cnnSession = null;
let currentImage = null;
let detections = [];

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

async function initializeApp() {
    try {
        console.log('Initializing ONNX models...');
        showLoading(true, 'モデルを読み込み中...');

        const yoloModelPath = './yolo.onnx';
        const cnnModelPath = './cnn.onnx';

        // Promise.allを使用して両方のモデルを並行して読み込む
        [yoloSession, cnnSession] = await Promise.all([
            ort.InferenceSession.create(yoloModelPath),
            ort.InferenceSession.create(cnnModelPath)
        ]);

        console.log('Models loaded successfully');
        showLoading(false);
    } catch (error) {
        console.error('Failed to initialize models:', error);
        showError('モデルの初期化に失敗しました: ' + error.message + ' (ファイルが正しい場所に配置されているか確認してください)');
        showLoading(false);
        // モデル読み込みに失敗した場合、操作ができないようにUIを非表示にする
        document.getElementById('mainControls').style.display = 'none';
    }
}


function setupEventListeners() {
    const imageInput = document.getElementById('imageInput');
    const processBtn = document.getElementById('processBtn');

    imageInput.addEventListener('change', handleImageUpload);
    processBtn.addEventListener('click', processImage);
}

async function handleImageUpload(event) {
    const files = event.target.files;
    if (files.length === 0) return;

    // モデルが読み込まれていない場合は処理を中断
    if (!yoloSession || !cnnSession) {
        showError('モデルが正常に読み込まれていません。ページを再読み込みしてください。');
        return;
    }

    const fileList = document.getElementById('fileList');
    fileList.textContent = `選択: ${files.length}枚の画像`;

    const processBtn = document.getElementById('processBtn');
    processBtn.style.display = 'inline-block';

    const file = files[0];
    const reader = new FileReader();

    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            drawImageOnCanvas(img);
            if (document.getElementById('modeSelect').value !== 'cnn') {
                processImage();
            }
        };
        img.src = e.target.result;
    };

    reader.readAsDataURL(file);
}

function drawImageOnCanvas(img) {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');

    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / img.width);

    canvas.width = img.width * scale;
    canvas.height = img.height * scale;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
}

async function processImage() {
    if (!currentImage) return;

    showLoading(true, '処理中...');
    clearResults();

    const mode = document.getElementById('modeSelect').value;

    try {
        if (mode === 'yolo' || mode === 'yolo-cnn') {
            detections = await runYOLO(currentImage);
            drawImageOnCanvas(currentImage); // BBox描画前に画像を再描画
            drawBBoxes(detections);

            if (mode === 'yolo-cnn' && detections.length > 0) {
                const predictions = await runCNNOnDetections(currentImage, detections);
                displayResults(predictions);

                if (document.getElementById('errorAnalysisCheck').checked) {
                    showErrorAnalysis(currentImage, detections, predictions);
                }
            } else {
                displayYOLOResults(detections);
            }
        } else if (mode === 'cnn') {
            showError('CNNのみモードは手動BBox指定が必要です（未実装）');
        }
    } catch (error) {
        console.error('Processing error:', error);
        showError('処理中にエラーが発生しました: ' + error.message);
    } finally {
        showLoading(false);
    }
}

async function runYOLO(img) {
    const inputTensor = preprocessImageForYOLO(img);
    const feeds = { images: inputTensor };
    const results = await yoloSession.run(feeds);

    const output = results['output0'];
    const detections = parseYOLOOutput(output, img.width, img.height);

    return filterAndSortDetections(detections);
}

function preprocessImageForYOLO(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 640;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(img, 0, 0, 640, 640);
    const imageData = ctx.getImageData(0, 0, 640, 640);

    const float32Data = new Float32Array(3 * 640 * 640);
    let idx = 0;

    for (let c = 0; c < 3; c++) {
        for (let h = 0; h < 640; h++) {
            for (let w = 0; w < 640; w++) {
                const pixelIdx = (h * 640 + w) * 4;
                float32Data[idx++] = imageData.data[pixelIdx + c] / 255.0;
            }
        }
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 640, 640]);
}

function parseYOLOOutput(output, imgWidth, imgHeight) {
    const data = output.data;
    const dims = output.dims;
    const numBoxes = dims[2];
    const detections = [];

    const scaleX = imgWidth / 640;
    const scaleY = imgHeight / 640;

    for (let i = 0; i < numBoxes; i++) {
        const baseIdx = i;
        const x_center = data[baseIdx] * scaleX;
        const y_center = data[numBoxes + baseIdx] * scaleY;
        const w = data[2 * numBoxes + baseIdx] * scaleX;
        const h = data[3 * numBoxes + baseIdx] * scaleY;
        const x1 = x_center - w / 2;
        const y1 = y_center - h / 2;

        let maxScore = 0;
        let maxClass = 0;

        for (let c = 0; c < 10; c++) {
            const score = data[(4 + c) * numBoxes + baseIdx];
            if (score > maxScore) {
                maxScore = score;
                maxClass = c;
            }
        }

        if (maxScore > 0.5) {
            detections.push({
                x: x1,
                y: y1,
                width: w,
                height: h,
                score: maxScore,
                class: maxClass
            });
        }
    }

    return detections;
}


function filterAndSortDetections(detections) {
    // NMS (Non-Maximum Suppression) like logic
    const filtered = [];
    detections.sort((a, b) => b.score - a.score); // Sort by score descending

    while(detections.length > 0) {
        const best = detections.shift();
        filtered.push(best);
        detections = detections.filter(det => {
            const iou = calculateIOU(best, det);
            return iou < 0.3; // IOU threshold
        });
    }

    // Sort by x-coordinate to get left-to-right order
    return filtered.sort((a, b) => a.x - b.x);
}

function calculateIOU(box1, box2) {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);

    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const box1Area = box1.width * box1.height;
    const box2Area = box2.width * box2.height;

    return intersectionArea / (box1Area + box2Area - intersectionArea);
}


async function runCNNOnDetections(img, detections) {
    const predictions = [];

    for (const det of detections) {
        const cropped = cropImage(img, det);
        const binarized = binarizeImage(cropped);
        const inputTensor = preprocessImageForCNN(binarized);

        const feeds = { input: inputTensor };
        const results = await cnnSession.run(feeds);
        const output = results['output'];

        const predictedClass = argmax(output.data);
        predictions.push({
            bbox: det,
            digit: predictedClass,
            confidence: det.score,
            stages: {
                original: cropped,
                binarized: binarized,
                preprocessed: tensorToCanvas(inputTensor)
            }
        });
    }

    return predictions;
}

function cropImage(img, bbox) {
    const canvas = document.createElement('canvas');
    canvas.width = bbox.width;
    canvas.height = bbox.height;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(img, bbox.x, bbox.y, bbox.width, bbox.height, 0, 0, bbox.width, bbox.height);
    return canvas;
}

function binarizeImage(canvas) {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    // 大津の二値化法を適用
    const histogram = new Array(256).fill(0);
    const grayData = [];

    for (let i = 0; i < data.length; i += 4) {
        const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
        histogram[gray]++;
        grayData.push(gray);
    }

    let total = grayData.length;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
        sum += i * histogram[i];
    }

    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let varMax = 0;
    let threshold = 0;

    for (let i = 0; i < 256; i++) {
        wB += histogram[i];
        if (wB === 0) continue;
        wF = total - wB;
        if (wF === 0) break;

        sumB += i * histogram[i];
        let mB = sumB / wB;
        let mF = (sum - sumB) / wF;

        let varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax) {
            varMax = varBetween;
            threshold = i;
        }
    }

    for (let i = 0; i < data.length; i += 4) {
        const value = grayData[i / 4] > threshold ? 255 : 0;
        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;
    }

    const outputCanvas = document.createElement('canvas');
    outputCanvas.width = canvas.width;
    outputCanvas.height = canvas.height;
    const outputCtx = outputCanvas.getContext('2d');
    outputCtx.putImageData(imageData, 0, 0);

    return outputCanvas;
}

function preprocessImageForCNN(canvas) {
    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = 224;
    resizedCanvas.height = 224;
    const ctx = resizedCanvas.getContext('2d');

    ctx.drawImage(canvas, 0, 0, 224, 224);
    const imageData = ctx.getImageData(0, 0, 224, 224);

    const float32Data = new Float32Array(3 * 224 * 224);
    let idx = 0;

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let c = 0; c < 3; c++) {
        for (let h = 0; h < 224; h++) {
            for (let w = 0; w < 224; w++) {
                const pixelIdx = (h * 224 + w) * 4;
                const value = imageData.data[pixelIdx + c] / 255.0;
                float32Data[idx++] = (value - mean[c]) / std[c];
            }
        }
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
}

function tensorToCanvas(tensor) {
    const canvas = document.createElement('canvas');
    canvas.width = 224;
    canvas.height = 224;
    const ctx = canvas.getContext('2d');

    const imageData = ctx.createImageData(224, 224);
    const data = tensor.data;

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let h = 0; h < 224; h++) {
        for (let w = 0; w < 224; w++) {
            const pixelIdx = (h * 224 + w) * 4;

            for (let c = 0; c < 3; c++) {
                const tensorIdx = c * 224 * 224 + h * 224 + w;
                const value = (data[tensorIdx] * std[c] + mean[c]) * 255;
                imageData.data[pixelIdx + c] = Math.max(0, Math.min(255, value));
            }
            imageData.data[pixelIdx + 3] = 255;
        }
    }

    ctx.putImageData(imageData, 0, 0);
    return canvas;
}

function argmax(array) {
    let maxIdx = 0;
    let maxVal = array[0];

    for (let i = 1; i < array.length; i++) {
        if (array[i] > maxVal) {
            maxVal = array[i];
            maxIdx = i;
        }
    }

    return maxIdx;
}

function drawBBoxes(detections) {
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');

    const scale = canvas.width / currentImage.naturalWidth;

    detections.forEach(det => {
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(det.x * scale, det.y * scale, det.width * scale, det.height * scale);

        const text = `${det.class} (${(det.score * 100).toFixed(1)}%)`;
        ctx.fillStyle = '#00FF00';
        ctx.font = '16px Arial';
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(det.x * scale, det.y * scale - 20, textWidth + 4, 20);
        
        ctx.fillStyle = '#000000';
        ctx.fillText(text, det.x * scale + 2, det.y * scale - 5);
    });
}

function displayResults(predictions) {
    const numbers = predictions.map(p => p.digit).join('');
    const avgConf = predictions.length > 0 ? predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length : 0;

    document.getElementById('predictedNumbers').textContent = numbers;
    document.getElementById('bboxCount').textContent = predictions.length;
    document.getElementById('avgConfidence').textContent = (avgConf * 100).toFixed(1) + '%';
}

function displayYOLOResults(detections) {
    const numbers = detections.map(d => d.class).join('');

    document.getElementById('predictedNumbers').textContent = numbers || '-';
    document.getElementById('bboxCount').textContent = detections.length;
    const avgConf = detections.length > 0
        ? detections.reduce((sum, d) => sum + d.score, 0) / detections.length
        : 0;
    document.getElementById('avgConfidence').textContent = (avgConf * 100).toFixed(1) + '%';
}

function showErrorAnalysis(img, detections, predictions) {
    const section = document.getElementById('errorAnalysisSection');
    const container = document.getElementById('bboxImagesContainer');

    section.style.display = 'block';
    container.innerHTML = '';

    predictions.forEach((pred, idx) => {
        const bboxItem = document.createElement('div');
        bboxItem.className = 'bbox-item';

        const title = document.createElement('div');
        title.textContent = `BBox ${idx + 1}: 予測=${pred.digit}`;
        title.style.fontWeight = 'bold';
        bboxItem.appendChild(title);

        const stages = document.createElement('div');
        stages.className = 'bbox-stages';

        const stageData = [
            { canvas: pred.stages.original, label: '元画像' },
            { canvas: pred.stages.binarized, label: '二値化' },
            { canvas: pred.stages.preprocessed, label: 'CNN入力' }
        ];

        stageData.forEach(stage => {
            const stageDiv = document.createElement('div');

            const img = document.createElement('img');
            img.className = 'stage-image';
            img.src = stage.canvas.toDataURL();
            img.width = 100;
            img.height = 100;
            img.style.objectFit = 'contain';

            const label = document.createElement('div');
            label.className = 'stage-label';
            label.textContent = stage.label;

            stageDiv.appendChild(img);
            stageDiv.appendChild(label);
            stages.appendChild(stageDiv);
        });

        bboxItem.appendChild(stages);
        container.appendChild(bboxItem);
    });
}

function showLoading(show, message = '処理中...') {
    const loadingEl = document.querySelector('.loading');
    loadingEl.textContent = message;
    loadingEl.classList.toggle('active', show);
}

function showError(message) {
    const errorEl = document.getElementById('errorMessage');
    errorEl.textContent = message;
    errorEl.classList.add('active');
    setTimeout(() => errorEl.classList.remove('active'), 5000);
}

function clearResults() {
    document.getElementById('predictedNumbers').textContent = '-';
    document.getElementById('bboxCount').textContent = '-';
    document.getElementById('avgConfidence').textContent = '-';
    document.getElementById('errorAnalysisSection').style.display = 'none';

    // Canvasをクリア
    const canvas = document.getElementById('imageCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}