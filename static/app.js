import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22";

(() => {
  const $ = (id) => document.getElementById(id);
  const video = $("video");
  const overlay = $("overlay");
  const ctx = overlay.getContext("2d");
  const cameraSelect = $("cameraSelect");
  const placeholder = $("cameraPlaceholder");
  const startBtn = $("startBtn");
  const stopBtn = $("stopBtn");
  const sentenceBox = $("sentenceBox");
  const addBtn = $("addBtn");
  const cropImage = $("cropImage");
  const cropPlaceholder = $("cropPlaceholder");

  const MODEL_URL = "/models/tfjs_model/model.json";
  const LABELS_URL = "/models/class_names.json";
  const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
  const IMAGE_SIZE = 128;
  const PREDICTION_INTERVAL_MS = 320;
  const MIN_ADD_CONFIDENCE = 0.55;

  let classifier = null;
  let classNames = [];
  let handLandmarker = null;
  let cameraStream = null;
  let timer = null;
  let busy = false;
  let latestLetter = null;
  let latestConfidence = 0;
  let predictionHistory = [];
  let lastVideoTime = -1;

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = IMAGE_SIZE;
  cropCanvas.height = IMAGE_SIZE;
  const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

  function setStatus(message, mode = "") {
    const box = $("statusMessage");
    box.className = "status-message" + (mode ? ` ${mode}` : "");
    box.innerHTML = `<span class="status-icon">${mode === "error" || mode === "warn" ? "!" : "i"}</span><span></span>`;
    box.querySelector("span:last-child").textContent = message;
  }

  function setModelStatus(ok, text) {
    $("healthDot").className = "dot " + (ok ? "ready" : "bad");
    $("healthText").textContent = text;
  }

  async function initializeAI() {
    try {
      if (!window.tf) throw new Error("TensorFlow.js CDN did not load. Check your internet connection.");
      try {
        await window.tf.setBackend("webgl");
        await window.tf.ready();
      } catch (_) {
        await window.tf.setBackend("cpu");
        await window.tf.ready();
      }

      const [labelsResponse, loadedClassifier] = await Promise.all([
        fetch(LABELS_URL, { cache: "no-cache" }),
        window.tf.loadLayersModel(MODEL_URL)
      ]);
      if (!labelsResponse.ok) throw new Error("Missing /models/class_names.json. Copy your class_names.json file from Google Drive into models/.");
      classNames = await labelsResponse.json();
      if (!Array.isArray(classNames) || classNames.length !== 26) {
        throw new Error("class_names.json must contain the same 26 labels used when training the CNN.");
      }
      classifier = loadedClassifier;
      const outShape = classifier.outputs?.[0]?.shape;
      if (outShape && outShape[outShape.length - 1] !== classNames.length) {
        throw new Error(`Model output has ${outShape[outShape.length - 1]} classes but labels contain ${classNames.length}.`);
      }

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.45,
        minHandPresenceConfidence: 0.45
      });

      $("classCount").textContent = String(classNames.length);
      setModelStatus(true, "AI models ready");
      setStatus(`Ready. TensorFlow.js backend: ${window.tf.getBackend()}. Start the camera when you're ready.`);
      $("modelWarning").classList.add("hidden");
    } catch (error) {
      setModelStatus(false, "Model setup needed");
      $("modelWarning").textContent = error.message + " See MODEL_EXPORT_GUIDE.md and place model.json plus all weight shards in models/tfjs_model/.";
      $("modelWarning").classList.remove("hidden");
      setStatus("AI setup incomplete. Follow the model export guide, then refresh this page.", "error");
      console.error(error);
    }
  }

  async function listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((device) => device.kind === "videoinput");
      cameraSelect.innerHTML = "";
      cameras.forEach((camera, index) => {
        const option = document.createElement("option");
        option.value = camera.deviceId;
        option.textContent = camera.label || `Camera ${index + 1}`;
        cameraSelect.appendChild(option);
      });
      if (!cameras.length) cameraSelect.add(new Option("No camera found", ""));
    } catch (_) { /* Browser may hide camera names until access is granted. */ }
  }

  async function startCamera() {
    if (!classifier || !handLandmarker) {
      setStatus("AI models are not ready. Follow MODEL_EXPORT_GUIDE.md first.", "error");
      return;
    }
    await stopCamera();
    try {
      const selectedId = cameraSelect.value;
      const videoConstraints = selectedId
        ? { deviceId: { exact: selectedId }, width: { ideal: 640 }, height: { ideal: 480 } }
        : { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } };
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
      video.srcObject = cameraStream;
      video.style.display = "block";
      placeholder.classList.add("hidden");
      await video.play();
      startBtn.disabled = true;
      stopBtn.disabled = false;
      cameraSelect.disabled = true;
      $("cameraBadge").textContent = "LIVE";
      $("cameraBadge").classList.add("green-badge");
      await listCameras();
      cameraSelect.disabled = true;
      resizeOverlay();
      setStatus("Camera connected. Place the complete hand in view and hold one sign steadily.");
      timer = setInterval(processCurrentFrame, PREDICTION_INTERVAL_MS);
    } catch (error) {
      setStatus(`Could not start camera: ${error.message}. Allow camera access in your browser and Windows settings.`, "error");
      await stopCamera();
    }
  }

  function resizeOverlay() {
    if (!video.videoWidth || !video.videoHeight) return;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
  }

  function resetPrediction(message = "Waiting for a hand…") {
    predictionHistory = [];
    latestLetter = null;
    latestConfidence = 0;
    $("letterValue").textContent = "—";
    $("confidenceValue").textContent = "—";
    $("confidenceBar").style.width = "0%";
    $("predictionHint").textContent = message;
    cropImage.style.display = "none";
    cropPlaceholder.style.display = "block";
    addBtn.disabled = true;
  }

  async function processCurrentFrame() {
    if (busy || !cameraStream || !classifier || !handLandmarker || video.readyState < 2 || !video.videoWidth) return;
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    busy = true;

    try {
      const width = video.videoWidth;
      const height = video.videoHeight;
      resizeOverlay();
      ctx.clearRect(0, 0, overlay.width, overlay.height);

      const result = handLandmarker.detectForVideo(video, performance.now());
      if (!result.landmarks || result.landmarks.length === 0) {
        resetPrediction();
        setStatus("No hand detected. Show a complete hand with better lighting.", "warn");
        return;
      }

      const points = result.landmarks.flat();
      const xs = points.map(point => point.x * width);
      const ys = points.map(point => point.y * height);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const centerX = (minX + maxX) / 2;
      const centerY = (minY + maxY) / 2;
      const side = Math.max(maxX - minX, maxY - minY) * 1.55 + 12;
      const x1 = Math.max(0, Math.floor(centerX - side / 2));
      const y1 = Math.max(0, Math.floor(centerY - side / 2));
      const x2 = Math.min(width, Math.ceil(centerX + side / 2));
      const y2 = Math.min(height, Math.ceil(centerY + side / 2));

      if (x2 <= x1 || y2 <= y1) {
        resetPrediction("Move your hand closer to the center.");
        return;
      }

      // Both the video and overlay are mirrored by CSS; coordinates remain in source pixels.
      ctx.strokeStyle = "#50e3a4";
      ctx.lineWidth = Math.max(2, width / 320);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      for (const point of points) {
        ctx.beginPath();
        ctx.fillStyle = "#ffdc28";
        ctx.arc(point.x * width, point.y * height, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      cropCtx.drawImage(video, x1, y1, x2 - x1, y2 - y1, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
      cropImage.src = cropCanvas.toDataURL("image/jpeg", 0.82);
      cropImage.style.display = "block";
      cropPlaceholder.style.display = "none";

      const inputTensor = window.tf.browser.fromPixels(cropCanvas).toFloat().expandDims(0);
      const outputTensor = classifier.predict(inputTensor);
      const tensorScores = Array.from(await outputTensor.data());
      inputTensor.dispose();
      outputTensor.dispose();

      let bestIndex = 0;
      for (let i = 1; i < tensorScores.length; i++) {
        if (tensorScores[i] > tensorScores[bestIndex]) bestIndex = i;
      }
      const letter = String(classNames[bestIndex]);
      const confidence = Number(tensorScores[bestIndex]);
      predictionHistory.push({ letter, confidence });
      if (predictionHistory.length > 5) predictionHistory.shift();

      const counts = new Map();
      predictionHistory.forEach(item => counts.set(item.letter, (counts.get(item.letter) || 0) + 1));
      const stableLetter = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
      const matching = predictionHistory.filter(item => item.letter === stableLetter);
      const stableConfidence = matching.reduce((sum, item) => sum + item.confidence, 0) / matching.length;
      const stableRatio = matching.length / predictionHistory.length;

      latestLetter = stableRatio >= 0.6 && stableConfidence >= MIN_ADD_CONFIDENCE ? stableLetter : null;
      latestConfidence = stableConfidence;
      $("letterValue").textContent = stableRatio >= 0.6 ? stableLetter : letter;
      $("confidenceValue").textContent = `${((stableRatio >= 0.6 ? stableConfidence : confidence) * 100).toFixed(1)}%`;
      $("confidenceBar").style.width = `${Math.max(0, Math.min(100, (stableRatio >= 0.6 ? stableConfidence : confidence) * 100))}%`;
      $("predictionHint").textContent = latestLetter ? "Stable — review before adding" : "Stabilizing / low confidence";
      addBtn.disabled = !latestLetter;

      const displayLetter = stableRatio >= 0.6 ? stableLetter : letter;
      ctx.font = `bold ${Math.max(16, width / 30)}px Manrope, Arial`;
      ctx.fillStyle = "#50e3a4";
      ctx.fillText(`${displayLetter} ${Math.round((stableRatio >= 0.6 ? stableConfidence : confidence) * 100)}%`, x1 + 4, Math.max(24, y1 - 8));

      setStatus(
        latestLetter
          ? `Hand detected. ${stableLetter} is stable across ${matching.length}/${predictionHistory.length} recent frames. Review before adding.`
          : "Prediction is uncertain or still stabilizing. Hold the sign steady.",
        latestLetter ? "" : "warn"
      );
    } catch (error) {
      setStatus(`Inference error: ${error.message}`, "error");
      console.error(error);
    } finally {
      busy = false;
    }
  }

  async function stopCamera() {
    if (timer) clearInterval(timer);
    timer = null;
    if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
    video.pause();
    video.srcObject = null;
    video.style.display = "none";
    placeholder.classList.remove("hidden");
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    startBtn.disabled = false;
    stopBtn.disabled = true;
    cameraSelect.disabled = false;
    $("cameraBadge").textContent = "OFFLINE";
    $("cameraBadge").classList.remove("green-badge");
    resetPrediction("Waiting for a hand…");
    setStatus("Camera stopped. Your text remains in the builder.");
  }

  function syncCharCount() {
    const count = sentenceBox.value.length;
    $("charCount").textContent = `${count} character${count === 1 ? "" : "s"}`;
  }

  startBtn.addEventListener("click", startCamera);
  stopBtn.addEventListener("click", stopCamera);
  cameraSelect.addEventListener("change", () => { if (cameraStream) startCamera(); });
  addBtn.addEventListener("click", () => {
    if (!latestLetter || latestLetter.length !== 1) return;
    sentenceBox.value += latestLetter;
    syncCharCount();
  });
  $("spaceBtn").addEventListener("click", () => { sentenceBox.value += " "; syncCharCount(); sentenceBox.focus(); });
  $("backspaceBtn").addEventListener("click", () => { sentenceBox.value = sentenceBox.value.slice(0, -1); syncCharCount(); sentenceBox.focus(); });
  $("clearBtn").addEventListener("click", () => { sentenceBox.value = ""; syncCharCount(); sentenceBox.focus(); });
  sentenceBox.addEventListener("input", syncCharCount);
  $("copyBtn").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(sentenceBox.value);
      setStatus("Text copied to clipboard.");
    } catch (_) {
      sentenceBox.select();
      document.execCommand("copy");
      setStatus("Text copied (fallback method).");
    }
  });
  $("speakBtn").addEventListener("click", () => {
    const text = sentenceBox.value.trim();
    if (!text) { setStatus("Add some letters to the text builder before speaking.", "warn"); return; }
    if (!("speechSynthesis" in window)) { setStatus("Speech synthesis is not supported in this browser.", "error"); return; }
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  });
  window.addEventListener("beforeunload", stopCamera);
  listCameras();
  initializeAI();
})();
