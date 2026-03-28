// === Webcam Detection (improved) ===
(function () {
    const startBtn = document.getElementById('webcamStart');
    const stopBtn = document.getElementById('webcamStop');
    const canvas = document.getElementById('webcamCanvas');
    const placeholder = document.getElementById('webcamPlaceholder');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    const fpsPill = document.getElementById('webcamFps');
    const videoEl = document.getElementById('webcamVideo');
    const captureCanvas = document.getElementById('captureCanvas');
    const ctx = canvas.getContext('2d');
    const captureCtx = captureCanvas.getContext('2d');
    const detectionList = document.getElementById('webcamDetectionList');
    const statsPanel = document.getElementById('webcamStats');
    const sessionSummary = document.getElementById('webcamSessionSummary');

    let ws = null;
    let stream = null;
    let running = false;

    // FPS tracking
    let frameCount = 0;
    let lastFpsTime = performance.now();

    // Alert deduplication: only alert once per weapon type per cooldown period
    const ALERT_COOLDOWN_MS = 10000; // 10 seconds between alerts for same weapon
    const lastAlertTime = {}; // label -> timestamp

    // Session tracking: unique weapons seen during this session
    const sessionWeapons = {}; // label -> { count, peak_confidence, first_seen }

    const resetSummaryBtn = document.getElementById('resetSummary');

    startBtn.addEventListener('click', start);
    stopBtn.addEventListener('click', stop);
    resetSummaryBtn.addEventListener('click', () => {
        Object.keys(sessionWeapons).forEach(k => delete sessionWeapons[k]);
        Object.keys(lastAlertTime).forEach(k => delete lastAlertTime[k]);
        sessionSummary.innerHTML = '<p class="no-detections">No weapons detected yet</p>';
    });

    // Send confidence updates to WebSocket in real-time
    confidenceSlider.addEventListener('input', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ confidence: getConfidence() }));
        }
    });

    async function start() {
        // Reset session
        Object.keys(sessionWeapons).forEach(k => delete sessionWeapons[k]);
        Object.keys(lastAlertTime).forEach(k => delete lastAlertTime[k]);
        frameCount = 0;
        lastFpsTime = performance.now();

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                }
            });
            videoEl.srcObject = stream;
            await videoEl.play();

            captureCanvas.width = videoEl.videoWidth;
            captureCanvas.height = videoEl.videoHeight;
            canvas.width = videoEl.videoWidth;
            canvas.height = videoEl.videoHeight;

            const conf = getConfidence();
            const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${wsProtocol}//${location.host}/api/detect/live?confidence=${conf}`);

            ws.onopen = () => {
                running = true;
                placeholder.classList.add('hidden');
                startBtn.classList.add('hidden');
                stopBtn.classList.remove('hidden');
                statusDot.classList.add('active');
                statusText.textContent = 'Active';
                fpsPill.classList.remove('hidden');
                statsPanel.classList.remove('hidden');
                sendFrame();
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.error) return;

                // Update FPS counter
                frameCount++;
                const now = performance.now();
                if (now - lastFpsTime >= 1000) {
                    fpsPill.textContent = `${frameCount} FPS`;
                    frameCount = 0;
                    lastFpsTime = now;
                }

                // Draw annotated frame
                const img = new Image();
                img.onload = () => {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    if (running) sendFrame();
                };
                img.src = `data:image/jpeg;base64,${data.image}`;

                // Update current detections display
                updateCurrentDetections(data.detections);

                // Handle alerts with deduplication
                if (data.detections.length > 0) {
                    handleDetections(data.detections);
                }
            };

            ws.onclose = () => {
                if (running) stop();
            };

            ws.onerror = () => {
                alert('WebSocket connection failed. Is the server running?');
                stop();
            };
        } catch (err) {
            alert(`Camera access denied: ${err.message}`);
        }
    }

    function sendFrame() {
        if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;
        captureCtx.drawImage(videoEl, 0, 0);
        captureCanvas.toBlob(
            (blob) => {
                if (blob && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(blob);
                }
            },
            'image/jpeg',
            0.6
        );
    }

    function updateCurrentDetections(detections) {
        if (detections.length === 0) {
            detectionList.innerHTML = '<p class="no-detections">No active detections</p>';
            return;
        }

        // Deduplicate by label, keep highest confidence
        const unique = {};
        for (const d of detections) {
            if (!unique[d.label] || d.confidence > unique[d.label].confidence) {
                unique[d.label] = d;
            }
        }

        detectionList.innerHTML = Object.values(unique).map(d => `
            <div class="detection-item detection-item-live">
                <span class="detection-label">
                    <span class="dot ${getDotClass(d.label)}"></span>
                    ${getThreatIcon(d.label)}
                </span>
                <span class="detection-confidence">${(d.confidence * 100).toFixed(1)}%</span>
            </div>
        `).join('');
    }

    function handleDetections(detections) {
        const now = Date.now();

        for (const det of detections) {
            const label = det.label;

            // Track session summary
            if (!sessionWeapons[label]) {
                sessionWeapons[label] = {
                    frames_detected: 1,
                    peak_confidence: det.confidence,
                    first_seen: new Date().toLocaleTimeString(),
                };
            } else {
                sessionWeapons[label].frames_detected++;
                if (det.confidence > sessionWeapons[label].peak_confidence) {
                    sessionWeapons[label].peak_confidence = det.confidence;
                }
            }

            // Alert with cooldown — only send one alert per weapon type per 10 seconds
            const lastTime = lastAlertTime[label] || 0;
            if (now - lastTime > ALERT_COOLDOWN_MS) {
                lastAlertTime[label] = now;
                // Show toast notification
                showToast(`${getThreatIcon(label)} detected (${(det.confidence * 100).toFixed(0)}% confidence)`);
                // Refresh alerts panel
                if (typeof refreshAlerts === 'function') {
                    refreshAlerts();
                }
            }
        }

        // Update session summary panel
        updateSessionSummary();
    }

    function updateSessionSummary() {
        const labels = Object.keys(sessionWeapons);
        if (labels.length === 0) {
            sessionSummary.innerHTML = '<p class="no-detections">No weapons detected yet</p>';
            return;
        }

        sessionSummary.innerHTML = labels.map(label => {
            const info = sessionWeapons[label];
            return `
                <div class="detection-item">
                    <span class="detection-label">
                        <span class="dot ${getDotClass(label)}"></span>
                        ${getThreatIcon(label)}
                    </span>
                    <span class="detection-confidence">${(info.peak_confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="stat-row stat-row-detail">
                    <span>First seen: ${info.first_seen}</span>
                    <span>${info.frames_detected} frames</span>
                </div>
            `;
        }).join('');
    }

    function stop() {
        running = false;
        if (ws) {
            ws.close();
            ws = null;
        }
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        placeholder.classList.remove('hidden');
        startBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
        statusDot.classList.remove('active');
        statusText.textContent = 'Offline';
        fpsPill.classList.add('hidden');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        detectionList.innerHTML = '<p class="no-detections">No active detections</p>';
    }
})();
