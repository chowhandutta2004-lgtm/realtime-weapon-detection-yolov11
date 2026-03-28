// === CCTV Detection ===
(function () {
    const connectBtn = document.getElementById('cctvConnect');
    const disconnectBtn = document.getElementById('cctvDisconnect');
    const urlInput = document.getElementById('cctvUrl');
    const fileInput = document.getElementById('cctvFileInput');
    const canvas = document.getElementById('cctvCanvas');
    const placeholder = document.getElementById('cctvPlaceholder');
    const statusDot = document.getElementById('cctvStatusDot');
    const statusText = document.getElementById('cctvStatusText');
    const fpsPill = document.getElementById('cctvFps');
    const detectionList = document.getElementById('cctvDetectionList');
    const statusMsg = document.getElementById('cctvStatusMsg');
    const ctx = canvas.getContext('2d');

    let ws = null;
    let connected = false;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    connectBtn.addEventListener('click', () => connectWithUrl());
    disconnectBtn.addEventListener('click', disconnect);

    // Allow Enter key to connect
    urlInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !connected) connectWithUrl();
    });

    // File upload handler
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        showStatus(`Uploading ${file.name}...`, 'info');
        connectBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch('/api/detect/cctv/upload', { method: 'POST', body: formData });
            if (!res.ok) throw new Error('Upload failed');

            const data = await res.json();
            // Connect using the server-side file path
            connectToSource(data.path);
        } catch (err) {
            showStatus(`Upload failed: ${err.message}`, 'error');
            connectBtn.disabled = false;
        }

        // Reset file input so same file can be re-selected
        fileInput.value = '';
    });

    // Send confidence updates
    confidenceSlider.addEventListener('input', () => {
        if (ws && ws.readyState === WebSocket.OPEN && connected) {
            ws.send(JSON.stringify({ confidence: getConfidence() }));
        }
    });

    function showStatus(message, type) {
        statusMsg.textContent = message;
        statusMsg.className = `cctv-status-msg cctv-status-${type}`;
        statusMsg.classList.remove('hidden');
    }

    function hideStatus() {
        statusMsg.classList.add('hidden');
    }

    function connectWithUrl() {
        const url = urlInput.value.trim();
        if (!url) {
            showStatus('Please enter a camera URL or upload a video file.', 'error');
            return;
        }
        connectBtn.disabled = true;
        connectToSource(url);
    }

    function connectToSource(source) {
        hideStatus();
        showStatus('Connecting...', 'info');

        const conf = getConfidence();
        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${wsProtocol}//${location.host}/api/detect/cctv?confidence=${conf}`);

        ws.onopen = () => {
            ws.send(JSON.stringify({ action: 'connect', url: source }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.status === 'connecting') {
                showStatus('Establishing connection...', 'info');
                return;
            }

            if (data.status === 'connected') {
                connected = true;
                hideStatus();
                placeholder.classList.add('hidden');
                connectBtn.classList.add('hidden');
                connectBtn.disabled = false;
                disconnectBtn.classList.remove('hidden');
                statusDot.classList.add('active');
                statusText.textContent = 'Live';
                fpsPill.classList.remove('hidden');
                urlInput.disabled = true;
                frameCount = 0;
                lastFpsTime = performance.now();
                return;
            }

            if (data.status === 'disconnected') {
                resetUI();
                showStatus('Disconnected.', 'info');
                return;
            }

            if (data.error) {
                showStatus(data.error, 'error');
                if (!connected) {
                    connectBtn.disabled = false;
                }
                return;
            }

            // Frame data
            if (data.image) {
                frameCount++;
                const now = performance.now();
                if (now - lastFpsTime >= 1000) {
                    fpsPill.textContent = `${frameCount} FPS`;
                    frameCount = 0;
                    lastFpsTime = now;
                }

                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                };
                img.src = `data:image/jpeg;base64,${data.image}`;

                updateDetections(data.detections);
            }
        };

        ws.onclose = () => {
            if (connected) {
                resetUI();
                showStatus('Connection lost.', 'error');
            } else {
                connectBtn.disabled = false;
            }
        };

        ws.onerror = () => {
            showStatus('WebSocket error. Is the server running?', 'error');
            connectBtn.disabled = false;
        };
    }

    function disconnect() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'disconnect' }));
        }
        resetUI();
    }

    function resetUI() {
        connected = false;
        if (ws) {
            ws.close();
            ws = null;
        }
        placeholder.classList.remove('hidden');
        connectBtn.classList.remove('hidden');
        connectBtn.disabled = false;
        disconnectBtn.classList.add('hidden');
        statusDot.classList.remove('active');
        statusText.textContent = 'Offline';
        fpsPill.classList.add('hidden');
        urlInput.disabled = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        detectionList.innerHTML = '<p class="no-detections">No active detections</p>';
    }

    function updateDetections(detections) {
        if (!detections || detections.length === 0) {
            detectionList.innerHTML = '<p class="no-detections">No active detections</p>';
            return;
        }

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
})();
