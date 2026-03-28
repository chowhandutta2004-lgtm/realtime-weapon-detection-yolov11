// === Unified Upload Detection (Images + Videos, up to 10 files) ===
(function () {
    const dropZone = document.getElementById('uploadDropZone');
    const loading = document.getElementById('uploadLoading');
    const resultsContainer = document.getElementById('uploadResults');
    const resetBtn = document.getElementById('uploadReset');
    const input = document.getElementById('uploadInput');
    const loadingText = document.getElementById('uploadLoadingText');
    const progressWrap = document.getElementById('uploadProgressWrap');
    const progressBar = document.getElementById('uploadProgressBar');
    const progressPct = document.getElementById('uploadProgressPct');

    const MAX_FILES = 10;

    setupDropZone('uploadDropZone', 'uploadInput', handleFiles);

    function handleFiles(fileOrFiles) {
        let files;
        if (fileOrFiles instanceof FileList || Array.isArray(fileOrFiles)) {
            files = Array.from(fileOrFiles);
        } else {
            files = [fileOrFiles];
        }

        // Filter to images and videos only
        files = files.filter(f => f.type.startsWith('image/') || f.type.startsWith('video/'));
        if (files.length === 0) {
            showToast('Please upload image or video files.', 'danger');
            return;
        }
        if (files.length > MAX_FILES) {
            showToast(`Maximum ${MAX_FILES} files allowed. Only first ${MAX_FILES} will be processed.`, 'danger');
            files = files.slice(0, MAX_FILES);
        }

        processFiles(files);
    }

    // Override drop handler
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(f =>
            f.type.startsWith('image/') || f.type.startsWith('video/')
        );
        if (files.length > 0) handleFiles(files);
    }, true);

    input.addEventListener('change', () => {
        if (input.files.length > 0) handleFiles(input.files);
    });

    // --- Image processing ---
    async function processOneImage(file, confidence) {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`/api/detect/image?confidence=${confidence}`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        const data = await res.json();

        const originalUrl = await readFileAsDataURL(file);

        return {
            type: 'image',
            filename: file.name,
            originalUrl,
            annotatedImage: data.image,
            detections: data.detections || [],
            weapon_detected: data.weapon_detected,
        };
    }

    function readFileAsDataURL(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });
    }

    // --- Video processing (SSE) ---
    function processOneVideo(file, confidence) {
        return new Promise((resolve) => {
            const formData = new FormData();
            formData.append('file', file);

            progressWrap.classList.remove('hidden');
            progressBar.style.width = '0%';
            progressPct.textContent = '0%';

            fetch(`/api/detect/video/stream?confidence=${confidence}`, {
                method: 'POST',
                body: formData,
            }).then(response => {
                if (!response.ok) throw new Error(`Server error: ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                function read() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            resolve({ type: 'video', filename: file.name, error: 'Stream ended unexpectedly' });
                            return;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();

                        for (const line of lines) {
                            if (!line.startsWith('data: ')) continue;
                            try {
                                const event = JSON.parse(line.slice(6));
                                if (event.type === 'progress') {
                                    progressBar.style.width = event.percent + '%';
                                    progressPct.textContent = event.percent + '%';
                                } else if (event.type === 'encoding') {
                                    progressBar.style.width = '100%';
                                    progressPct.textContent = '';
                                    loadingText.innerHTML = 'Encoding video for playback...';
                                } else if (event.type === 'done') {
                                    progressBar.style.width = '100%';
                                    progressPct.textContent = '100%';
                                    resolve({
                                        type: 'video',
                                        filename: file.name,
                                        data: event,
                                    });
                                    return;
                                }
                            } catch (e) { /* ignore */ }
                        }

                        read();
                    }).catch(err => {
                        resolve({ type: 'video', filename: file.name, error: err.message });
                    });
                }

                read();
            }).catch(err => {
                resolve({ type: 'video', filename: file.name, error: err.message });
            });
        });
    }

    // --- Unified processing ---
    async function processFiles(files) {
        dropZone.classList.add('hidden');
        loading.classList.remove('hidden');
        resultsContainer.classList.add('hidden');

        const images = files.filter(f => f.type.startsWith('image/'));
        const videos = files.filter(f => f.type.startsWith('video/'));
        const total = files.length;
        let processed = 0;
        let html = '';

        // Process images first
        if (images.length > 0) {
            loadingText.textContent = `Scanning ${images.length} image${images.length > 1 ? 's' : ''}... (0/${images.length})`;
            progressWrap.classList.add('hidden');

            let allDetections = [];
            let gridHtml = '';

            for (const file of images) {
                try {
                    const conf = getConfidence();
                    const result = await processOneImage(file, conf);

                    gridHtml += `
                        <div class="result-card">
                            <div class="result-header">
                                <span class="result-tag">Original</span>
                                <span class="result-filename">${result.filename}</span>
                            </div>
                            <div class="result-img-wrap">
                                <img src="${result.originalUrl}" alt="Original">
                            </div>
                        </div>
                        <div class="result-card result-card-detected">
                            <div class="result-header">
                                <span class="result-tag tag-detected">Analyzed</span>
                                <span class="result-count">${result.detections.length} detection${result.detections.length !== 1 ? 's' : ''}</span>
                            </div>
                            <div class="result-img-wrap">
                                <img src="data:image/jpeg;base64,${result.annotatedImage}" alt="Annotated">
                            </div>
                        </div>
                    `;

                    result.detections.forEach(d => {
                        allDetections.push({ ...d, source: file.name });
                    });

                    if (result.weapon_detected && typeof refreshAlerts === 'function') {
                        refreshAlerts();
                    }
                } catch (err) {
                    gridHtml += `
                        <div class="result-card" style="grid-column: span 2;">
                            <div class="result-header">
                                <span class="result-tag" style="color: var(--danger);">Error</span>
                            </div>
                            <p style="padding: 20px; color: var(--text-muted);">${file.name}: ${err.message}</p>
                        </div>
                    `;
                }

                processed++;
                loadingText.textContent = `Scanning images... (${processed - videos.length > 0 ? processed : processed}/${images.length})`;
            }

            // Build image results section
            let detectionsHtml;
            if (allDetections.length === 0) {
                detectionsHtml = '<p class="no-detections">No weapons detected in any image</p>';
            } else {
                detectionsHtml = allDetections.map(d => `
                    <div class="detection-item">
                        <span class="detection-label">
                            <span class="dot ${getDotClass(d.label)}"></span>
                            ${getThreatIcon(d.label)}
                        </span>
                        <span class="detection-confidence">${(d.confidence * 100).toFixed(1)}%</span>
                    </div>
                `).join('');
            }

            html += `
                <div class="upload-section-header">Image Results</div>
                <div class="result-grid">${gridHtml}</div>
                <div class="detections-panel">
                    <h3 class="panel-title">Image Detections</h3>
                    <div class="detection-list">${detectionsHtml}</div>
                </div>
            `;
        }

        // Process videos
        if (videos.length > 0) {
            let videoIdx = 0;
            for (const file of videos) {
                videoIdx++;
                loadingText.innerHTML = `Processing video ${videoIdx}/${videos.length}: <strong>${file.name}</strong> (${(file.size / 1024 / 1024).toFixed(1)} MB)...`;

                const conf = getConfidence();
                const result = await processOneVideo(file, conf);

                if (result.error) {
                    html += `
                        <div class="detections-panel">
                            <h3 class="panel-title" style="color: var(--danger);">Error — ${file.name}</h3>
                            <p class="no-detections">${result.error}</p>
                        </div>
                    `;
                } else {
                    const data = result.data;
                    const weapons = Object.entries(data.detections_summary);
                    let statsHtml = '';

                    if (weapons.length === 0) {
                        statsHtml = '<p class="no-detections">No weapons detected</p>';
                    } else {
                        for (const [label, info] of weapons) {
                            statsHtml += `
                                <div class="detection-item">
                                    <span class="detection-label">
                                        <span class="dot ${getDotClass(label)}"></span>
                                        ${getThreatIcon(label)}
                                    </span>
                                    <span class="detection-confidence">${(info.peak_confidence * 100).toFixed(1)}%</span>
                                </div>
                                <div class="stat-row stat-row-detail">
                                    <span>First seen at ${info.first_seen}</span>
                                    <span>Last seen at ${info.last_seen}</span>
                                </div>
                            `;
                        }
                        statsHtml += `
                            <div class="stat-row stat-row-muted">
                                <span>Analyzed ${data.total_frames} frames</span>
                                <span>${data.frames_with_detections} with detections</span>
                            </div>
                        `;
                    }

                    html += `
                        <div class="video-result-block">
                            <div class="result-card result-card-video">
                                <div class="result-header">
                                    <span class="result-tag tag-detected">${file.name}</span>
                                </div>
                                <video controls src="${data.video_url}"></video>
                            </div>
                            <div class="detections-panel">
                                <h3 class="panel-title">Summary — ${file.name}</h3>
                                <div class="detection-list">${statsHtml}</div>
                            </div>
                        </div>
                    `;

                    if (Object.keys(data.detections_summary).length > 0 && typeof refreshAlerts === 'function') {
                        refreshAlerts();
                    }
                }

                processed++;
            }
        }

        // Render all results
        resultsContainer.innerHTML = `
            ${html}
            <button id="uploadReset" class="btn btn-outline">Upload More Files</button>
        `;

        // Re-attach reset handler
        document.getElementById('uploadReset').addEventListener('click', () => {
            resultsContainer.innerHTML = '';
            resultsContainer.classList.add('hidden');
            dropZone.classList.remove('hidden');
            input.value = '';
        });

        progressWrap.classList.add('hidden');
        loading.classList.add('hidden');
        resultsContainer.classList.remove('hidden');
    }

    resetBtn.addEventListener('click', () => {
        resultsContainer.classList.add('hidden');
        dropZone.classList.remove('hidden');
        input.value = '';
    });
})();
