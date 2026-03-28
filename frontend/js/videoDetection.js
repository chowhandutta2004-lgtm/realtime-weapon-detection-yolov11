// === Video Detection (batch up to 10, with SSE progress) ===
(function () {
    const dropZone = document.getElementById('videoDropZone');
    const loading = document.getElementById('videoLoading');
    const results = document.getElementById('videoResults');
    const resetBtn = document.getElementById('videoReset');
    const input = document.getElementById('videoInput');
    const loadingText = document.getElementById('videoLoadingText');
    const progressWrap = document.getElementById('videoProgressWrap');
    const progressBar = document.getElementById('videoProgressBar');
    const progressPct = document.getElementById('videoProgressPct');

    // Allow multiple files
    input.setAttribute('multiple', '');

    const MAX_FILES = 10;

    setupDropZone('videoDropZone', 'videoInput', handleFiles);

    function handleFiles(fileOrFiles) {
        let files;
        if (fileOrFiles instanceof FileList || Array.isArray(fileOrFiles)) {
            files = Array.from(fileOrFiles);
        } else {
            files = [fileOrFiles];
        }

        files = files.filter(f => f.type.startsWith('video/'));
        if (files.length === 0) {
            alert('Please upload video files (MP4, AVI, MOV).');
            return;
        }
        if (files.length > MAX_FILES) {
            alert(`Maximum ${MAX_FILES} videos allowed. You selected ${files.length}.`);
            files = files.slice(0, MAX_FILES);
        }

        processVideos(files);
    }

    // Override drop handler for multiple files
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('video/'));
        if (files.length > 0) handleFiles(files);
    }, true);

    input.addEventListener('change', () => {
        if (input.files.length > 0) handleFiles(input.files);
    });

    function processOneVideoSSE(file, confidence) {
        return new Promise((resolve) => {
            const formData = new FormData();
            formData.append('file', file);

            // Show progress bar
            progressWrap.classList.remove('hidden');
            progressBar.style.width = '0%';
            progressPct.textContent = '0%';

            fetch(`/api/detect/video/stream?confidence=${confidence}`, {
                method: 'POST',
                body: formData,
            }).then(response => {
                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                function read() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            // If we never got a 'done' event, resolve with error
                            resolve({ error: 'Stream ended unexpectedly' });
                            return;
                        }

                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop(); // keep incomplete line

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
                                    resolve({ data: event });
                                    return;
                                }
                            } catch (e) {
                                // ignore parse errors
                            }
                        }

                        read();
                    }).catch(err => {
                        resolve({ error: err.message });
                    });
                }

                read();
            }).catch(err => {
                resolve({ error: err.message });
            });
        });
    }

    async function processVideos(files) {
        dropZone.classList.add('hidden');
        loading.classList.remove('hidden');
        results.classList.add('hidden');

        const resultsContainer = document.getElementById('videoResults');
        let videosHtml = '';
        let processed = 0;

        for (const file of files) {
            loadingText.innerHTML = `Processing video ${processed + 1}/${files.length}: <strong>${file.name}</strong> (${(file.size / 1024 / 1024).toFixed(1)} MB)...`;

            const conf = getConfidence();
            const result = await processOneVideoSSE(file, conf);

            if (result.error) {
                videosHtml += `
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

                videosHtml += `
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

        // Replace results content
        resultsContainer.innerHTML = `
            ${videosHtml}
            <button id="videoReset" class="btn btn-outline">Process More Videos</button>
        `;

        // Re-attach reset handler
        document.getElementById('videoReset').addEventListener('click', () => {
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
        results.classList.add('hidden');
        dropZone.classList.remove('hidden');
        input.value = '';
    });
})();
