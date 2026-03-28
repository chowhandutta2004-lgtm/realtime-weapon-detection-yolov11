// === Image Detection (batch up to 10) ===
(function () {
    const dropZone = document.getElementById('imageDropZone');
    const loading = document.getElementById('imageLoading');
    const results = document.getElementById('imageResults');
    const resetBtn = document.getElementById('imageReset');
    const input = document.getElementById('imageInput');

    // Allow multiple files
    input.setAttribute('multiple', '');

    const MAX_FILES = 10;

    setupDropZone('imageDropZone', 'imageInput', handleFiles);

    function handleFiles(fileOrFiles) {
        let files;
        if (fileOrFiles instanceof FileList || Array.isArray(fileOrFiles)) {
            files = Array.from(fileOrFiles);
        } else {
            files = [fileOrFiles];
        }

        // Filter to images only
        files = files.filter(f => f.type.startsWith('image/'));
        if (files.length === 0) {
            alert('Please upload image files (JPG, PNG, WebP).');
            return;
        }
        if (files.length > MAX_FILES) {
            alert(`Maximum ${MAX_FILES} images allowed. You selected ${files.length}.`);
            files = files.slice(0, MAX_FILES);
        }

        processImages(files);
    }

    // Override drop handler to support multiple files
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length > 0) handleFiles(files);
    }, true);

    input.addEventListener('change', () => {
        if (input.files.length > 0) handleFiles(input.files);
    });

    async function processImages(files) {
        dropZone.classList.add('hidden');
        loading.classList.remove('hidden');
        results.classList.add('hidden');

        const resultGrid = document.querySelector('#imageResults .result-grid');
        const detectionList = document.getElementById('imageDetectionList');

        // Build results HTML
        let gridHtml = '';
        let allDetections = [];
        let processed = 0;

        // Update loading text
        const loadingText = loading.querySelector('p');
        loadingText.textContent = `Scanning ${files.length} image${files.length > 1 ? 's' : ''}... (0/${files.length})`;

        for (const file of files) {
            const formData = new FormData();
            formData.append('file', file);

            try {
                const conf = getConfidence();
                const res = await fetch(`/api/detect/image?confidence=${conf}`, {
                    method: 'POST',
                    body: formData,
                });

                if (!res.ok) throw new Error(`Server error: ${res.status}`);
                const data = await res.json();

                // Read original as data URL
                const originalUrl = await readFileAsDataURL(file);

                gridHtml += `
                    <div class="result-card">
                        <div class="result-header">
                            <span class="result-tag">Original</span>
                            <span class="result-filename">${file.name}</span>
                        </div>
                        <div class="result-img-wrap">
                            <img src="${originalUrl}" alt="Original">
                        </div>
                    </div>
                    <div class="result-card result-card-detected">
                        <div class="result-header">
                            <span class="result-tag tag-detected">Analyzed</span>
                            <span class="result-count">${data.detections.length} detection${data.detections.length !== 1 ? 's' : ''}</span>
                        </div>
                        <div class="result-img-wrap">
                            <img src="data:image/jpeg;base64,${data.image}" alt="Annotated">
                        </div>
                    </div>
                `;

                data.detections.forEach(d => {
                    allDetections.push({ ...d, source: file.name });
                });

                if (data.weapon_detected && typeof refreshAlerts === 'function') {
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
            loadingText.textContent = `Scanning ${files.length} image${files.length > 1 ? 's' : ''}... (${processed}/${files.length})`;
        }

        resultGrid.innerHTML = gridHtml;

        // Render combined detection list
        if (allDetections.length === 0) {
            detectionList.innerHTML = '<p class="no-detections">No weapons detected in any image</p>';
        } else {
            detectionList.innerHTML = allDetections.map(d => `
                <div class="detection-item">
                    <span class="detection-label">
                        <span class="dot ${getDotClass(d.label)}"></span>
                        ${getThreatIcon(d.label)}
                    </span>
                    <span class="detection-confidence">${(d.confidence * 100).toFixed(1)}%</span>
                </div>
            `).join('');
        }

        loading.classList.add('hidden');
        results.classList.remove('hidden');
    }

    function readFileAsDataURL(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.readAsDataURL(file);
        });
    }

    resetBtn.addEventListener('click', () => {
        results.classList.add('hidden');
        dropZone.classList.remove('hidden');
        input.value = '';
    });
})();
