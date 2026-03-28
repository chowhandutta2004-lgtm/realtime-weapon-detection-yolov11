// === Video History ===
function loadVideoHistory() {
    const container = document.getElementById('historyList');
    container.innerHTML = '<p class="no-detections">Loading...</p>';

    fetch('/api/detect/video/history')
        .then(res => res.json())
        .then(videos => {
            if (!videos.length) {
                container.innerHTML = '<p class="no-detections">No processed videos yet. Upload a video to get started.</p>';
                return;
            }

            container.innerHTML = videos.map(v => {
                const date = new Date(v.processed_at);
                const timeStr = date.toLocaleDateString(undefined, {
                    year: 'numeric', month: 'short', day: 'numeric',
                    hour: '2-digit', minute: '2-digit'
                });

                const totalDetections = Object.values(v.detections_summary || {}).reduce((a, b) => a + b, 0);
                const hasDetections = totalDetections > 0;

                let summaryChips = '';
                if (hasDetections) {
                    summaryChips = Object.entries(v.detections_summary).map(([label, count]) =>
                        `<span class="history-chip history-chip-threat"><span class="dot ${getDotClass(label)}"></span>${getThreatIcon(label)}: ${count}</span>`
                    ).join('');
                } else {
                    summaryChips = '<span class="history-chip history-chip-safe">No threats</span>';
                }

                return `
                    <div class="history-card" data-video-id="${v.video_id}">
                        <div class="history-card-preview">
                            <video src="${v.video_url}" preload="metadata" muted></video>
                            <div class="history-play-overlay">
                                <svg width="36" height="36" viewBox="0 0 24 24" fill="white"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                            </div>
                        </div>
                        <div class="history-card-info">
                            <div class="history-card-title" title="${v.original_filename}">${v.original_filename}</div>
                            <div class="history-card-meta">
                                <span>${timeStr}</span>
                                <span>${v.file_size_mb} MB</span>
                            </div>
                            <div class="history-card-stats">
                                <span>${v.total_frames} frames</span>
                                <span class="${hasDetections ? 'text-danger' : 'text-safe'}">${v.frames_with_detections} with detections</span>
                            </div>
                            <div class="history-card-chips">${summaryChips}</div>
                            <button class="btn btn-ghost btn-delete-video" data-video-id="${v.video_id}" title="Delete video">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                                Delete
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            // Play on click
            container.querySelectorAll('.history-card-preview').forEach(preview => {
                preview.addEventListener('click', () => {
                    const video = preview.querySelector('video');
                    if (video.paused) {
                        video.controls = true;
                        video.muted = false;
                        video.play();
                        preview.querySelector('.history-play-overlay').classList.add('hidden');
                    }
                });
            });

            // Delete handlers
            container.querySelectorAll('.btn-delete-video').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const videoId = btn.dataset.videoId;
                    if (!confirm('Delete this processed video?')) return;

                    fetch(`/api/detect/video/history/${videoId}`, { method: 'DELETE' })
                        .then(res => {
                            if (res.ok) {
                                const card = container.querySelector(`[data-video-id="${videoId}"]`);
                                if (card) card.remove();
                                if (!container.children.length) {
                                    container.innerHTML = '<p class="no-detections">No processed videos yet. Upload a video to get started.</p>';
                                }
                            }
                        });
                });
            });
        })
        .catch(() => {
            container.innerHTML = '<p class="no-detections">Failed to load video history.</p>';
        });
}
