// === Confidence Slider ===
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');

confidenceSlider.addEventListener('input', () => {
    confidenceValue.textContent = `${confidenceSlider.value}%`;
});
confidenceValue.textContent = `${confidenceSlider.value}%`;

function getConfidence() {
    return parseInt(confidenceSlider.value) / 100;
}

// === Tab Switching ===
const TAB_TITLES = {
    webcam: { title: 'Live Camera', desc: 'Real-time weapon detection from your webcam' },
    cctv:   { title: 'CCTV Monitor', desc: 'Connect your IP camera for continuous surveillance' },
    upload: { title: 'Upload Files for Testing', desc: 'Upload images or videos to test weapon detection' },
};

document.querySelectorAll('.nav-item[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;

        // Update active nav
        document.querySelectorAll('.nav-item[data-tab]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Show/hide tab content
        document.querySelectorAll('.tab-content').forEach(sec => {
            sec.classList.toggle('active', sec.id === `tab-${tab}`);
            sec.classList.toggle('hidden', sec.id !== `tab-${tab}`);
        });

        // Update topbar title
        const info = TAB_TITLES[tab];
        if (info) {
            document.getElementById('pageTitle').textContent = info.title;
            document.getElementById('pageDesc').textContent = info.desc;
        }
    });
});

// === Drop Zone Setup (shared by image & video upload) ===
function setupDropZone(dropZoneId, inputId, onFiles) {
    const zone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);
    if (!zone || !input) return;

    // Click to open file picker
    zone.addEventListener('click', (e) => {
        if (e.target === input) return;
        input.click();
    });

    // Drag effects
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            onFiles(e.dataTransfer.files);
        }
    });

    // Glow effect following mouse
    zone.addEventListener('mousemove', (e) => {
        const rect = zone.getBoundingClientRect();
        zone.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
        zone.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
    });
}

// === Shared Helpers ===
function getDotClass(label) {
    const lower = label.toLowerCase();
    if (lower.includes('handgun') || lower.includes('pistol')) return 'dot-handgun';
    if (lower.includes('long_gun') || lower.includes('rifle') || lower.includes('shotgun')) return 'dot-longgun';
    if (lower.includes('knife')) return 'dot-knife';
    if (lower.includes('explosive') || lower.includes('grenade')) return 'dot-explosive';
    return 'dot-other';
}

function getThreatIcon(label) {
    const lower = label.toLowerCase();
    if (lower.includes('handgun') || lower.includes('pistol')) return 'Handgun';
    if (lower.includes('long_gun')) return 'Long Gun';
    if (lower.includes('knife')) return 'Knife';
    if (lower.includes('explosive') || lower.includes('grenade')) return 'Explosive';
    return label;
}

// === Toast Notifications ===
function showToast(message, type = 'danger') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
        <span>${message}</span>
    `;
    container.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add('toast-visible'));
    setTimeout(() => {
        toast.classList.remove('toast-visible');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
