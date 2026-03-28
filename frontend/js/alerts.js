// === Alerts Panel ===
(function () {
    const toggleBtn = document.getElementById('alertToggle');
    const panel = document.getElementById('alertsPanel');
    const clearBtn = document.getElementById('clearAlerts');
    const alertsList = document.getElementById('alertsList');
    const badge = document.getElementById('alertBadge');

    let lastAlertId = null;
    let unseenCount = 0;

    toggleBtn.addEventListener('click', () => {
        panel.classList.toggle('hidden');
        if (!panel.classList.contains('hidden')) {
            unseenCount = 0;
            badge.classList.add('hidden');
            refreshAlerts();
        }
    });

    clearBtn.addEventListener('click', async () => {
        await fetch('/api/alerts', { method: 'DELETE' });
        alertsList.innerHTML = '<p class="alerts-empty">No alerts yet</p>';
        unseenCount = 0;
        badge.classList.add('hidden');
        lastAlertId = null;
    });

    window.refreshAlerts = async function () {
        try {
            const res = await fetch('/api/alerts?limit=50');
            const alerts = await res.json();

            if (alerts.length === 0) {
                alertsList.innerHTML = '<p class="alerts-empty">No alerts yet</p>';
                return;
            }

            // Count new alerts since last check
            if (lastAlertId && alerts[0].id !== lastAlertId) {
                const newCount = alerts.findIndex(a => a.id === lastAlertId);
                const added = newCount === -1 ? alerts.length : newCount;
                if (panel.classList.contains('hidden')) {
                    unseenCount += added;
                    badge.textContent = unseenCount > 99 ? '99+' : unseenCount;
                    badge.classList.remove('hidden');
                }
            }
            lastAlertId = alerts[0]?.id;

            alertsList.innerHTML = alerts.map(a => {
                const time = new Date(a.timestamp).toLocaleTimeString();
                return `
                    <div class="alert-card">
                        <img src="data:image/jpeg;base64,${a.thumbnail}" alt="${a.label}">
                        <div class="alert-info">
                            <div class="alert-type">
                                <span class="dot ${getDotClass(a.label)}"></span>
                                ${getThreatIcon(a.label)}
                            </div>
                            <div class="alert-conf">${(a.confidence * 100).toFixed(1)}%</div>
                            <div class="alert-time">${time}</div>
                        </div>
                    </div>
                `;
            }).join('');
        } catch {
            // Silently fail
        }
    };

    // Poll less frequently (10 seconds instead of 5)
    setInterval(window.refreshAlerts, 10000);
})();
