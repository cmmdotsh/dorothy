// Dorothy PWA app logic

// Register service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/static/sw.js')
      .then(reg => console.log('Service Worker registered'))
      .catch(err => console.log('Service Worker registration failed', err));
  });
}

// Mobile navigation - swipe gestures on story pages
if (document.querySelector('.story-detail')) {
  let touchStartX = 0;
  let touchEndX = 0;

  const handleSwipe = () => {
    const swipeThreshold = 100;
    const diff = touchEndX - touchStartX;

    if (Math.abs(diff) > swipeThreshold) {
      if (diff > 0) {
        // Swipe right - previous article
        const prevLink = document.querySelector('.story-nav-prev');
        if (prevLink) prevLink.click();
      } else {
        // Swipe left - next article
        const nextLink = document.querySelector('.story-nav-next');
        if (nextLink) nextLink.click();
      }
    }
  };

  document.addEventListener('touchstart', e => {
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });

  document.addEventListener('touchend', e => {
    touchEndX = e.changedTouches[0].screenX;
    handleSwipe();
  }, { passive: true });
}

// Keyboard navigation
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.key === 'ArrowLeft') {
    const prevLink = document.querySelector('.story-nav-prev');
    if (prevLink) {
      e.preventDefault();
      prevLink.click();
    }
  } else if (e.key === 'ArrowRight') {
    const nextLink = document.querySelector('.story-nav-next');
    if (nextLink) {
      e.preventDefault();
      nextLink.click();
    }
  }
});

// Relative time formatting
function relativeTime(dt) {
  const now = new Date();
  const diffMs = now - dt;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHrs = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return diffMins + 'm ago';
  if (diffHrs < 24) return diffHrs + 'h ago';
  if (diffDays < 7) return diffDays + 'd ago';
  return dt.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// Compact relative time for listing pages
function updateTimeago() {
  document.querySelectorAll('time.timeago').forEach(el => {
    const dt = new Date(el.getAttribute('datetime'));
    if (isNaN(dt)) return;
    el.textContent = relativeTime(dt);
  });

  // Full readable dates for story detail pages: "Feb 12, 2:34 PM (3h ago)"
  document.querySelectorAll('time.timeago-full').forEach(el => {
    const dt = new Date(el.getAttribute('datetime'));
    if (isNaN(dt)) return;
    const formatted = dt.toLocaleDateString(undefined, {
      month: 'short', day: 'numeric', year: 'numeric'
    }) + ', ' + dt.toLocaleTimeString(undefined, {
      hour: 'numeric', minute: '2-digit'
    });
    const rel = relativeTime(dt);
    el.textContent = formatted + ' (' + rel + ')';
  });
}
updateTimeago();

// Column page sort controls
const sortControls = document.querySelector('.sort-controls');
if (sortControls) {
  const container = document.getElementById('stories-list');
  const buttons = sortControls.querySelectorAll('.sort-btn');

  const sortFns = {
    hotness: (a, b) => parseFloat(b.dataset.hotness || 0) - parseFloat(a.dataset.hotness || 0),
    newest: (a, b) => (b.dataset.generatedAt || '').localeCompare(a.dataset.generatedAt || ''),
    sources: (a, b) => parseInt(b.dataset.sourceCount || 0) - parseInt(a.dataset.sourceCount || 0),
  };

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const sortKey = btn.dataset.sort;
      const cards = Array.from(container.querySelectorAll('.story-card'));
      cards.sort(sortFns[sortKey]);
      cards.forEach(card => container.appendChild(card));

      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}
