// Dorothy PWA app logic

// Theme toggle
(function() {
  var root = document.documentElement;
  var toggle = document.getElementById('theme-toggle');
  var metaTheme = document.querySelector('meta[name="theme-color"]');

  function getTheme() {
    return root.dataset.theme || 'light';
  }

  function updateToggle() {
    if (!toggle) return;
    toggle.textContent = getTheme() === 'dark' ? '\u2600' : '\u263E';
  }

  function updateMeta() {
    if (!metaTheme) return;
    metaTheme.content = getTheme() === 'dark' ? '#1a1a1a' : '#f8f6f1';
  }

  function setTheme(theme) {
    root.dataset.theme = theme;
    updateToggle();
    updateMeta();
  }

  updateToggle();
  updateMeta();

  if (toggle) {
    toggle.addEventListener('click', function() {
      var next = getTheme() === 'dark' ? 'light' : 'dark';
      localStorage.setItem('theme', next);
      setTheme(next);
    });
  }

  // Follow OS preference if user hasn't manually chosen
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
      if (localStorage.getItem('theme')) return;
      setTheme(e.matches ? 'dark' : 'light');
    });
  }
})();

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

  let swipeTarget = null;

  document.addEventListener('touchstart', e => {
    swipeTarget = e.target;
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });

  document.addEventListener('touchend', e => {
    // Don't trigger swipe navigation from inside the similarity graph/matrix
    if (swipeTarget && swipeTarget.closest('#similarity-web')) return;
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

// Podcast audio player
function initPlayer(playBtn, audio, progressWrap, progressBar, timeEl) {
  if (!playBtn || !audio) return;

  let loaded = false;

  function fmtTime(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m + ':' + (sec < 10 ? '0' : '') + sec;
  }

  playBtn.addEventListener('click', () => {
    if (!loaded) {
      audio.load();
      loaded = true;
    }
    if (audio.paused) {
      audio.play();
      playBtn.innerHTML = '\u275A\u275A';
    } else {
      audio.pause();
      playBtn.innerHTML = '\u25B6';
    }
  });

  audio.addEventListener('timeupdate', () => {
    if (!audio.duration) return;
    const pct = (audio.currentTime / audio.duration) * 100;
    progressBar.style.width = pct + '%';
    timeEl.textContent = fmtTime(audio.currentTime) + ' / ' + fmtTime(audio.duration);
  });

  audio.addEventListener('ended', () => {
    playBtn.innerHTML = '\u25B6';
    progressBar.style.width = '0%';
  });

  if (progressWrap) {
    progressWrap.addEventListener('click', (e) => {
      if (!audio.duration) return;
      const rect = progressWrap.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      audio.currentTime = pct * audio.duration;
    });
  }
}

// Front page player strip
initPlayer(
  document.getElementById('ps-play'),
  document.getElementById('ps-audio'),
  document.getElementById('ps-progress-wrap'),
  document.getElementById('ps-progress-bar'),
  document.getElementById('ps-time')
);

// Podcast archive page player
initPlayer(
  document.getElementById('fp-play'),
  document.getElementById('fp-audio'),
  document.getElementById('fp-progress-wrap'),
  document.getElementById('fp-progress-bar'),
  document.getElementById('fp-time')
);

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
