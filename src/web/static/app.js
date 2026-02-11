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
