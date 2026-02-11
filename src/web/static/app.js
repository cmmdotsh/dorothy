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
