/* =========================================================
   TAS Cinematic · interactions
   - before/after slider (mouse + touch + keyboard)
   - count-up stat animation on scroll
   - scroll reveal
   - feature-card cursor glow
   ========================================================= */

(() => {
  'use strict';

  /* ---------- Before / After slider ---------- */
  const baWrap = document.getElementById('baWrap');
  const baClip = document.getElementById('baClip');
  const baHandle = document.getElementById('baHandle');

  if (baWrap && baClip && baHandle) {
    const stage = baWrap.querySelector('.ba-stage');
    let dragging = false;

    const setPos = (clientX) => {
      const rect = stage.getBoundingClientRect();
      let pct = ((clientX - rect.left) / rect.width) * 100;
      pct = Math.max(0, Math.min(100, pct));
      baClip.style.width = pct + '%';
      baHandle.style.left = pct + '%';
      baHandle.setAttribute('aria-valuenow', Math.round(pct));
    };

    const startDrag = (e) => {
      dragging = true;
      stage.style.cursor = 'grabbing';
      const x = (e.touches ? e.touches[0].clientX : e.clientX);
      setPos(x);
      e.preventDefault();
    };
    const moveDrag = (e) => {
      if (!dragging) return;
      const x = (e.touches ? e.touches[0].clientX : e.clientX);
      setPos(x);
    };
    const endDrag = () => {
      dragging = false;
      stage.style.cursor = 'ew-resize';
    };

    stage.addEventListener('mousedown', startDrag);
    stage.addEventListener('touchstart', startDrag, { passive: false });
    window.addEventListener('mousemove', moveDrag);
    window.addEventListener('touchmove', moveDrag, { passive: true });
    window.addEventListener('mouseup', endDrag);
    window.addEventListener('touchend', endDrag);

    // click-to-seek even without drag
    stage.addEventListener('click', (e) => {
      if (dragging) return;
      setPos(e.clientX);
    });

    // keyboard
    baHandle.addEventListener('keydown', (e) => {
      const current = parseFloat(baClip.style.width) || 50;
      let next = current;
      if (e.key === 'ArrowLeft')  next = Math.max(0, current - 4);
      if (e.key === 'ArrowRight') next = Math.min(100, current + 4);
      if (e.key === 'Home')       next = 0;
      if (e.key === 'End')        next = 100;
      if (next !== current) {
        baClip.style.width = next + '%';
        baHandle.style.left = next + '%';
        baHandle.setAttribute('aria-valuenow', Math.round(next));
        e.preventDefault();
      }
    });

    // auto-demo wiggle once, to hint interactivity
    const hint = () => {
      let t = 0;
      const from = 50, amp = 14, dur = 1400;
      const start = performance.now();
      const tick = (now) => {
        t = now - start;
        if (t >= dur) {
          baClip.style.width = '50%';
          baHandle.style.left = '50%';
          return;
        }
        const p = t / dur;
        const eased = Math.sin(p * Math.PI * 2) * (1 - p);
        const v = from + eased * amp;
        baClip.style.width = v + '%';
        baHandle.style.left = v + '%';
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    };
    // play the hint once when the slider first enters the viewport
    const io = new IntersectionObserver((entries) => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          hint();
          io.disconnect();
        }
      });
    }, { threshold: 0.4 });
    io.observe(baWrap);
  }

  /* ---------- Count-up stats ---------- */
  const stats = document.querySelectorAll('.stat-num');
  if (stats.length) {
    const animateNum = (el) => {
      const target = parseFloat(el.dataset.target || '0');
      const suffix = el.dataset.suffix || '';
      const dur = 1600;
      const start = performance.now();
      const isInt = Number.isInteger(target);
      const step = (now) => {
        const p = Math.min(1, (now - start) / dur);
        // easeOutCubic
        const eased = 1 - Math.pow(1 - p, 3);
        const v = target * eased;
        el.textContent = (isInt ? Math.round(v) : v.toFixed(1)) + suffix;
        if (p < 1) requestAnimationFrame(step);
        else el.textContent = (isInt ? target : target.toFixed(1)) + suffix;
      };
      requestAnimationFrame(step);
    };
    const statIO = new IntersectionObserver((entries) => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          animateNum(en.target);
          statIO.unobserve(en.target);
        }
      });
    }, { threshold: 0.5 });
    stats.forEach(s => statIO.observe(s));
  }

  /* ---------- Scroll reveal ---------- */
  const revealTargets = document.querySelectorAll(
    '.section-head, .feat-card, .dl-card, .stat, .ba-wrap, .hero-terminal'
  );
  revealTargets.forEach(el => el.classList.add('reveal'));
  const revealIO = new IntersectionObserver((entries) => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        en.target.classList.add('in');
        revealIO.unobserve(en.target);
      }
    });
  }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
  revealTargets.forEach(el => revealIO.observe(el));

  /* ---------- Feature card cursor glow ---------- */
  document.querySelectorAll('.feat-card').forEach(card => {
    card.addEventListener('mousemove', (e) => {
      const r = card.getBoundingClientRect();
      const mx = ((e.clientX - r.left) / r.width) * 100;
      const my = ((e.clientY - r.top) / r.height) * 100;
      card.style.setProperty('--mx', mx + '%');
      card.style.setProperty('--my', my + '%');
    });
  });

  /* ---------- Nav shadow on scroll ---------- */
  const nav = document.querySelector('.nav');
  if (nav) {
    const onScroll = () => {
      if (window.scrollY > 12) {
        nav.style.boxShadow = '0 10px 30px -20px rgba(0,0,0,0.9)';
        nav.style.background = 'rgba(10,11,18,0.85)';
      } else {
        nav.style.boxShadow = 'none';
        nav.style.background = 'rgba(10,11,18,0.65)';
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  /* ---------- Pipeline stage sequential reveal ---------- */
  const pStages = document.querySelectorAll('.pipeline-flow .pstage');
  if (pStages.length) {
    const pipeIO = new IntersectionObserver((entries) => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          en.target.classList.add('in');
          pipeIO.unobserve(en.target);
        }
      });
    }, { threshold: 0.25, rootMargin: '0px 0px -40px 0px' });
    pStages.forEach(s => pipeIO.observe(s));
  }

  /* ---------- CLI copy-to-clipboard ---------- */
  const cliBtn = document.getElementById('cliCopy');
  const cliCode = document.getElementById('cliCode');
  if (cliBtn && cliCode) {
    cliBtn.addEventListener('click', async () => {
      // strip leading `$ ` and `›` log lines, keep only the command
      const raw = cliCode.textContent || '';
      const cmd = raw
        .split('\n')
        .filter(l => !l.trim().startsWith('›'))
        .join('\n')
        .replace(/^\$\s*/, '')
        .trim();
      try {
        await navigator.clipboard.writeText(cmd);
      } catch (_) {
        // fallback for file:// + older browsers
        const ta = document.createElement('textarea');
        ta.value = cmd;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (_) {}
        document.body.removeChild(ta);
      }
      const label = cliBtn.querySelector('span');
      const original = label ? label.textContent : 'Copy';
      cliBtn.classList.add('copied');
      if (label) label.textContent = 'Copied';
      setTimeout(() => {
        cliBtn.classList.remove('copied');
        if (label) label.textContent = original;
      }, 1600);
    });
  }

  /* ---------- Hero parallax on mouse move ---------- */
  const heroMesh = document.getElementById('heroMesh');
  if (heroMesh && window.matchMedia('(pointer: fine)').matches) {
    const layers = heroMesh.querySelectorAll('[data-parallax]');
    const CAP = 10; // px
    let raf = 0, targetX = 0, targetY = 0, curX = 0, curY = 0;
    const onMove = (e) => {
      const w = window.innerWidth, h = window.innerHeight;
      // center is 0, edges are ±1
      targetX = ((e.clientX / w) - 0.5) * 2;
      targetY = ((e.clientY / h) - 0.5) * 2;
      if (!raf) raf = requestAnimationFrame(tick);
    };
    const tick = () => {
      curX += (targetX - curX) * 0.08;
      curY += (targetY - curY) * 0.08;
      layers.forEach(el => {
        const r = parseFloat(el.dataset.parallax || '0');
        const x = curX * CAP * (r / 0.05);
        const y = curY * CAP * (r / 0.05);
        el.style.transform = `translate3d(${x.toFixed(2)}px, ${y.toFixed(2)}px, 0)`;
      });
      if (Math.abs(targetX - curX) > 0.001 || Math.abs(targetY - curY) > 0.001) {
        raf = requestAnimationFrame(tick);
      } else {
        raf = 0;
      }
    };
    window.addEventListener('mousemove', onMove, { passive: true });
  }
})();
