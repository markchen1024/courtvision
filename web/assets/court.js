/* Top-down court, in metres.
 *
 * Shared by the viewer and the marketing page so there is exactly one court
 * drawing in the project — a landing page whose court is a slightly different
 * shape from the product's is worse than no landing page.
 *
 * Everything is FIBA: 28 × 15 m floor, 6.75 m arc, corner lines 0.9 m in from
 * the sideline, basket centre 1.575 m from the baseline.
 */
const Court = (() => {

  /* Match the backing store to the laid-out size (and to the device pixel
     ratio) so the court stays sharp and never stretches. Returns null while
     the canvas is still unlaid-out. */
  function fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
    if (!w || !h) return null;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    // Draw in CSS pixels from here on; the transform handles the retina scale.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: cw, h: ch };
  }

  /* Metres → CSS pixels, letterboxed inside the canvas. */
  function geometry(cw, ch, L, W, pad = 26) {
    const s = Math.min((cw - pad * 2) / L, (ch - pad * 2) / W);
    const ox = (cw - L * s) / 2, oy = (ch - W * s) / 2;
    return { s, L, W, X: m => ox + m * s, Y: m => oy + m * s };
  }

  function lines(ctx, g, o = {}) {
    const bg = o.bg ?? '#050608';
    const line = o.line ?? '#23272f';
    const strong = o.strong ?? '#333a45';
    const rim = o.rim ?? '#4a525f';
    const { L, W, s, X, Y } = g;

    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, o.w ?? ctx.canvas.width, o.h ?? ctx.canvas.height);

    ctx.lineWidth = o.weight ?? 1.5;
    ctx.strokeStyle = strong;
    ctx.strokeRect(X(0), Y(0), L * s, W * s);

    ctx.strokeStyle = line;
    ctx.beginPath(); ctx.moveTo(X(L / 2), Y(0)); ctx.lineTo(X(L / 2), Y(W)); ctx.stroke();
    ctx.beginPath(); ctx.arc(X(L / 2), Y(W / 2), 1.8 * s, 0, Math.PI * 2); ctx.stroke();

    for (const left of [true, false]) {
      const bx = left ? 1.575 : L - 1.575;          // basket centre
      const dir = left ? 1 : -1;
      ctx.strokeRect(X(left ? 0 : L - 5.8), Y(W / 2 - 2.45), 5.8 * s, 4.9 * s);
      ctx.beginPath(); ctx.arc(X(left ? 5.8 : L - 5.8), Y(W / 2), 1.8 * s, 0, Math.PI * 2); ctx.stroke();

      // `a` is the angle at which the 6.75 m arc meets the corner line 0.9 m in
      // from the sideline — asin, not acos.
      const a = Math.asin((W / 2 - 0.9) / 6.75);
      ctx.beginPath();
      ctx.arc(X(bx), Y(W / 2), 6.75 * s, left ? -a : Math.PI - a, left ? a : Math.PI + a);
      ctx.stroke();
      const cx = bx + dir * 6.75 * Math.cos(a);
      ctx.beginPath();
      ctx.moveTo(X(left ? 0 : L), Y(0.9));     ctx.lineTo(X(cx), Y(0.9));
      ctx.moveTo(X(left ? 0 : L), Y(W - 0.9)); ctx.lineTo(X(cx), Y(W - 0.9));
      ctx.stroke();

      ctx.strokeStyle = rim;
      ctx.beginPath(); ctx.arc(X(bx), Y(W / 2), 0.225 * s, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = line;
    }
  }

  return { fit, geometry, lines };
})();
