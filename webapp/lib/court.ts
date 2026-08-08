/* Top-down court, in metres — a TypeScript port of the static page's
   assets/court.js, kept behaviourally identical so the two viewers cannot
   drift apart while both exist.

   Everything is FIBA: 28 × 15 m floor, 6.75 m arc, corner lines 0.9 m in from
   the sideline, basket centre 1.575 m from the baseline. The NBA clip's data
   is expressed in the dataset's own 28 × 15 model, so one drawing serves both
   sources — the court model note lives with the data, not here. */

export interface Surface {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
}

export interface Geometry {
  s: number;
  L: number;
  W: number;
  X: (m: number) => number;
  Y: (m: number) => number;
}

/* Match the backing store to the laid-out size and device pixel ratio, so the
   court stays sharp and never stretches. Null while the canvas is unlaid-out. */
export function fit(canvas: HTMLCanvasElement): Surface | null {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  const w = Math.round(cw * dpr);
  const h = Math.round(ch * dpr);
  if (!w || !h) return null;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: cw, h: ch };
}

/* Metres → CSS pixels, letterboxed inside the canvas. */
export function geometry(cw: number, ch: number, L: number, W: number, pad = 26): Geometry {
  const s = Math.min((cw - pad * 2) / L, (ch - pad * 2) / W);
  const ox = (cw - L * s) / 2;
  const oy = (ch - W * s) / 2;
  return { s, L, W, X: m => ox + m * s, Y: m => oy + m * s };
}

export function lines(ctx: CanvasRenderingContext2D, g: Geometry, o: {
  w?: number; h?: number; bg?: string; line?: string; strong?: string; rim?: string;
  weight?: number;
} = {}): void {
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
    const bx = left ? 1.575 : L - 1.575; // basket centre
    const dir = left ? 1 : -1;
    ctx.strokeRect(X(left ? 0 : L - 5.8), Y(W / 2 - 2.45), 5.8 * s, 4.9 * s);
    ctx.beginPath(); ctx.arc(X(left ? 5.8 : L - 5.8), Y(W / 2), 1.8 * s, 0, Math.PI * 2); ctx.stroke();

    // The angle at which the 6.75 m arc meets the corner line 0.9 m in from
    // the sideline — asin, not acos.
    const a = Math.asin((W / 2 - 0.9) / 6.75);
    ctx.beginPath();
    ctx.arc(X(bx), Y(W / 2), 6.75 * s, left ? -a : Math.PI - a, left ? a : Math.PI + a);
    ctx.stroke();
    const cx = bx + dir * 6.75 * Math.cos(a);
    ctx.beginPath();
    ctx.moveTo(X(left ? 0 : L), Y(0.9)); ctx.lineTo(X(cx), Y(0.9));
    ctx.moveTo(X(left ? 0 : L), Y(W - 0.9)); ctx.lineTo(X(cx), Y(W - 0.9));
    ctx.stroke();

    ctx.strokeStyle = rim;
    ctx.beginPath(); ctx.arc(X(bx), Y(W / 2), 0.225 * s, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = line;
  }
}
