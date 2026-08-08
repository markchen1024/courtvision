'use client';

/* NBA-2K-style hot zones: an upright halfcourt cut into annular sectors --
   rim disc, a close ring in three wedges, a mid-range ring in five, the two
   corner strips, and three above-the-break sectors.

   The ready-made chart drew its own zone shapes and offers no way to change
   them, so the shapes are ours. The machinery is the one already proven on
   the static page: classify a fine grid of floor cells, fill them, then
   stroke every edge where two neighbouring cells disagree -- the classifier
   is exact, so the cheapest correct drawing is to ask it per cell. Colours
   keep the site's ramp (red cold, teal hot) rather than 2K's red-hot, so the
   two charts in this repo read the same way. */

import { useEffect, useRef } from 'react';

export type ZoneStat = { bucket: string; fgm: number; fga: number };

const W = 15; // court width, metres
const DEPTH = 11; // how much floor the chart shows
const BASKET = { x: 7.5, d: 1.575 };
const THREE_R = 7.24;
const CORNER_LAT = 7.5 - 0.914; // lateral offset of the corner three line
const CORNER_DEPTH = BASKET.d + Math.sqrt(THREE_R ** 2 - CORNER_LAT ** 2);

function zoneAt(x: number, depth: number): string | null {
  if (depth > DEPTH) return null;
  const lat = x - BASKET.x;
  const fwd = depth - BASKET.d;
  const d = Math.hypot(lat, fwd);
  const deg = Math.abs((Math.atan2(Math.abs(lat), fwd) * 180) / Math.PI);
  const side = lat < 0 ? 'L' : 'R';

  const corner = Math.abs(lat) > CORNER_LAT && depth < CORNER_DEPTH;
  if (corner) return side + '-C3';
  if (d <= 1.4) return 'RIM';
  if (d <= 4.0) return deg <= 35 ? 'M-FL' : side + '-FL';
  if (d > THREE_R) return deg <= 30 ? 'M-ATB' : side + '-ATB';
  if (deg <= 35) return 'M-MR';
  return deg <= 75 ? side + 'W-MR' : side + 'B-MR';
}

function tint(fgm: number, fga: number, avg: number): string {
  const t = Math.max(0, Math.min(1, ((fga ? fgm / fga : 0) - avg + 0.18) / 0.36));
  const f = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  const mix = (a: number, b: number) => Math.round(a + (b - a) * f);
  const rgb =
    t < 0.5
      ? [mix(228, 138), mix(86, 133), mix(74, 120)]
      : [mix(138, 47), mix(133, 191), mix(120, 158)];
  const alpha = (0.45 + 0.3 * Math.abs(t - 0.5) * 2).toFixed(2);
  return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
}

const LABELS: Record<string, [number, number]> = {
  RIM: [7.5, 1.7],
  'M-FL': [7.5, 4.3],
  'L-FL': [5.1, 3.1],
  'R-FL': [9.9, 3.1],
  'M-MR': [7.5, 7.3],
  'LW-MR': [4.0, 5.9],
  'RW-MR': [11.0, 5.9],
  'LB-MR': [2.4, 2.2],
  'RB-MR': [12.6, 2.2],
  'L-C3': [0.8, 2.3],
  'R-C3': [14.2, 2.3],
  'M-ATB': [7.5, 9.7],
  'L-ATB': [2.3, 8.0],
  'R-ATB': [12.7, 8.0],
};

export default function HotZones({ data }: { data: readonly ZoneStat[] }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext('2d');
    if (!ctx) return;

    const S = 56; // px per metre
    cv.width = W * S;
    cv.height = DEPTH * S;
    const X = (x: number) => x * S;
    const Y = (d: number) => (DEPTH - d) * S; // baseline at the bottom

    const stats: Record<string, ZoneStat> = Object.fromEntries(
      data.map(z => [z.bucket, z]),
    );
    const made = data.reduce((t, z) => t + z.fgm, 0);
    const att = data.reduce((t, z) => t + z.fga, 0);
    const avg = att ? made / att : 0.45;

    ctx.fillStyle = '#101113';
    ctx.fillRect(0, 0, cv.width, cv.height);

    const step = 0.05;
    const cols = Math.round(W / step);
    const rows = Math.round(DEPTH / step);
    const grid: (string | null)[][] = [];
    for (let i = 0; i < cols; i++) {
      grid.push([]);
      for (let j = 0; j < rows; j++) {
        grid[i].push(zoneAt((i + 0.5) * step, (j + 0.5) * step));
      }
    }
    const cell = step * S + 1;
    for (let i = 0; i < cols; i++) {
      for (let j = 0; j < rows; j++) {
        const key = grid[i][j];
        const z = key ? stats[key] : undefined;
        if (!z) continue;
        ctx.fillStyle = tint(z.fgm, z.fga, avg);
        ctx.fillRect(X(i * step), Y((j + 1) * step), cell, cell);
      }
    }

    // The strokes between disagreeing neighbours are the zone outlines.
    ctx.strokeStyle = 'rgba(16, 17, 19, .9)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < cols - 1; i++) {
      for (let j = 0; j < rows - 1; j++) {
        if (grid[i][j] !== grid[i + 1][j]) {
          ctx.moveTo(X((i + 1) * step), Y(j * step));
          ctx.lineTo(X((i + 1) * step), Y((j + 1) * step));
        }
        if (grid[i][j] !== grid[i][j + 1]) {
          ctx.moveTo(X(i * step), Y((j + 1) * step));
          ctx.lineTo(X((i + 1) * step), Y((j + 1) * step));
        }
      }
    }
    ctx.stroke();

    // Court furniture, after the zones so it stays visible.
    ctx.strokeStyle = 'rgba(232, 230, 223, .8)';
    ctx.lineWidth = 2;
    const line = (x1: number, d1: number, x2: number, d2: number) => {
      ctx.beginPath();
      ctx.moveTo(X(x1), Y(d1));
      ctx.lineTo(X(x2), Y(d2));
      ctx.stroke();
    };
    line(0, 0, W, 0); // baseline
    line(0, 0, 0, DEPTH);
    line(W, 0, W, DEPTH);
    ctx.strokeRect(X(7.5 - 2.45), Y(5.8), 4.9 * S, 5.8 * S); // the key
    ctx.beginPath();
    ctx.arc(X(7.5), Y(5.8), 1.8 * S, 0, Math.PI * 2); // free-throw circle
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(X(7.5), Y(BASKET.d), 0.23 * S, 0, Math.PI * 2); // rim
    ctx.stroke();
    line(7.5 - 0.9, BASKET.d - 0.375, 7.5 + 0.9, BASKET.d - 0.375); // backboard
    // The arc, swept between where it meets the two corner lines. Canvas
    // measures angles from +x with y downward, and the basket-up direction is
    // -y on screen, so the sweep runs symmetrically about -90 degrees.
    const half = Math.asin(CORNER_LAT / THREE_R);
    ctx.beginPath();
    ctx.arc(X(7.5), Y(BASKET.d), THREE_R * S, -Math.PI / 2 - half, -Math.PI / 2 + half);
    ctx.stroke();
    line(7.5 - CORNER_LAT, 0, 7.5 - CORNER_LAT, CORNER_DEPTH);
    line(7.5 + CORNER_LAT, 0, 7.5 + CORNER_LAT, CORNER_DEPTH);

    for (const z of data) {
      const spot = LABELS[z.bucket];
      if (!spot) continue;
      const cx = X(spot[0]);
      const cy = Y(spot[1]);
      ctx.textAlign = 'center';
      ctx.globalAlpha = 0.75;
      ctx.fillStyle = '#0b0c10';
      ctx.fillRect(cx - 23, cy - 14, 46, 28);
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#e8e6df';
      ctx.font = '600 13px ui-monospace, monospace';
      ctx.fillText(Math.round((100 * z.fgm) / Math.max(1, z.fga)) + '%', cx, cy - 1);
      ctx.fillStyle = '#9a978d';
      ctx.font = '400 10px ui-monospace, monospace';
      ctx.fillText(z.fgm + '-' + z.fga, cx, cy + 11);
    }
  }, [data]);

  return <canvas ref={ref} className="w-full rounded-lg" />;
}
