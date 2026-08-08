'use client';

/* The zone chart the static page approximated with five rasterised blocks,
   rendered properly by the library built for it: @pietrus/shotchart.d3.ts,
   whose ZonedShotchart draws the standard fourteen-zone halfcourt.

   The library ships React 18 as a plain dependency, which nests a second copy
   of React and dies with an invalid-hook crash the moment it renders. The npm
   overrides in package.json pin the whole tree to one React -- that pin is
   load-bearing; remove it and this page breaks.

   Zone detail is a placeholder split, but each club's buckets sum to its real
   FG and 3P totals from the ESPN box, so the chart's totals are true even
   where its detail is illustrative. */

import { ZonedShotchart } from '@pietrus/shotchart.d3.ts';
// The library extracts its styles to a css file it expects the consumer to
// import -- nothing in its bundle pulls this in, and without it the SVG text
// renders at the default 16 viewBox units on a ~50-unit-wide court: every
// label becomes a wall of hundred-pixel glyphs over the whole chart.
import '@pietrus/shotchart.d3.ts/dist/esm/index.css';
import { useRef, useState } from 'react';

type Bucket = { bucket: string; fgm: number; fga: number; percentile: number };

const pct = (m: number, a: number) => Math.round((100 * m) / Math.max(1, a));
const z = (bucket: string, fgm: number, fga: number): Bucket =>
  ({ bucket, fgm, fga, percentile: pct(fgm, fga) });

// GSW: FG 38/74, 10/30 from three. MEM: FG 34/76, 14/37.
const CLUBS = {
  gsw: {
    label: 'Warriors — 94',
    data: [
      z('RIM', 14, 19),
      z('M-FL', 4, 7), z('L-FL', 2, 3), z('R-FL', 2, 3),
      z('M-MR', 2, 4), z('LW-MR', 1, 2), z('RW-MR', 1, 2), z('LB-MR', 1, 2), z('RB-MR', 1, 2),
      z('L-C3', 2, 5), z('R-C3', 2, 4),
      z('M-ATB', 2, 7), z('L-ATB', 2, 7), z('R-ATB', 2, 7),
    ],
  },
  mem: {
    label: 'Grizzlies — 90',
    data: [
      z('RIM', 9, 14),
      z('M-FL', 2, 4), z('L-FL', 2, 4), z('R-FL', 2, 4),
      z('M-MR', 1, 3), z('LW-MR', 1, 3), z('RW-MR', 1, 3), z('LB-MR', 1, 2), z('RB-MR', 1, 2),
      z('L-C3', 3, 5), z('R-C3', 2, 5),
      z('M-ATB', 3, 9), z('L-ATB', 3, 9), z('R-ATB', 3, 9),
    ],
  },
} as const;

export default function ShotsPage() {
  const [club, setClub] = useState<keyof typeof CLUBS>('gsw');
  const svgRef = useRef<SVGSVGElement>(null);
  return (
    <main className="mx-auto max-w-3xl p-8">
      <h1 className="text-xl font-semibold mb-1">Zone efficiency</h1>
      <p className="text-sm text-neutral-400 mb-4">
        GSW @ MEM · 2026 Summer League final · zone detail is a placeholder split;
        each club&apos;s buckets sum to its real FG and 3P totals.
      </p>
      <div className="flex gap-2 mb-6">
        {(Object.keys(CLUBS) as (keyof typeof CLUBS)[]).map(k => (
          <button
            key={k}
            onClick={() => setClub(k)}
            className={`rounded-full border px-4 py-1.5 text-xs font-mono ${
              club === k ? 'border-lime-400 text-lime-400' : 'border-neutral-700 text-neutral-400'
            }`}
          >
            {CLUBS[k].label}
          </button>
        ))}
      </div>
      {/* Keyed remount: the component redraws into an SVG it owns by id, and
          swapping data in place leaves stale zones behind. */}
      <ZonedShotchart
        key={club}
        id={club === 'gsw' ? 1 : 2}
        courtType="NBA"
        theme="B/O"
        backgroundTheme="Dark"
        svgRef={svgRef as never}
        data={CLUBS[club].data as unknown as never}
      />
    </main>
  );
}
