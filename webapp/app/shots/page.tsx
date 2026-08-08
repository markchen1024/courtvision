'use client';

/* Fourteen-zone hot chart in the NBA 2K shapes -- annular sectors around the
   basket -- for the two clubs of the Summer League final.

   The ready-made chart tried first (@pietrus/shotchart.d3.ts) hardcodes its
   own zone shapes in createSectionedZones with no way to swap them, so the
   shapes are drawn by our own component using the grid-classifier machinery
   proven on the static page. Its zone vocabulary is kept, so the data would
   drop straight back into the library if its shapes ever became wanted.

   Zone detail is a placeholder split, but each club's buckets sum to its real
   FG and 3P totals from the ESPN box, so the chart's totals are true even
   where its detail is illustrative. */

import HotZones from './HotZones';
import { useState } from 'react';

type Bucket = { bucket: string; fgm: number; fga: number; percentile: number };

const pct = (m: number, a: number) => Math.round((100 * m) / Math.max(1, a));
const z = (bucket: string, fgm: number, fga: number): Bucket =>
  ({ bucket, fgm, fga, percentile: pct(fgm, fga) });

// GSW: FG 38/74, 10/30 from three. MEM: FG 34/76, 14/37.
const CLUBS = {
  gsw: {
    label: 'Warriors — 94',
    data: [
      z('RIM', 14, 19), z('PAINT', 4, 7), z('FT', 2, 4),
      z('L-MB', 2, 3), z('R-MB', 2, 3),
      z('L-WING', 1, 2), z('R-WING', 1, 2), z('TOPMID', 2, 4),
      z('L-C3', 2, 5), z('R-C3', 2, 4),
      z('L-W3', 2, 7), z('R-W3', 2, 7), z('TOP3', 2, 7),
    ],
  },
  mem: {
    label: 'Grizzlies — 90',
    data: [
      z('RIM', 9, 14), z('PAINT', 3, 6), z('FT', 2, 4),
      z('L-MB', 1, 3), z('R-MB', 1, 3),
      z('L-WING', 1, 3), z('R-WING', 1, 2), z('TOPMID', 2, 4),
      z('L-C3', 3, 5), z('R-C3', 2, 5),
      z('L-W3', 3, 9), z('R-W3', 3, 9), z('TOP3', 3, 9),
    ],
  },
} as const;

export default function ShotsPage() {
  const [club, setClub] = useState<keyof typeof CLUBS>('gsw');
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
      <HotZones data={CLUBS[club].data} />
    </main>
  );
}
