'use client';

// The identity-correction queue, shaped after the flow Veo documents for its
// Lineup screen: AI resolves what it can, a human clears the remainder by
// looking at crops and clicking the roster entry they show. Decisions POST to
// /api/review and are marked identity:"human" -- the UI never launders a hand
// correction into model output.

import { useEffect, useMemo, useState } from 'react';

type Track = {
  id: number;
  status: 'named' | 'number-only' | 'anonymous' | 'human' | 'ignored';
  team?: 'home' | 'away';
  number?: string;
  name?: string | null;
  ocr: { number: string | null; club: string | null;
         votes: Record<string, number> };
  crops: string[];
  samples: number;
  span: [number | null, number | null];
};

type RosterRow = { num: number | string; name: string };
type Manifest = { clubs: Record<string, RosterRow[]>; tracks: Track[] };

const TABS = [
  { key: 'todo', label: 'To review' },
  { key: 'done', label: 'Resolved' },
  { key: 'auto', label: 'Auto-named' },
] as const;
type TabKey = (typeof TABS)[number]['key'];

const inTab = (t: Track, tab: TabKey) =>
  tab === 'todo' ? t.status === 'number-only' || t.status === 'anonymous'
  : tab === 'done' ? t.status === 'human' || t.status === 'ignored'
  : t.status === 'named';

export default function Review() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [tab, setTab] = useState<TabKey>('todo');
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [customNum, setCustomNum] = useState('');
  const [customName, setCustomName] = useState('');

  useEffect(() => {
    fetch('/data/review/manifest.json', { cache: 'no-store' })
      .then(r => r.json()).then(setManifest)
      .catch(() => setManifest(null));
  }, []);

  // Which side of the court each club plays for, learned from the tracks the
  // pipeline already resolved rather than hardcoded.
  const clubTeam = useMemo(() => {
    const votes: Record<string, Record<string, number>> = {};
    for (const t of manifest?.tracks ?? []) {
      if (t.status === 'named' && t.team && t.ocr.club) {
        votes[t.ocr.club] ??= {};
        votes[t.ocr.club][t.team] = (votes[t.ocr.club][t.team] ?? 0) + 1;
      }
    }
    const out: Record<string, 'home' | 'away'> = {};
    for (const [club, v] of Object.entries(votes)) {
      out[club] = (v.home ?? 0) >= (v.away ?? 0) ? 'home' : 'away';
    }
    return out;
  }, [manifest]);

  const queue = useMemo(
    () => (manifest?.tracks ?? []).filter(t => inTab(t, tab)),
    [manifest, tab]);
  const current = queue.find(t => t.id === selected) ?? queue[0] ?? null;

  async function decide(body: Record<string, unknown>) {
    if (!current || busy) return;
    setBusy(true);
    try {
      const res = await fetch('/api/review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trackId: current.id, ...body }),
      });
      if (!res.ok) throw new Error(String(res.status));
      const idx = queue.indexOf(current);
      setManifest(m => m && {
        ...m,
        tracks: m.tracks.map(t => t.id !== current.id ? t : {
          ...t,
          status: body.action === 'ignore' ? 'ignored' : 'human',
          number: (body.number as string) ?? t.number,
          name: (body.name as string) ?? null,
          team: (body.team as Track['team']) ?? t.team,
        }),
      });
      const next = queue[idx + 1] ?? queue[idx - 1];
      setSelected(next && next.id !== current.id ? next.id : null);
      setCustomNum('');
      setCustomName('');
    } finally {
      setBusy(false);
    }
  }

  const assignRoster = (club: string, row: RosterRow) => decide({
    action: 'assign', number: String(row.num), name: row.name,
    team: clubTeam[club],
  });

  if (!manifest) {
    return <div className="rev-shell"><p className="rev-empty">
      No review queue found — run <code>python pipeline/make_review.py</code> first.
    </p></div>;
  }

  const counts = Object.fromEntries(
    TABS.map(t => [t.key, manifest.tracks.filter(x => inTab(x, t.key)).length]));

  return (
    <div className="rev-shell">
      <header className="rev-top">
        <span className="rev-brand"><i />courtvision</span>
        <span className="rev-label">Identity review</span>
        <nav className="rev-tabs">
          {TABS.map(t => (
            <button key={t.key} className={tab === t.key ? 'on' : ''}
              onClick={() => { setTab(t.key); setSelected(null); }}>
              {t.label} <b>{counts[t.key]}</b>
            </button>
          ))}
        </nav>
      </header>

      <div className="rev-cols">
        <aside className="rev-queue">
          {queue.length === 0 && <div className="rev-empty">nothing here</div>}
          {queue.map(t => (
            <button key={t.id}
              className={`rev-item ${current?.id === t.id ? 'on' : ''}`}
              onClick={() => setSelected(t.id)}>
              {t.crops[0]
                ? <img src={`/data/review/crops/${t.crops[0]}`} alt="" />
                : <span className="rev-noimg" />}
              <span className="rev-item-main">
                <span className="rev-item-name">
                  {t.name ?? (t.number ? `#${t.number}` : `T${t.id}`)}
                </span>
                <span className="rev-item-sub">
                  track {t.id} · {t.samples} samples
                </span>
              </span>
              <span className={`rev-chip ${t.status}`}>{t.status}</span>
            </button>
          ))}
        </aside>

        <main className="rev-detail">
          {!current ? <div className="rev-empty">select a track</div> : <>
            <div className="rev-crops">
              {current.crops.map(c => (
                <img key={c} src={`/data/review/crops/${c}`} alt="" />
              ))}
            </div>
            <div className="rev-evidence">
              <span>track <b>{current.id}</b></span>
              <span>OCR read <b>{current.ocr.number ? `#${current.ocr.number}` : '—'}</b></span>
              <span>cluster <b>{current.ocr.club ?? '—'}</b></span>
              <span>{current.samples} samples
                {current.span[0] != null && ` · ${current.span[0]}s–${current.span[1]}s`}</span>
            </div>

            {Object.entries(manifest.clubs).map(([club, roster]) => (
              <section key={club} className="rev-club">
                <h3>{club} <small>{clubTeam[club] ?? ''}</small></h3>
                <div className="rev-roster">
                  {roster.map(row => (
                    <button key={String(row.num)} disabled={busy}
                      className={String(row.num) === current.ocr.number &&
                                 club === current.ocr.club ? 'hint' : ''}
                      onClick={() => assignRoster(club, row)}>
                      <b>#{row.num}</b> {row.name}
                    </button>
                  ))}
                </div>
              </section>
            ))}

            <div className="rev-custom">
              <input placeholder="#" inputMode="numeric" value={customNum}
                onChange={e => setCustomNum(e.target.value.replace(/\D/g, ''))} />
              <input placeholder="name (optional)" value={customName}
                onChange={e => setCustomName(e.target.value)} />
              <button disabled={busy || !customNum}
                onClick={() => decide({ action: 'assign', number: customNum,
                  name: customName || undefined,
                  team: current.ocr.club ? clubTeam[current.ocr.club] : undefined })}>
                Assign
              </button>
              <button className="ghost" disabled={busy}
                onClick={() => decide({ action: 'ignore' })}>
                Not a player
              </button>
            </div>
          </>}
        </main>
      </div>
    </div>
  );
}
