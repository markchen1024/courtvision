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
  frames?: number[];
};

type Conflict = { name: string; a: Track; b: Track; coFrames: number };

// two tracks claiming the same player in the same frames -- at least one is wrong
function findConflicts(tracks: Track[]): Conflict[] {
  const byName = new Map<string, Track[]>();
  for (const t of tracks) {
    if (t.name && (t.status === 'named' || t.status === 'human')) {
      byName.set(t.name, [...(byName.get(t.name) ?? []), t]);
    }
  }
  const out: Conflict[] = [];
  for (const [name, ts] of byName) {
    for (let i = 0; i < ts.length; i++) {
      const fa = new Set(ts[i].frames ?? []);
      for (let j = i + 1; j < ts.length; j++) {
        const co = (ts[j].frames ?? []).filter(f => fa.has(f)).length;
        if (co > 0) out.push({ name, a: ts[i], b: ts[j], coFrames: co });
      }
    }
  }
  return out.sort((x, y) => y.coFrames - x.coFrames);
}

type RosterRow = { num: number | string; name: string };
type Manifest = { clubs: Record<string, RosterRow[]>; tracks: Track[]; builtAt?: number };

const TABS = [
  { key: 'todo', label: 'To review' },
  { key: 'done', label: 'Resolved' },
  { key: 'auto', label: 'Auto-named' },
] as const;
type TabKey = (typeof TABS)[number]['key'];

// 82.6 -> "1:22.6", matching what a video player's timeline shows
const mmss = (s: number) =>
  `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, '0')}`;

const inTab = (t: Track, tab: TabKey) =>
  tab === 'todo' ? t.status === 'number-only' || t.status === 'anonymous'
  : tab === 'done' ? t.status === 'human' || t.status === 'ignored'
  : t.status === 'named';

export default function Review() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [tab, setTab] = useState<TabKey>('todo');
  const [showConflicts, setShowConflicts] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [customNum, setCustomNum] = useState('');
  const [customName, setCustomName] = useState('');

  useEffect(() => {
    const load = () =>
      fetch('/data/review/manifest.json', { cache: 'no-store' })
        .then(r => r.json()).then(setManifest)
        .catch(() => setManifest(null));
    load();
    // The queue is rebuilt by pipeline reruns while a tab sits open; refetch
    // whenever the reviewer comes back to it so they never work a stale list.
    window.addEventListener('focus', load);
    return () => window.removeEventListener('focus', load);
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
  const conflicts = useMemo(
    () => findConflicts(manifest?.tracks ?? []), [manifest]);
  // assignment-time guard: would naming `current` as `name` collide live?
  const wouldConflict = (name: string) => {
    if (!current) return null;
    const mine = new Set(current.frames ?? []);
    for (const t of manifest?.tracks ?? []) {
      if (t.id !== current.id && t.name === name &&
          (t.status === 'named' || t.status === 'human') &&
          (t.frames ?? []).some(f => mine.has(f))) return t;
    }
    return null;
  };
  const [pendingAssign, setPendingAssign] =
    useState<{ name: string; body: Record<string, unknown> } | null>(null);

  async function post(trackId: number, body: Record<string, unknown>) {
    const res = await fetch('/api/review', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trackId, ...body }),
    });
    if (!res.ok) throw new Error(String(res.status));
    setManifest(m => m && {
      ...m,
      tracks: m.tracks.map(t => t.id !== trackId ? t : {
        ...t,
        status: body.action === 'ignore' ? 'ignored'
          : body.action === 'unassign'
            ? (t.ocr.number ? 'number-only' : 'anonymous')
            : 'human',
        number: body.action === 'unassign'
          ? (t.ocr.number ?? `T${t.id}`)
          : (body.number as string) ?? t.number,
        name: body.action === 'assign' ? ((body.name as string) ?? null) : null,
        team: (body.team as Track['team']) ?? t.team,
      }),
    });
  }

  async function decide(body: Record<string, unknown>) {
    if (!current || busy) return;
    setBusy(true);
    try {
      const idx = queue.indexOf(current);
      await post(current.id, body);
      const next = queue[idx + 1] ?? queue[idx - 1];
      setSelected(next && next.id !== current.id ? next.id : null);
      setCustomNum('');
      setCustomName('');
      setPendingAssign(null);
    } finally {
      setBusy(false);
    }
  }

  // conflict view: keep one side, strip the other back into the queue
  async function resolveConflict(c: Conflict, keep: Track) {
    if (busy) return;
    setBusy(true);
    try {
      const strip = keep.id === c.a.id ? c.b : c.a;
      await post(strip.id, { action: 'unassign', ocrNumber: strip.ocr.number });
    } finally {
      setBusy(false);
    }
  }

  const assignRoster = (club: string, row: RosterRow) => {
    const clash = wouldConflict(row.name);
    const body = { action: 'assign', number: String(row.num), name: row.name,
                   team: clubTeam[club] };
    if (clash && pendingAssign?.name !== row.name) {
      setPendingAssign({ name: row.name, body });
      return;
    }
    decide(body);
  };

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
              onClick={() => { setTab(t.key); setSelected(null); setShowConflicts(false); }}>
              {t.label} <b>{counts[t.key]}</b>
            </button>
          ))}
          <button className={showConflicts ? 'on warn' : 'warn'}
            onClick={() => setShowConflicts(true)}>
            Conflicts <b>{conflicts.length}</b>
          </button>
        </nav>
      </header>

      {showConflicts ? (
        <div className="rev-conflicts">
          {conflicts.length === 0 &&
            <div className="rev-empty">no same-frame identity conflicts — clean</div>}
          {conflicts.map(c => (
            <div className="rev-conflict" key={`${c.a.id}-${c.b.id}`}>
              <div className="rev-conflict-head">
                <b>{c.name}</b> claimed by two tracks in {c.coFrames} shared frames
                — pick who really is {c.name}; the other returns to the queue
              </div>
              <div className="rev-conflict-sides">
                {[c.a, c.b].map(t => (
                  <div className="rev-side" key={t.id}>
                    <div className="rev-side-crops">
                      {t.crops.slice(0, 3).map(f => (
                        // eslint-disable-next-line @next/next/no-img-element -- pipeline crops served off local disk; the optimizer would only re-encode one-shot images
                        <img key={f} src={`/data/review/crops/${f}?v=${manifest?.builtAt ?? 0}`} alt="" />
                      ))}
                    </div>
                    <div className="rev-side-meta">
                      track {t.id} · {t.status} · {t.samples} samples
                      {t.span[0] != null && t.span[1] != null &&
                        ` · ${mmss(t.span[0])}–${mmss(t.span[1])}`}
                      {t.ocr.number && ` · OCR #${t.ocr.number}`}
                    </div>
                    <button disabled={busy} onClick={() => resolveConflict(c, t)}>
                      This is {c.name}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
      <div className="rev-cols">
        <aside className="rev-queue">
          {queue.length === 0 && <div className="rev-empty">nothing here</div>}
          {queue.map(t => (
            <button key={t.id}
              className={`rev-item ${current?.id === t.id ? 'on' : ''}`}
              onClick={() => setSelected(t.id)}>
              {t.crops[0]
                // eslint-disable-next-line @next/next/no-img-element -- same local one-shot crops as above
                ? <img src={`/data/review/crops/${t.crops[0]}?v=${manifest?.builtAt ?? 0}`} alt="" />
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
                // eslint-disable-next-line @next/next/no-img-element -- same local one-shot crops as above
                <img key={c} src={`/data/review/crops/${c}?v=${manifest.builtAt ?? 0}`} alt="" />
              ))}
            </div>
            <div className="rev-evidence">
              <span>track <b>{current.id}</b></span>
              <span>OCR read <b>{current.ocr.number ? `#${current.ocr.number}` : '—'}</b></span>
              <span>cluster <b>{current.ocr.club ?? '—'}</b></span>
              <span>{current.samples} samples
                {current.span[0] != null && current.span[1] != null &&
                  ` · ${mmss(current.span[0])}–${mmss(current.span[1])} (${current.span[0]}s)`}</span>
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

            {pendingAssign && (
              <div className="rev-warn">
                <b>{pendingAssign.name}</b> is already assigned to another track
                in the same frames — one of them must be wrong.
                <button disabled={busy}
                  onClick={() => decide(pendingAssign.body)}>
                  Assign anyway
                </button>
                <button className="ghost" onClick={() => setPendingAssign(null)}>
                  Cancel
                </button>
              </div>
            )}

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
      )}
    </div>
  );
}
