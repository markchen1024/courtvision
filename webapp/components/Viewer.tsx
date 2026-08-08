'use client';

/* The demo's centre: footage on the left, the measured top-down court on the
   right, moving in step. A port of the static viewer, behaviour first.

   Rendering is split by rate. The court redraws imperatively inside
   requestAnimationFrame, because it follows the video frame-for-frame; the
   React side (clock, box score, play-by-play) re-renders at 4Hz from the same
   loop, because a table that re-renders at 60Hz is just a hotter table. */

import { useCallback, useEffect, useRef, useState } from 'react';
import * as Court from '@/lib/court';
import { TrackingData, frameAt, fmtClock, statsAt } from '@/lib/tracking';

const TEAM: Record<'home' | 'away', string> = { home: '#c8f031', away: '#3bc9a8' };

interface Source {
  id: string;
  label: string;
  video: string;
  data: string;
  note: string;
  game?: { home: string; away: string; title: string };
}

/* Two clips, because the difference between them is the point: the first
   calibrates itself per frame from a keypoint model, and on the second the
   same code cannot, so it runs on a calibration done by hand. */
const SOURCES: Source[] = [
  {
    id: 'nba',
    label: 'NBA Summer League final',
    video: '/media/nba.mp4',
    data: '/data/nba.json',
    note: 'court solved per frame by a keypoint model — no human input',
    game: { home: 'Grizzlies', away: 'Warriors', title: 'GSW 94–90 MEM · Las Vegas · 19 Jul 2026' },
  },
  {
    id: 'bigv',
    label: 'Big V community',
    video: '/media/game.mp4',
    data: '/data/sample.json',
    note: 'same code, court calibrated by hand — the model finds nothing here',
  },
];

export default function Viewer() {
  const [source, setSource] = useState<Source>(SOURCES[0]);
  const [data, setData] = useState<TrackingData | null>(null);
  const [status, setStatus] = useState('loading…');
  const [uiTime, setUiTime] = useState(0);
  const [videoMissing, setVideoMissing] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dataRef = useRef<TrackingData | null>(null);
  const clockStart = useRef<number | null>(null);

  useEffect(() => {
    let dead = false;
    setData(null);
    dataRef.current = null;
    setVideoMissing(false);
    clockStart.current = null;
    fetch(source.data)
      .then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((d: TrackingData) => {
        if (dead) return;
        dataRef.current = d;
        setData(d);
        const gaps = Math.max(0,
          Math.round(d.video.duration * (d.video.hz || 5)) - d.frames.length);
        setStatus(
          (source.game ? source.game.title + ' · ' : '') +
          `${d.source} · ${d.frames.length} tracked frames` +
          (gaps ? `, ${gaps} with no court in view` : '') +
          ` · ${source.note}`,
        );
      })
      .catch(() => { if (!dead) setStatus(`no data for ${source.label} — see the README`); });
    return () => { dead = true; };
  }, [source]);

  /* The clock the court follows: the video when there is one, a wall clock
     when the file is missing, so the preview always moves. */
  const currentTime = useCallback((): number => {
    const v = videoRef.current;
    if (v && !videoMissing && v.readyState >= 2) return v.currentTime;
    const d = dataRef.current;
    if (clockStart.current !== null && d) {
      return ((performance.now() - clockStart.current) / 1000) % d.video.duration;
    }
    return 0;
  }, [videoMissing]);

  useEffect(() => {
    let raf = 0;
    let lastUi = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const d = dataRef.current;
      if (!d) return;
      const t = currentTime();
      drawCourt(d, t);
      const now = performance.now();
      if (now - lastUi > 250) { lastUi = now; setUiTime(t); }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [currentTime]);

  function drawCourt(d: TrackingData, t: number) {
    const cv = canvasRef.current;
    if (!cv) return;
    const surface = Court.fit(cv);
    if (!surface) return;
    const { ctx, w, h } = surface;
    const g = Court.geometry(w, h, d.court.length_m, d.court.width_m, 12);
    ctx.clearRect(0, 0, w, h);
    Court.lines(ctx, g, { w, h });

    for (const e of statsAt(d, t).shots) {
      if (e.x === undefined || e.y === undefined) continue;
      const made = e.type === 'shot_made';
      ctx.beginPath();
      ctx.arc(g.X(e.x), g.Y(e.y), 5.5, 0, Math.PI * 2);
      ctx.globalAlpha = made ? 0.9 : 0.5;
      const colour = TEAM[e.team ?? 'home'];
      if (made) { ctx.fillStyle = colour; ctx.fill(); }
      else { ctx.lineWidth = 1.5; ctx.strokeStyle = colour; ctx.stroke(); }
      ctx.globalAlpha = 1;
    }

    const frame = frameAt(d, t);
    if (!frame) {
      ctx.fillStyle = 'rgba(200,215,225,.55)';
      ctx.font = "500 13px var(--font-geist-mono), ui-monospace, monospace";
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('no court in view — replay or close-up', w / 2, h / 2);
      return;
    }
    const byId = new Map(d.players.map(p => [p.id, p]));
    for (const pos of frame.positions) {
      const p = byId.get(pos.id);
      if (!p) continue;
      ctx.beginPath();
      ctx.arc(g.X(pos.x), g.Y(pos.y), 12, 0, Math.PI * 2);
      ctx.fillStyle = TEAM[p.team];
      ctx.fill();
      ctx.fillStyle = '#08090c';
      ctx.font = "500 11px var(--font-geist-mono), ui-monospace, monospace";
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      // A track id, not a jersey number — reading shirts is OCR, not attempted.
      ctx.fillText(String(p.number).replace(/^T/, ''), g.X(pos.x), g.Y(pos.y) + 0.5);
    }
  }

  const stats = data ? statsAt(data, uiTime) : null;
  const rows = stats
    ? [...stats.per.values()]
        .filter(s => s.fga || s.reb || s.ast || s.to)
        .sort((a, b) => b.pts - a.pts || b.reb - a.reb || a.p.id - b.p.id)
    : [];
  const feed = data
    ? data.events.filter(e => e.t <= uiTime).slice(-24).reverse()
    : [];
  const teamName = (side: 'home' | 'away') =>
    source.game ? source.game[side] : side === 'home' ? 'Home' : 'Away';

  return (
    <div className="mx-auto max-w-6xl px-6 py-6 space-y-5">
      <header className="flex items-center gap-5">
        <h1 className="font-semibold tracking-tight">
          court<span className="text-lime-300">vision</span>
        </h1>
        <div className="ml-auto flex items-center gap-4 font-mono text-sm">
          <span className="text-lime-300">{teamName('home')} {stats?.home ?? 0}</span>
          <span className="text-neutral-600">·</span>
          <span className="text-teal-300">{stats?.away ?? 0} {teamName('away')}</span>
          <span className="text-neutral-500 w-14 text-right">{fmtClock(uiTime)}</span>
        </div>
      </header>

      <div className="flex gap-2">
        {SOURCES.map(s => (
          <button
            key={s.id}
            onClick={() => setSource(s)}
            className={`rounded-full border px-4 py-1.5 text-xs font-mono transition-colors ${
              source.id === s.id
                ? 'border-lime-400/60 text-lime-300'
                : 'border-neutral-800 text-neutral-500 hover:text-neutral-300'
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.15fr_1fr]">
        <section className="rounded-xl border border-neutral-800 overflow-hidden">
          <PanelHead label="Footage" right={source.video} />
          <div className="relative aspect-video bg-black">
            {/* keyed so switching clips swaps the element rather than racing srcs */}
            <video
              key={source.id}
              ref={videoRef}
              className="h-full w-full object-contain"
              src={source.video}
              muted
              playsInline
              loop
              controls
              autoPlay
              preload="auto"
              onError={() => {
                setVideoMissing(true);
                if (clockStart.current === null) clockStart.current = performance.now();
              }}
            />
            {videoMissing && (
              <div className="absolute inset-0 grid place-content-center text-center text-sm text-neutral-500">
                <p className="font-medium text-neutral-300">No footage loaded</p>
                <p>see the README for how to produce one</p>
              </div>
            )}
          </div>
        </section>

        <section className="rounded-xl border border-neutral-800 overflow-hidden flex flex-col">
          <PanelHead label="Court" right={`${stats?.shots.length ?? 0} shots`} />
          <p className="px-4 pt-2 text-[11px] font-mono text-neutral-500">
            positions accurate to about 20cm · labels are track ids, not jersey numbers
          </p>
          <canvas ref={canvasRef} className="min-h-[300px] w-full flex-1" />
        </section>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1.7fr_1fr]">
        <section className="rounded-xl border border-neutral-800 overflow-hidden">
          <PanelHead label="Box score" right="events hand-tagged · names are placeholders" />
          <div className="overflow-x-auto">
            <table className="w-full text-right font-mono text-[13px]">
              <thead>
                <tr className="text-[10px] uppercase tracking-widest text-neutral-500">
                  {['Player', 'PTS', 'FG', '3P', 'REB', 'AST', 'TO'].map((h, i) => (
                    <th key={h} className={`px-3 py-2 border-b border-neutral-800 ${i === 0 ? 'text-left' : ''}`}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(s => (
                  <tr key={s.p.id} className="border-b border-neutral-900 last:border-0">
                    <td className="px-3 py-1.5 text-left font-sans text-neutral-200 whitespace-nowrap">
                      <span
                        className="mr-2 inline-flex h-5 min-w-7 items-center justify-center rounded px-1 text-[11px] text-neutral-950"
                        style={{ background: TEAM[s.p.team] }}
                      >
                        {s.p.number}
                      </span>
                      {s.p.name ?? teamName(s.p.team)}
                    </td>
                    <td className="px-3 text-neutral-100">{s.pts}</td>
                    <td className="px-3 text-neutral-400">{s.fgm}/{s.fga}</td>
                    <td className="px-3 text-neutral-400">{s.tpm}/{s.tpa}</td>
                    <td className="px-3 text-neutral-400">{s.reb}</td>
                    <td className="px-3 text-neutral-400">{s.ast}</td>
                    <td className="px-3 text-neutral-400">{s.to}</td>
                  </tr>
                ))}
                {!rows.length && (
                  <tr><td className="px-3 py-4 text-left text-neutral-500" colSpan={7}>
                    waiting for tip-off…
                  </td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="rounded-xl border border-neutral-800 overflow-hidden">
          <PanelHead label="Play by play" />
          <div className="max-h-72 overflow-y-auto">
            {feed.map((e, i) => {
              const p = data!.players.find(x => x.id === e.player);
              if (!p) return null;
              const who = p.name ?? `#${p.number}`;
              const made = e.type === 'shot_made';
              const txt = {
                shot_made: `${who} scored +${e.points}`,
                shot_missed: `${who} missed ${e.points ?? 2}`,
                rebound: `${who} rebound`,
                assist: `${who} assist`,
                turnover: `${who} turnover`,
              }[e.type];
              return (
                <div key={`${e.t}-${i}`}
                     className="flex items-baseline gap-3 border-b border-neutral-900 px-4 py-2 last:border-0">
                  <span className="font-mono text-xs text-neutral-500 w-10">{fmtClock(e.t)}</span>
                  <span className="h-1.5 w-1.5 flex-none rounded-full"
                        style={{ background: TEAM[p.team] }} />
                  <span className={`text-[13px] ${made ? 'text-neutral-100' : 'text-neutral-400'}`}>
                    {txt}
                  </span>
                </div>
              );
            })}
            {!feed.length && (
              <p className="px-4 py-4 text-sm text-neutral-500">waiting for tip-off…</p>
            )}
          </div>
        </section>
      </div>

      <footer className="font-mono text-xs text-neutral-500">{status}</footer>
    </div>
  );
}

function PanelHead({ label, right }: { label: string; right?: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-neutral-800 px-4 py-2.5">
      <span className="font-mono text-[10px] uppercase tracking-widest text-neutral-400">{label}</span>
      {right && <span className="font-mono text-[11px] text-neutral-500">{right}</span>}
    </div>
  );
}
