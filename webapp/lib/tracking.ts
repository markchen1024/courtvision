/* The shape of what the pipeline writes, and the two computations the viewer
   does with it: cumulative stats up to a moment, and the tracked frame at a
   moment.

   Frames are looked up by time, never by index. Broadcast footage drops out —
   a replay, a bench shot, too little floor in view — so the tracked frames are
   not a uniform grid, and indexing by t*hz drifts the court steadily out of
   step with the video, which is the one thing this view exists to show. */

export interface Player {
  id: number;
  team: 'home' | 'away';
  number: string | number;
  name?: string;
}

export interface Position { id: number; x: number; y: number }
export interface Frame { t: number; positions: Position[] }

export interface GameEvent {
  t: number;
  type: 'shot_made' | 'shot_missed' | 'rebound' | 'assist' | 'turnover';
  player: number;
  team?: 'home' | 'away';
  x?: number;
  y?: number;
  points?: number;
}

export interface TrackingData {
  source: string;
  video: { duration: number; hz: number };
  court: { length_m: number; width_m: number };
  players: Player[];
  frames: Frame[];
  events: GameEvent[];
}

export interface StatLine {
  p: Player;
  pts: number; fgm: number; fga: number; tpm: number; tpa: number;
  reb: number; ast: number; to: number;
}

export interface StatsAt {
  per: Map<number, StatLine>;
  home: number;
  away: number;
  shots: GameEvent[];
}

/* Everything that has happened up to t. Events are sorted by time, so the
   scan stops at the first future event. */
export function statsAt(data: TrackingData, t: number): StatsAt {
  const per = new Map<number, StatLine>();
  for (const p of data.players) {
    per.set(p.id, { p, pts: 0, fgm: 0, fga: 0, tpm: 0, tpa: 0, reb: 0, ast: 0, to: 0 });
  }
  let home = 0;
  let away = 0;
  const shots: GameEvent[] = [];

  for (const e of data.events) {
    if (e.t > t) break;
    const s = per.get(e.player);
    if (!s) continue;
    if (e.type === 'shot_made' || e.type === 'shot_missed') {
      const made = e.type === 'shot_made';
      s.fga++;
      if (made) s.fgm++;
      if (e.points === 3) { s.tpa++; if (made) s.tpm++; }
      if (made && e.points) {
        s.pts += e.points;
        if (e.team === 'home') home += e.points; else away += e.points;
      }
      shots.push(e);
    } else if (e.type === 'rebound') s.reb++;
    else if (e.type === 'assist') s.ast++;
    else if (e.type === 'turnover') s.to++;
  }
  return { per, home, away, shots };
}

/* The tracked frame nearest t, or null when the nearest one is more than a
   sample and a half away — the court then says "no court in view" rather than
   holding a stale position. */
export function frameAt(data: TrackingData, t: number): Frame | null {
  const f = data.frames;
  if (!f.length) return null;
  let lo = 0;
  let hi = f.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (f[mid].t < t) lo = mid + 1; else hi = mid;
  }
  const a = f[lo];
  const b = f[Math.max(0, lo - 1)];
  const near = Math.abs(a.t - t) <= Math.abs(b.t - t) ? a : b;
  const step = 1 / (data.video.hz || 5);
  return Math.abs(near.t - t) <= step * 1.5 ? near : null;
}

export function fmtClock(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}
