/* The homepage's behaviour, carried over verbatim from web/index.html's
   script block. It talks to the DOM by id exactly as before -- the markup in
   Home.tsx is the same markup -- with only the seams changed: Court comes from
   the shared module instead of a global, Plyr is loaded from the vendored file
   on demand, and everything started here is stopped in the returned cleanup so
   React can mount the page twice in dev without doubling the machinery. */

import * as CourtModule from '@/lib/court';

const Court = CourtModule;

export function initHome() {
  let cancelled = false;
  let rafId = 0;
  let plyr = null;
  const onResize = () => {
    const p = document.getElementById('p-shots');
    if (p && !p.hidden) drawShots();
  };

  const boot = () => {
    if (cancelled) return;
    run();
  };
  if (window.Plyr) boot();
  else {
    const tag = document.createElement('script');
    tag.src = '/vendor/plyr.min.js';
    tag.onload = boot;
    document.head.appendChild(tag);
  }

  function run() {
    const Plyr = window.Plyr;

    const TEAM = { home: '#c8f031', away: '#3bc9a8' };
    // Two renders of the same 42.6s possession (NYK @ DET G4, 3rd quarter),
    // so switching sources never breaks the court sync: the AI view carries
    // club-tinted silhouettes and jersey-OCR name chips burned in by
    // pipeline/render_final.py, measured at 100% precision / 98.8% coverage
    // against hand ground truth (eval/seg43c_truth.json).
    const VIEWS = { ai: '/media/nba_ai.mp4', raw: '/media/nba.mp4' };
    const CLIP = VIEWS.ai;
    const TRAIL = 7;            // ghost frames behind each player

    let DATA = null;
    let SIDE = new Map();       // track id → team, so the draw loop never scans
    let film = null;            // set once the clip is actually playing
    let clockStart = null;      // fallback so the preview always moves

    init();

    async function init(){
      try {
        const res = await fetch('/data/nba.json');
        if (!res.ok) throw new Error(res.status);
        DATA = await res.json();
      } catch {
        document.getElementById('colophonData').textContent = 'no tracking data — run pipeline/project.py';
      }

      if (DATA) {
        const c = DATA.court, v = DATA.video;
        for (const p of DATA.players) SIDE.set(p.id, p.team);
        document.getElementById('figCourt').textContent = `${c.length_m} × ${c.width_m}`;
        document.getElementById('figHz').textContent = `${v.hz} Hz`;
        document.getElementById('figSolved').textContent =
          `${DATA.frames.length}/${Math.round(v.duration * v.hz)}`;
        document.getElementById('colophonData').textContent =
          `${DATA.frames.length} frames · ${v.hz} Hz · ${v.duration.toFixed(0)} s · ${DATA.source}`;
      }

      mountFilm();
      rafId = requestAnimationFrame(tick);
    }

    /* Autoplay the clip. An 80 MB file on a cold cache can take several seconds to
       hand over its first frame, so a bare timeout declares the clip missing while
       it is still on the wire. Give up only when it errors or when nothing is in
       flight — and if it turns up late, take it. */
    function mountFilm(){
      const el = document.getElementById('film');
      const ready = () => {
        film = el;
        clockStart = null;
        document.getElementById('filmFallback').hidden = true;
      };
      // `loadedmetadata` is enough to read currentTime. `playing` is the belt to
      // that pair of braces: if pixels are moving, a "no clip" notice has no
      // business sitting on top of them whatever the rest of this concluded.
      el.addEventListener('loadedmetadata', ready);
      el.addEventListener('playing', ready);
      el.addEventListener('error', () => fallback(true));
      el.src = CLIP;

      // Transport controls. The clip is 3 minutes but the tracking only covers the
      // first stretch of it, so the seek bar carries a marker where the data stops
      // — scrub past it and the court says so rather than freezing without
      // explanation.
      plyr = new Plyr(el, {
        iconUrl: '/vendor/plyr.svg',
        controls: ['play', 'progress', 'current-time', 'duration', 'mute', 'fullscreen'],
        hideControls: false,
        keyboard: { focused: true, global: false },
        tooltips: { controls: false, seek: true },
        markers: DATA ? { enabled: true, points: [{ time: DATA.video.duration, label: 'tracking ends' }] } : undefined,
      });

      // AI view <-> broadcast. Same 42.6s timeline in both files (17ms apart),
      // so the swap only carries the clock across; the court canvas never
      // notices.
      for (const b of document.querySelectorAll('.viewtoggle button')) {
        b.addEventListener('click', () => {
          if (b.classList.contains('on')) return;
          for (const o of document.querySelectorAll('.viewtoggle button')) {
            o.classList.toggle('on', o === b);
            o.setAttribute('aria-selected', String(o === b));
          }
          const t = el.currentTime, playing = !el.paused;
          el.src = VIEWS[b.dataset.view];
          el.addEventListener('loadedmetadata', () => {
            el.currentTime = t;
            if (playing) el.play();
          }, { once: true });
        });
      }

      // Some static servers never fire `error` for a file that isn't there.
      setTimeout(() => {
        if (!film) fallback(el.networkState !== HTMLMediaElement.NETWORK_LOADING);
      }, 2500);
    }

    /* Keep the court moving off an internal clock either way; only claim the clip
       is missing once we are actually sure of it. The notice is positioned over the
       whole stage, so it covers the player without needing to hide it. */
    function fallback(missing){
      if (film) return;
      if (!clockStart) clockStart = performance.now();
      if (missing) document.getElementById('filmFallback').hidden = false;
    }

    function currentTime(){
      const span = DATA ? DATA.video.duration : 100;
      if (film) return film.currentTime;
      if (clockStart) return ((performance.now() - clockStart) / 1000) % span;
      return 0;
    }

    function tick(){
      const t = currentTime();
      draw(t);
      document.getElementById('capClock').textContent = fmt(t);
      rafId = requestAnimationFrame(tick);
    }

    function draw(t){
      const cv = document.getElementById('court');
      const surface = Court.fit(cv);
      if (!surface) return;
      const { ctx, w, h } = surface;

      const L = DATA ? DATA.court.length_m : 28;
      const W = DATA ? DATA.court.width_m : 15;
      const g = Court.geometry(w, h, L, W, 18);

      ctx.clearRect(0, 0, w, h);
      Court.lines(ctx, g, { w, h });
      if (!DATA) return;

      // The clip runs longer than the tracking does. Once you scrub past the data,
      // show an empty floor and say so — leaving everyone frozen on their last
      // known position would read as a tracker that had lost the plot.
      const cap = document.getElementById('capTracks');
      if (t > DATA.video.duration) {
        cap.textContent = `no tracking past ${fmt(DATA.video.duration)}`;
        return;
      }

      const hz = DATA.video.hz || 6;
      const i = Math.max(0, Math.min(Math.round(t * hz), DATA.frames.length - 1));

      // Ghosts first, oldest and faintest at the back, so the current position
      // always sits on top of its own trail.
      for (let k = TRAIL; k >= 1; k--) {
        const f = DATA.frames[i - k];
        if (!f) continue;
        ctx.globalAlpha = 0.06 + 0.045 * (TRAIL - k);
        for (const pos of f.positions) dot(ctx, g, pos, 3.5, false);
      }
      ctx.globalAlpha = 1;

      const frame = DATA.frames[i];
      if (!frame) return;
      for (const pos of frame.positions) dot(ctx, g, pos, 6, true);
      cap.textContent = `${frame.positions.length} track${frame.positions.length === 1 ? '' : 's'}`;
    }

    function dot(ctx, g, pos, r, ring){
      const colour = TEAM[SIDE.get(pos.id) || 'home'];
      ctx.beginPath();
      ctx.arc(g.X(pos.x), g.Y(pos.y), r, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      if (ring) {
        ctx.beginPath();
        ctx.arc(g.X(pos.x), g.Y(pos.y), r + 3.5, 0, Math.PI * 2);
        ctx.globalAlpha = .35; ctx.lineWidth = 1; ctx.strokeStyle = colour; ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    function fmt(t){
      const m = Math.floor(t / 60), s = Math.floor(t % 60);
      return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
    }

    /* ═══════════════════════════════════════════════════════════════════════════
       Stat tabs — one invented game, kept internally consistent.

       Every number below is derived from the roster: the team comparison adds it
       up, the zone splits add back to the same field goals, and the shot chart is
       generated from the zone counts. A marketing page whose box score disagrees
       with its own shot chart is a page nobody looks at twice.
       ═══════════════════════════════════════════════════════════════════════════ */
    const TEAMS = [
      // ESPN box score, gameId 401768043 -- NYK 94 DET 93, East first round
      // game 4, Little Caesars Arena, 27 April 2025. Only players who took
      // the floor. ESPN reports total field goals; two-pointers here are that
      // minus the threes, so every column still adds to the team line:
      // Knicks 34-91 FG, 15-33 3PT, 38 reb, 19 ast. DNPs are omitted.
      { club: 'Knicks', score: 94, tone: 'nyk', roster: [
        { n: 11, name: 'Jalen Brunson',      min: 39, m2: 9, a2: 19, m3: 4, a3: 7, mf: 2, af: 2, oreb: 3, dreb: 2, ast: 11, stl: 0, blk: 1, tov: 2, pf: 3, pm:  5 },
        { n: 32, name: 'Karl-Anthony Towns', min: 40, m2: 5, a2: 16, m3: 5, a3: 7, mf: 2, af: 2, oreb: 1, dreb: 8, ast:  1, stl: 0, blk: 1, tov: 3, pf: 5, pm:  5 },
        { n:  3, name: 'Josh Hart',          min: 42, m2: 1, a2:  4, m3: 3, a3: 5, mf: 3, af: 6, oreb: 2, dreb: 8, ast:  5, stl: 4, blk: 0, tov: 2, pf: 1, pm:  2 },
        { n:  8, name: 'OG Anunoby',         min: 39, m2: 3, a2: 10, m3: 0, a3: 3, mf: 2, af: 2, oreb: 1, dreb: 1, ast:  0, stl: 3, blk: 0, tov: 0, pf: 2, pm: -9 },
        { n: 25, name: 'Mikal Bridges',      min: 44, m2: 1, a2:  6, m3: 2, a3: 6, mf: 0, af: 0, oreb: 2, dreb: 4, ast:  1, stl: 3, blk: 1, tov: 1, pf: 2, pm:  2 },
        { n:  2, name: 'Miles McBride',      min: 19, m2: 0, a2:  0, m3: 1, a3: 5, mf: 2, af: 2, oreb: 0, dreb: 1, ast:  0, stl: 0, blk: 0, tov: 0, pf: 2, pm:  1 },
        { n: 23, name: 'Mitchell Robinson',  min:  9, m2: 0, a2:  1, m3: 0, a3: 0, mf: 0, af: 4, oreb: 3, dreb: 0, ast:  0, stl: 0, blk: 0, tov: 1, pf: 1, pm:  3 },
        { n:  1, name: 'Cameron Payne',      min:  7, m2: 0, a2:  2, m3: 0, a3: 0, mf: 0, af: 0, oreb: 0, dreb: 2, ast:  1, stl: 1, blk: 0, tov: 1, pf: 1, pm: -4 },
        { n: 44, name: 'Landry Shamet',      min:  1, m2: 0, a2:  0, m3: 0, a3: 0, mf: 0, af: 0, oreb: 0, dreb: 0, ast:  0, stl: 0, blk: 0, tov: 0, pf: 0, pm:  0 },
      ]},
      { club: 'Pistons', score: 93, tone: 'det', roster: [
        { n:  2, name: 'Cade Cunningham',    min: 42, m2: 11, a2: 20, m3: 0, a3:  3, mf: 3, af: 4, oreb: 1, dreb:  9, ast: 10, stl: 0, blk: 4, tov: 7, pf: 2, pm:   3 },
        { n: 12, name: 'Tobias Harris',      min: 35, m2:  6, a2: 12, m3: 0, a3:  3, mf: 6, af: 6, oreb: 1, dreb:  7, ast:  0, stl: 1, blk: 1, tov: 1, pf: 5, pm:  -9 },
        { n:  8, name: 'Tim Hardaway Jr.',   min: 37, m2:  1, a2:  3, m3: 4, a3: 13, mf: 0, af: 1, oreb: 1, dreb:  1, ast:  1, stl: 0, blk: 0, tov: 0, pf: 1, pm:   9 },
        { n:  5, name: 'Malik Beasley',      min: 26, m2:  2, a2:  3, m3: 3, a3:  9, mf: 0, af: 0, oreb: 0, dreb:  5, ast:  1, stl: 1, blk: 0, tov: 0, pf: 1, pm:  -1 },
        { n:  9, name: 'Ausar Thompson',     min: 22, m2:  3, a2:  6, m3: 0, a3:  1, mf: 2, af: 2, oreb: 0, dreb:  4, ast:  0, stl: 1, blk: 1, tov: 4, pf: 2, pm:   8 },
        { n:  0, name: 'Jalen Duren',        min: 38, m2:  3, a2:  5, m3: 0, a3:  0, mf: 0, af: 0, oreb: 6, dreb: 11, ast:  4, stl: 0, blk: 2, tov: 3, pf: 3, pm: -17 },
        { n: 17, name: 'Dennis Schroder',    min: 27, m2:  2, a2:  4, m3: 0, a3:  0, mf: 2, af: 2, oreb: 2, dreb:  1, ast:  2, stl: 3, blk: 0, tov: 0, pf: 1, pm:  -4 },
        { n:  7, name: 'Paul Reed',          min:  8, m2:  1, a2:  2, m3: 0, a3:  0, mf: 0, af: 0, oreb: 2, dreb:  2, ast:  0, stl: 0, blk: 1, tov: 0, pf: 1, pm:   8 },
        { n: '00', name: 'Ronald Holland II', min: 6, m2: 0, a2:  0, m3: 0, a3:  0, mf: 1, af: 2, oreb: 1, dreb:  0, ast:  0, stl: 0, blk: 0, tov: 2, pf: 1, pm:  -2 },
      ]},
    ];
    const ROSTER = TEAMS[0].roster;   // the Knicks line, summed into US below

    // Detroit's line, for the comparison column.
    const OPP = { pts: 93, m2: 29, a2: 55, m3: 7, a3: 29, mf: 14, af: 17,
                  oreb: 14, reb: 54, ast: 18, stl: 6, blk: 9, tov: 17, pf: 17 };

    // Splits of the roster's field goals by where they were taken. m/a here sum
    // back to the two- and three-point totals above.
    const ZONESBY = {
      // The zone split is illustrative; the totals are not. ESPN's box score
      // gives field goals and threes but not where they were taken, so each
      // column here is distributed by hand and made to add back to the real
      // line: Knicks 34-91 with 15-33 from three, Pistons 36-84 with 7-29.
      // The totals row of this chart is therefore true while the per-zone
      // breakdown is not, and stays that way until shot events are tracked.
      nyk: [
        { key: 'ra',    name: 'Restricted area',   m:  9, a: 20 },
        { key: 'paint', name: 'Paint (non-RA)',    m:  5, a: 18 },
        { key: 'mid',   name: 'Mid-range',         m:  5, a: 20 },
        { key: 'c3',    name: 'Corner 3',          m:  5, a: 10 },
        { key: 'atb',   name: 'Above the break 3', m: 10, a: 23 },
      ],
      det: [
        { key: 'ra',    name: 'Restricted area',   m: 15, a: 24 },
        { key: 'paint', name: 'Paint (non-RA)',    m:  8, a: 17 },
        { key: 'mid',   name: 'Mid-range',         m:  6, a: 14 },
        { key: 'c3',    name: 'Corner 3',          m:  2, a:  8 },
        { key: 'atb',   name: 'Above the break 3', m:  5, a: 21 },
      ],
    };
    const zonesFor = () => ZONESBY[drawShots.club ?? 'nyk'];

    const CLIPS = [
      // PLACEHOLDER. Event detection is not implemented: nothing in pipeline/
      // reads a shot, a rebound or a turnover out of the footage, and the
      // eight seconds this demo covers contain none of them anyway -- it is a
      // half-court possession at 6:32 of the third.
      //
      // What is real here is only the roster: these are players who were on
      // the floor, identified by the pipeline, on the correct clubs. The
      // events themselves are invented to show the shape of the output, and
      // the panel says so above the table. ShotEventTracker ships in
      // sports@feat/basketball and would replace this.
      { q: 'Q3', clock: '—', kind: 'score', cls: 'score', what: '3PT made',   who: 'Tim Hardaway Jr. · DET', at: '00:01' },
      { q: 'Q3', clock: '—', kind: 'board', cls: 'board', what: 'Rebound',    who: 'Jalen Duren · DET',      at: '00:02' },
      { q: 'Q3', clock: '—', kind: 'score', cls: 'score', what: '2PT made',   who: 'Cade Cunningham · DET',  at: '00:03' },
      { q: 'Q3', clock: '—', kind: 'score', cls: 'score', what: '3PT made',   who: 'Karl-Anthony Towns · NYK', at: '00:04' },
      { q: 'Q3', clock: '—', kind: 'board', cls: 'board', what: 'Rebound',    who: 'Josh Hart · NYK',        at: '00:05' },
      { q: 'Q3', clock: '—', kind: 'score', cls: 'score', what: '2PT missed', who: 'Mikal Bridges · NYK',    at: '00:06' },
      { q: 'Q3', clock: '—', kind: 'loss',  cls: 'loss',  what: 'Turnover',   who: 'Ausar Thompson · DET',   at: '00:07' },
      { q: 'Q3', clock: '—', kind: 'score', cls: 'score', what: '2PT made',   who: 'Jalen Brunson · NYK',    at: '00:08' },
    ];

    const sum = k => ROSTER.reduce((t, p) => t + p[k], 0);
    const pts = p => p.m2 * 2 + p.m3 * 3 + p.mf;
    const pct = (m, a) => a ? (100 * m / a) : 0;
    const pc1 = (m, a) => a ? `${(100 * m / a).toFixed(1)}%` : '—';

    const US = {
      pts: sum('m2') * 2 + sum('m3') * 3 + sum('mf'),
      m2: sum('m2'), a2: sum('a2'), m3: sum('m3'), a3: sum('a3'), mf: sum('mf'), af: sum('af'),
      oreb: sum('oreb'), reb: sum('oreb') + sum('dreb'),
      ast: sum('ast'), stl: sum('stl'), blk: sum('blk'), tov: sum('tov'), pf: sum('pf'),
    };

    buildTabs();
    renderBox();
    for (const host of document.querySelectorAll('.clubs')) {
      host.addEventListener('click', ev => {
        const b = ev.target.closest('button');
        if (!b) return;
        for (const x of host.querySelectorAll('button')) {
          const on = x === b;
          x.classList.toggle('on', on);
          x.setAttribute('aria-selected', on);
        }
        if (b.dataset.club !== undefined) renderBox(+b.dataset.club);
        if (b.dataset.minsclub !== undefined) renderMins(+b.dataset.minsclub);
        if (b.dataset.shotclub !== undefined) {
          drawShots.club = b.dataset.shotclub;
          renderZones();
          drawShots();
        }
      });
    }
    renderCompare();
    renderZones();
    renderClips();
    renderMins();

    /* ── tabs ────────────────────────────────────────────────────────────────── */
    function buildTabs(){
      const tabs = [...document.querySelectorAll('.tabs [role="tab"]')];
      const show = tab => {
        for (const t of tabs) {
          const on = t === tab;
          t.setAttribute('aria-selected', String(on));
          document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
        }
        // The chart has no size to draw into until its panel is on screen.
        if (tab.id === 'tab-shots') drawShots();
      };
      tabs.forEach((tab, i) => {
        tab.addEventListener('click', () => show(tab));
        tab.addEventListener('keydown', e => {
          const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
          if (!step) return;
          e.preventDefault();
          const next = tabs[(i + step + tabs.length) % tabs.length];
          next.focus(); show(next);
        });
      });
      window.addEventListener('resize', onResize);
    }

    /* ── box score ───────────────────────────────────────────────────────────── */
    function renderBox(club){
      // State lives on the function: renderBox is hoisted and gets called above
      // this point in the file, where a `let` would still be in its dead zone --
      // which took every tab down with it, not just this one.
      club = renderBox.club = club ?? renderBox.club ?? 0;
      const pts = p => 2 * p.m2 + 3 * p.m3 + p.mf;

      const section = t => {
        const top = k => Math.max(...t.roster.map(k));
        const topPts = top(pts), topReb = top(p => p.oreb + p.dreb), topAst = top(p => p.ast);
        const sum = k => t.roster.reduce((a, p) => a + p[k], 0);
        const rows = t.roster.map(p => {
          const P = pts(p), R = p.oreb + p.dreb;
          return `<tr>
            <td class="who"><span class="num ${t.tone}">${p.n}</span>${p.name}</td>
            <td>${p.min}</td>
            <td class="${P === topPts ? 'hi' : ''}">${P}</td>
            <td>${p.m2 + p.m3}/${p.a2 + p.a3}</td>
            <td>${p.m3}/${p.a3}</td>
            <td>${p.mf}/${p.af}</td>
            <td>${p.oreb}</td>
            <td>${p.dreb}</td>
            <td class="${R === topReb ? 'hi' : ''}">${R}</td>
            <td class="${p.ast === topAst ? 'hi' : ''}">${p.ast}</td>
            <td>${p.stl}</td>
            <td>${p.blk}</td>
            <td>${p.tov}</td>
            <td>${p.pf}</td>
          </tr>`;
        }).join('');
        // Totals from the rows above, so the table always agrees with itself --
        // ESPN's own totals row did not (4-8 FT under a score that needs eight made).
        return `
          ${rows}
          <tr class="totals">
            <td class="who">Totals</td>
            <td>${sum('min')}</td>
            <td>${t.roster.reduce((a, p) => a + pts(p), 0)}</td>
            <td>${sum('m2') + sum('m3')}/${sum('a2') + sum('a3')}</td>
            <td>${sum('m3')}/${sum('a3')}</td>
            <td>${sum('mf')}/${sum('af')}</td>
            <td>${sum('oreb')}</td>
            <td>${sum('dreb')}</td>
            <td>${sum('oreb') + sum('dreb')}</td>
            <td>${sum('ast')}</td>
            <td>${sum('stl')}</td>
            <td>${sum('blk')}</td>
            <td>${sum('tov')}</td>
            <td>${sum('pf')}</td>
          </tr>`;
      };

      document.getElementById('boxTable').innerHTML = `
        <thead><tr>
          <th>Player</th><th>MIN</th><th>PTS</th><th>FG</th><th>3P</th><th>FT</th>
          <th>OREB</th><th>DREB</th><th>REB</th><th>AST</th><th>STL</th><th>BLK</th><th>TO</th><th>PF</th>
        </tr></thead>
        <tbody>${section(TEAMS[club])}</tbody>`;
    }

    /* ── team comparison ─────────────────────────────────────────────────────── */
    function renderCompare(){
      const legend = `<div class="r">
        <span class="v us">Knicks</span><span></span><span class="k">NYK · DET</span>
        <span></span><span class="v them">Pistons</span></div>`;
      const rows = [
        ['Points',      US.pts,                       OPP.pts,                       false],
        ['Field goal %', pct(US.m2 + US.m3, US.a2 + US.a3), pct(OPP.m2 + OPP.m3, OPP.a2 + OPP.a3), true],
        ['2PT %',       pct(US.m2, US.a2),            pct(OPP.m2, OPP.a2),           true],
        ['3PT %',       pct(US.m3, US.a3),            pct(OPP.m3, OPP.a3),           true],
        ['FT %',        pct(US.mf, US.af),            pct(OPP.mf, OPP.af),           true],
        ['Rebounds',    US.reb,                       OPP.reb,                       false],
        ['Off. rebounds', US.oreb,                    OPP.oreb,                      false],
        ['Assists',     US.ast,                       OPP.ast,                       false],
        ['Steals',      US.stl,                       OPP.stl,                       false],
        ['Blocks',      US.blk,                       OPP.blk,                       false],
        ['Turnovers',   US.tov,                       OPP.tov,                       false],
        ['Fouls',       US.pf,                        OPP.pf,                        false],
      ];

      document.getElementById('compare').innerHTML = legend + rows.map(([k, a, b, isPct]) => {
        const span = isPct ? 100 : Math.max(a, b) || 1;
        const show = v => isPct ? `${v.toFixed(1)}%` : v;
        return `<div class="r">
          <span class="v us">${show(a)}</span>
          <span class="track l"><i style="width:${100 * a / span}%"></i></span>
          <span class="k">${k}</span>
          <span class="track r"><i style="width:${100 * b / span}%"></i></span>
          <span class="v them">${show(b)}</span>
        </div>`;
      }).join('');
    }

    /* ── shot chart ──────────────────────────────────────────────────────────── */
    /* Deterministic, so the chart is the same every load and matches the zone
       table beside it. */
    function lcg(seed){
      let s = seed >>> 0;
      return () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
    }

    function shotPoints(){
      const rnd = lcg(20260807);
      const out = [];
      for (const z of zonesFor()) {
        for (let i = 0; i < z.a; i++) {
              const p = place(z.key, true, rnd);
          out.push({ ...p, made: i < z.m });
        }
      }
      return out;
    }

    function place(zone, left, rnd){
      const bx = left ? 1.575 : 28 - 1.575, by = 7.5;
      const at = (r, deg) => {
        const th = deg * Math.PI / 180;
        return { x: bx + (left ? 1 : -1) * r * Math.cos(th), y: by + r * Math.sin(th) };
      };
      const inKey = p => Math.abs(p.y - by) <= 2.45 && (left ? p.x <= 5.8 : p.x >= 28 - 5.8);

      if (zone === 'ra')    return at(0.2 + rnd() * 1.1, -90 + rnd() * 180);
      if (zone === 'paint') {
        for (let k = 0; k < 24; k++) {
          const p = at(1.4 + rnd() * 4.0, -85 + rnd() * 170);
          if (inKey(p)) return p;
        }
        return at(3.2, -20 + rnd() * 40);
      }
      if (zone === 'mid') {
        for (let k = 0; k < 24; k++) {
          const p = at(2.7 + rnd() * 3.8, -78 + rnd() * 156);
          if (!inKey(p)) return p;
        }
        return at(5.4, 55);
      }
      if (zone === 'c3') {
        const x = 0.4 + rnd() * 2.4;
        return { x: left ? x : 28 - x, y: rnd() < .5 ? 0.35 + rnd() * 0.5 : 14.15 + rnd() * 0.5 };
      }
      return at(6.95 + rnd() * 1.45, -50 + rnd() * 100);   // above the break
    }

        function drawShots(){
      const cv = document.getElementById('shotChart');
      const surface = Court.fit(cv);
      if (!surface) return;
      const { ctx, w, h } = surface;

      /* An upright halfcourt, baseline at the bottom -- the orientation every shot
         chart a basketball reader has ever seen. All attempts land at one basket
         now; a full court with shots at both ends was the tracking view's framing
         leaking into a chart that is really about one offence. */
      const DEPTH = 11;
      const pad = 14;
      const s = Math.min((w - pad * 2) / 15, (h - pad * 2) / DEPTH);
      const ox = (w - 15 * s) / 2, oy = (h - DEPTH * s) / 2;
      const X = lat => ox + lat * s;
      const Y = depth => oy + (DEPTH - depth) * s;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#050608';
      ctx.fillRect(0, 0, w, h);

      const line = '#23272f', strong = '#333a45', rim = '#4a525f';
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = strong;
      ctx.strokeRect(X(0), Y(DEPTH), 15 * s, DEPTH * s);
      ctx.strokeStyle = line;
      ctx.strokeRect(X(7.5 - 2.45), Y(5.8), 4.9 * s, 5.8 * s);
      ctx.beginPath(); ctx.arc(X(7.5), Y(5.8), 1.8 * s, 0, Math.PI * 2); ctx.stroke();
      // The 6.75 arc, swept about straight-up, meeting the corner lines 0.9 in.
      const a = Math.asin((7.5 - 0.9) / 6.75);
      ctx.beginPath();
      ctx.arc(X(7.5), Y(1.575), 6.75 * s, -Math.PI / 2 - a, -Math.PI / 2 + a);
      ctx.stroke();
      const meet = 1.575 + 6.75 * Math.cos(a);
      ctx.beginPath();
      ctx.moveTo(X(0.9), Y(0)); ctx.lineTo(X(0.9), Y(meet));
      ctx.moveTo(X(15 - 0.9), Y(0)); ctx.lineTo(X(15 - 0.9), Y(meet));
      ctx.stroke();
      ctx.strokeStyle = rim;
      ctx.beginPath(); ctx.arc(X(7.5), Y(1.575), 0.225 * s, 0, Math.PI * 2); ctx.stroke();

      // place() gives left-basket full-court coords: x is depth off the baseline,
      // y is lateral -- exactly the two axes of this view.
      for (const sh of shotPoints()) {
        if (sh.x > DEPTH) continue;
        ctx.beginPath();
        ctx.arc(X(sh.y), Y(sh.x), 4.5, 0, Math.PI * 2);
        if (sh.made) { ctx.fillStyle = '#c8f031'; ctx.globalAlpha = .92; ctx.fill(); }
        else { ctx.strokeStyle = '#c8f031'; ctx.globalAlpha = .55; ctx.lineWidth = 1.5; ctx.stroke(); }
        ctx.globalAlpha = 1;
      }

      const m = zonesFor().reduce((t, z) => t + z.m, 0), a2 = zonesFor().reduce((t, z) => t + z.a, 0);
      document.getElementById('shotTotal').textContent = `${m}/${a2} · ${pc1(m, a2)}`;
    }

    function renderZones(){
      document.getElementById('zones').innerHTML = zonesFor().map(z => `
        <div class="zone">
          <div class="top">
            <span class="nm">${z.name}</span>
            <span class="pc">${pc1(z.m, z.a)}</span>
            <span class="of">${z.m}/${z.a}</span>
          </div>
          <div class="bar"><i style="width:${pct(z.m, z.a)}%"></i></div>
        </div>`).join('');
    }

    /* ── timeline ────────────────────────────────────────────────────────────── */
    function renderClips(){
      document.getElementById('clips').innerHTML = CLIPS.map(c => `
        <div class="clip">
          <span class="q">${c.q} ${c.clock}</span>
          <span class="kind ${c.cls}">${c.kind === 'score' ? 'Scoring' : c.kind === 'board' ? 'Rebound' : 'Turnover'}</span>
          <span class="what">${c.what}</span>
          <span class="who">${c.who}</span>
          <span class="at">${c.at}</span>
        </div>`).join('');
    }

    /* ── minutes and impact ──────────────────────────────────────────────────── */
    function renderMins(club){
      // Same hoisting-state pattern as renderBox: called before this line runs.
      club = renderMins.club = club ?? renderMins.club ?? 0;
      const team = TEAMS[club];
      const top = Math.max(...team.roster.map(p => p.min));
      const rows = [...team.roster].sort((a, b) => b.min - a.min).map(p => `
        <tr>
          <td class="who"><span class="num ${team.tone}">${p.n}</span>${p.name}</td>
          <td>${p.min}</td>
          <td><span class="mbar"><i style="width:${100 * p.min / top}%"></i></span></td>
          <td class="${p.pm >= 0 ? 'pm-pos' : 'pm-neg'}">${p.pm > 0 ? '+' : ''}${p.pm}</td>
          <td>${pts(p)}</td>
          <td>${p.oreb + p.dreb}</td>
          <td>${p.ast}</td>
        </tr>`).join('');

      document.getElementById('minsTable').innerHTML = `
        <thead><tr>
          <th>Player</th><th>MIN</th><th></th><th>+/−</th><th>PTS</th><th>REB</th><th>AST</th>
        </tr></thead>
        <tbody>${rows}</tbody>`;
    }
  }

  return () => {
    cancelled = true;
    cancelAnimationFrame(rafId);
    window.removeEventListener('resize', onResize);
    if (plyr) { try { plyr.destroy(); } catch {} }
  };
}
