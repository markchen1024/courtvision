# One pass on the desktop — then delete this file

Context, as of 2026-08-17: the laptop is the source of truth. Its pass is
done (42.6s NYK @ DET verified end to end, README frame committed, shipped
renders promoted to `finals/`), and everything code-side is pushed. The
desktop's `web/media` + `web/data` still hold the Summer League render;
none of the binaries below travel through git, so they must be copied.
Everything runs from the repo root.

## 1. Sync

- [ ] `git pull` — expect ~14 commits, through `bef6797`. Highlights: court
      data re-projected at 30Hz and cleaned the notebook's way, Measured
      section + reel on the homepage, real ESPN play-by-play in the
      Timeline tab, Vercel Analytics, the `finals/` convention.
      If pull says "Repository not found", that is credentials, not the
      repo: `gh auth login`, then retry.

## 2. Copy the binaries from the laptop (not in git)

- [ ] Into `web/media/` — **write in place** (the webapp junction shares
      inodes); keep SL copies elsewhere first if wanted:
      `nba_ai.mp4` (62MB) · `nba.mp4` (86MB) · `reel.mp4` (48MB) ·
      `poster.jpg` · `reel_poster.jpg`
- [ ] Into `web/data/`: `nba.json` (the 42.6s / 30Hz / cleaned version) ·
      `pbp.json`
- [ ] Into `finals/` (new directory, see its README): all six shipped
      renders, ~360MB — the five possession finals plus `reel_final.mp4`.
      `pipeline/make_reel.py` now reads from here, not `out/`.

## 3. Verify the desktop really is NYK @ DET now

- [ ] `ffprobe -v error -show_entries format=duration -of default=nw=1 web/media/nba_ai.mp4`
      → expect **~42.6**, not 178.
- [ ] `cd webapp && npm i && npx next dev -p 3100`, open
      http://localhost:3100 — film is NYK @ DET, court dots track it,
      box score reads Knicks 94–93, the reel plays in the Measured section.
- [ ] Timeline tab: the ESPN play-by-play renders and opens scrolled to the
      five highlighted "on film" rows (Payne 00:18 … Beasley's three 00:40).

## 4. Nothing to deploy

The live site (courtvision-bubblekids-projects.vercel.app) deploys from a
staged tree on the laptop and is already current — the desktop needs no
Vercel setup. If the interview demo runs locally, step 3's dev server is
the whole job.

## 5. Clean up

- [ ] `git rm TODO.desktop.md && git commit -m "Done with the desktop pass" && git push`
