# One pass on the laptop — then delete this file

Context: the desktop's `web/media` + `web/data` were overwritten during the
Summer League close-out (they now hold the 178s GSW–MEM render), while the
committed homepage and README describe the NYK @ DET 42.6s possession. The
files below exist only on this machine. Everything runs from the repo root.

## 1. Sync

- [ ] `git pull` — today's commits: README visual layer (`assets/readme/`),
      README content fixes.

## 2. Verify the homepage really is NYK @ DET here

- [ ] `ffprobe -v error -show_entries format=duration -of default=nw=1 web/media/nba_ai.mp4`
      → expect **~42.6**, not 178.
- [ ] `cd webapp && npx next dev -p 3100`, open http://localhost:3100 —
      film is NYK @ DET, court dots track it, box score reads Knicks 94–93.
- [ ] Measured section: the reel plays (`web/media/reel.mp4` exists) and its
      poster shows.
- [ ] Timeline tab: ESPN play-by-play renders, the five "on film" rows
      highlight and carry clip timestamps.

## 3. Grab the README proof frame

- [ ] `ffmpeg -ss 18 -i web/media/nba_ai.mp4 -frames:v 1 assets/readme/possession.png`
      — try a few `-ss` values; you want all ten players visible, name chips
      legible, no motion blur. Eyeball it before committing.
- [ ] `git add assets/readme/possession.png`
      `git commit -m "Add a 42.6s-possession frame for the README"`
      `git push`
      → the next session embeds it into the README with a caption.

## 4. Only if the interview demo will run on the desktop instead

- [ ] Copy to the desktop's `web/media/`: `nba_ai.mp4`, `nba.mp4`,
      `reel.mp4`, `reel_poster.jpg`, `poster.jpg`
      (overwrites the Summer League pair there — keep SL copies if wanted).
- [ ] Copy `web/data/nba.json` (the 42.6s version) — the desktop's is SL.
- [ ] Re-run step 2 on the desktop afterwards.

## 5. Clean up

- [ ] `git rm TODO.laptop.md && git commit -m "Done with the laptop pass" && git push`
