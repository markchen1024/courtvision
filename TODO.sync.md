# Cross-machine sync — the last physical step, then delete this file

State as of 2026-08-17 evening: everything code-side is done and pushed
through `51750e7` — README hero, the shipped-possession frame, the updated
flowchart, the `finals/` convention. The laptop pass is done. The one job
left is moving the NYK @ DET binaries laptop → desktop; none of them travel
through git (all footage is deliberately untracked — see `finals/README.md`).

## On the laptop

- [ ] `git pull` — brings the README's visual layer, the possession frame
      embed, and the corrected flowchart. Worth one scroll before the
      interview, since this machine is where the repo gets shown.
- [ ] Gather onto a USB stick or share (paths relative to repo root):
  - `web/media/`: `nba_ai.mp4` (62MB) · `nba.mp4` (86MB) · `reel.mp4` (48MB)
    · `poster.jpg` · `reel_poster.jpg`
  - `web/data/`: `nba.json` · `pbp.json`
  - `finals/`: all six renders (~360MB) — the five possession finals plus
    `reel_final.mp4`

## On the desktop

- [ ] Optional: keep the Summer League pair first —
      `copy web\media\nba_ai.mp4 out\sl_backup\` (copy, don't move).
- [ ] Drop the files into the same paths. For `web/media`, **overwrite the
      existing files in place — never delete-then-copy**: the webapp serves
      hardlinks that share inodes with these paths, and a delete breaks the
      link while the old content keeps being served.
- [ ] Quick check both sides of the hardlink:
      `ffprobe -v error -show_entries format=duration -of default=nw=1 web/media/nba_ai.mp4`
      and the same for `webapp/public/media/nba_ai.mp4` → both **~42.6**,
      not 178. If the second still says 178, the link broke — re-make it:
      `New-Item -ItemType HardLink -Path webapp\public\media\nba_ai.mp4 -Target web\media\nba_ai.mp4 -Force`
- [ ] `reel.mp4` and the two posters are new files with no hardlinks yet —
      tell the session to run the homepage verification; it will link them
      into `webapp/public/media/`, start the dev server, check film / court
      sync / box score / reel / timeline with screenshots, then delete this
      file and push.
