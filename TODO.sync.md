# Media sync, third attempt — this time through git. Delete when done.

Two handoffs in a row failed at the same step: the manual binary copy never
happened, and nothing noticed. The JSONs were fixed by tracking them
(`d03801f`); this pass does the same for the five shipped homepage files.
`.gitignore` now lets exactly those five into the repo — everything else
(raw footage, reference downloads, workbench renders) stays out.

## On the laptop — where the correct files are

- [ ] `git pull` — brings the new `.gitignore`.
- [ ] Sanity-check sizes: every file below must be **under 100MB** (GitHub
      hard-blocks at 100MB; the ~50MB warning is noise). Expected:
      `nba_ai.mp4` 62MB · `nba.mp4` 86MB · `reel.mp4` 48MB · two small jpgs.
- [ ] ```
      git add web/media/nba_ai.mp4 web/media/nba.mp4 web/media/reel.mp4 \
              web/media/poster.jpg web/media/reel_poster.jpg
      git commit -m "Ship the homepage media through git"
      git push        # ~200MB, give it a few minutes
      ```
- [ ] `git status` afterwards — the five must show as committed, nothing
      else from `web/media/` staged.

## On the desktop — say the word, the session runs it

The pull alone is **not** enough there: git refuses to overwrite the
untracked Summer League files, and the webapp's hardlinks would keep
serving the old bytes even after they were replaced. The session will:
back up the SL pair to `out/sl_backup/`, move them aside, pull, re-make
all five hardlinks in `webapp/public/media/`, `ffprobe` both sides of
every link (expect 42.6s), then run the homepage verification with
screenshots and delete this file.

## Explicitly out of scope

`finals/` (six renders, ~360MB) is archival and **not needed by the demo**
— the homepage plays the `web/media` encodes. Sync it by USB whenever
convenient, or not at all before the interview. The CRF18 reel master may
exceed 100MB, so it must never be added to git.
