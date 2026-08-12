"""Put the broadcast audio back on a rendered segment.

The renders were silent because every cut along the way passed -an: the
three-minute clip out of the recording, then each segment out of that, so by
the time SAM2 saw it there was no audio left to carry. Commentary matters for
a demo -- it is what makes a possession read as a possession -- and it costs
nothing to restore, because the video does not need re-encoding.

Times compose down the chain. seg_01m48.88s_33s came from det.mp4 at 108.88s,
and det.mp4 came from the recording at 5632.1s, so its audio starts at
5740.98s of the mkv.

    python pipeline/add_audio.py --video out/seg33_final.mp4 \
        --source "d:/game.mkv" --at 5740.98

Writes alongside the input as <name>_sound.mp4 unless --out says otherwise.
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="the silent render")
    ap.add_argument("--source", required=True, help="recording holding the audio")
    ap.add_argument("--at", type=float, required=True,
                    help="seconds into --source where this render begins")
    ap.add_argument("--out")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    out = Path(args.out) if args.out else video.with_name(video.stem + "_sound.mp4")
    if not out.is_absolute():
        out = ROOT / out

    # -c:v copy: the render is already H.264 and re-encoding it would only lose
    # quality. -shortest stops at the video, which is the shorter stream.
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-i", str(video),
           "-ss", f"{args.at:.3f}", "-i", str(args.source),
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-shortest", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"failed: {r.stderr.strip()[:400]}")
        return 1
    mb = out.stat().st_size / 1e6
    print(f"wrote {out.name}  {mb:.1f}MB  (audio from {args.source} at "
          f"{int(args.at)//60:02d}:{args.at % 60:05.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
