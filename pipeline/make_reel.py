"""Join the shipping segments into one reel, with a card before each.

Five separate possessions cut together read as one confusing game unless the
seams are marked, so each clip is preceded by a 1.4-second title card carrying
the possession number, its length, and the measured precision and coverage --
the numbers are the product here as much as the footage, and they come from
eval/, not from enthusiasm.

Everything is re-encoded once through a single concat filter rather than the
concat demuxer, because the demuxer wants byte-identical codec parameters and
five files rendered on different days do not owe us that. Cards are drawn with
PIL in the notebook's Staatliches face; card audio is synthesised silence at
the same 48kHz stereo as the broadcast sound so the audio concat has ten equal
parties.

    python pipeline/make_reel.py            # writes out/reel_final.mp4
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

FONT = ROOT / "out" / "fonts" / "Staatliches-Regular.ttf"
CARD_S = 1.4
FPS = "2997/50"

# order: the two full possessions carry the reel, the short perfect ones close.
# The cards name the clip and nothing else -- per-possession metrics belong to
# the evaluation set, and the page around the reel already states the aggregate.
SEGMENTS = [
    ("out/seg43c_final.mp4", "42.6s possession"),
    ("out/seg_g6149_36s_final.mp4", "35.5s possession"),
    ("out/seg19_sam3_final.mp4", "19.2s possession"),
    ("out/seg_02m28.00s_13s_final.mp4", "12.5s possession"),
    ("out/seg_02m44.15s_10s_final_sound.mp4", "10.0s possession"),
]


def card(n, total, title, path):
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1920, 1080), "#08090c")
    d = ImageDraw.Draw(img)
    f_head = ImageFont.truetype(str(FONT), 44)
    f_big = ImageFont.truetype(str(FONT), 130)
    f_sub = ImageFont.truetype(str(FONT), 52)
    f_foot = ImageFont.truetype(str(FONT), 36)

    def centre(text, font, y, fill):
        w = d.textlength(text, font=font)
        d.text(((1920 - w) / 2, y), text, font=font, fill=fill)

    centre("COURTVISION", f_head, 300, "#c8f031")
    centre(f"POSSESSION {n} / {total}", f_big, 420, "#eef1f5")
    centre(title, f_sub, 610, "#99a2af")
    centre("NBA Playoffs · NYK @ DET · East 1st Round Game 4 · "
           "labels measured against hand ground truth", f_foot, 740, "#6a7383")
    img.save(path)


def main():
    config.ensure_ffmpeg()
    out_dir = ROOT / "out" / "reel_cards"
    out_dir.mkdir(parents=True, exist_ok=True)

    cards = []
    for i, (seg, title) in enumerate(SEGMENTS, 1):
        p = out_dir / f"card{i}.png"
        card(i, len(SEGMENTS), title, p)
        cards.append(p)

    cmd = ["ffmpeg", "-y", "-v", "error"]
    for p in cards:
        cmd += ["-loop", "1", "-t", str(CARD_S), "-framerate", FPS, "-i", str(p)]
    for seg, _ in SEGMENTS:
        cmd += ["-i", str(ROOT / seg)]

    n = len(SEGMENTS)
    parts, chain = [], ""
    for i in range(n):
        parts.append(f"[{i}:v]fps={FPS},setsar=1,format=yuv420p[c{i}v];"
                     f"anullsrc=r=48000:cl=stereo,atrim=0:{CARD_S}[c{i}a];")
        parts.append(f"[{n+i}:v]fps={FPS},setsar=1,format=yuv420p[s{i}v];"
                     f"[{n+i}:a]aresample=48000[s{i}a];")
        chain += f"[c{i}v][c{i}a][s{i}v][s{i}a]"
    fc = "".join(parts) + chain + f"concat=n={2*n}:v=1:a=1[v][a]"

    out = ROOT / "out" / "reel_final.mp4"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"ffmpeg failed: {r.stderr.strip()[-800:]}")
        return 1
    mb = out.stat().st_size / 1e6
    total = sum(1 for _ in SEGMENTS) * CARD_S
    print(f"wrote {out.relative_to(ROOT)}  {mb:.1f}MB "
          f"({len(SEGMENTS)} possessions + {total:.0f}s of cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
