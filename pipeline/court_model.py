"""Which court position each keypoint index means, for the reloc2 layout.

A keypoint model returns numbered points. Without the table saying what those
numbers are on the floor, they cannot be used for anything -- and neither the
Roboflow dataset export nor its API carries one. The names in the COCO export are
literally "0", "1", "2".

These eighteen come from the tactical view converter in
github.com/abdullahtarek/basketball_analysis (MIT), which is what the reloc2
dataset was labelled against. Taken rather than re-derived by eye: guessing the
mapping is exactly the kind of mistake that stays silent and corrupts everything
downstream.

Note the model is 28 x 15 -- FIBA -- while the free-throw line sits at 5.79m,
which is the NBA's 19 feet, and the key edges are at 5.18 and 10.0, which are not
symmetric about the halfway width. It is not a self-consistent court. That does
not matter: the labels were drawn against these numbers, so these are the numbers
that make the labels mean what they say. Using a tidier court model here would
put a systematic error into every homography.
"""

COURT_LENGTH = 28.0
COURT_WIDTH = 15.0

# Index -> (x, y) in metres. x runs along the 28m length, y across the 15m width.
RELOC2 = [
    (0.0, 0.0),        # 0   left baseline, one corner
    (0.0, 0.91),       # 1   left baseline, three-point corner inset
    (0.0, 5.18),       # 2   left baseline, key edge
    (0.0, 10.0),       # 3   left baseline, key edge
    (0.0, 14.1),       # 4   left baseline, three-point corner inset
    (0.0, 15.0),       # 5   left baseline, other corner
    (14.0, 15.0),      # 6   halfway line meets one sideline
    (14.0, 0.0),       # 7   halfway line meets the other
    (5.79, 5.18),      # 8   left free-throw line, one end
    (5.79, 10.0),      # 9   left free-throw line, other end
    (28.0, 15.0),      # 10  right baseline, one corner
    (28.0, 14.1),      # 11  right baseline, three-point corner inset
    (28.0, 10.0),      # 12  right baseline, key edge
    (28.0, 5.18),      # 13  right baseline, key edge
    (28.0, 0.91),      # 14  right baseline, three-point corner inset
    (28.0, 0.0),       # 15  right baseline, other corner
    (22.21, 5.18),     # 16  right free-throw line, one end
    (22.21, 10.0),     # 17  right free-throw line, other end
]


def court_point(name):
    """Model output names are the index as a string."""
    try:
        return RELOC2[int(name)]
    except (ValueError, IndexError):
        return None


def selftest():
    assert len(RELOC2) == 18
    xs = sorted({x for x, _ in RELOC2})
    assert xs == [0.0, 5.79, 14.0, 22.21, 28.0], xs
    # The two ends have to mirror each other, or one of them is mistyped.
    left = sorted(y for x, y in RELOC2 if x == 0.0)
    right = sorted(y for x, y in RELOC2 if x == COURT_LENGTH)
    assert left == right, (left, right)
    assert sorted(y for x, y in RELOC2 if x == 5.79) == [5.18, 10.0]
    assert sorted(y for x, y in RELOC2 if x == 22.21) == [5.18, 10.0]
    assert abs((COURT_LENGTH - 22.21) - 5.79) < 1e-9, "free-throw lines are not mirrored"
    # Enough spread in both directions to fix a homography.
    assert len({x for x, _ in RELOC2}) >= 2 and len({y for _, y in RELOC2}) >= 2
    print(f"reloc2 court model ok: {len(RELOC2)} points, "
          f"x in {min(xs)}..{max(xs)}, y in {min(y for _, y in RELOC2)}..{max(y for _, y in RELOC2)}")
    print("  (28x15 court with a 5.79m free-throw line and key edges at 5.18/10.0,")
    print("   copied from the dataset's own converter rather than tidied up)")


if __name__ == "__main__":
    selftest()
