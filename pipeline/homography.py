"""Solve a homography from point correspondences, line correspondences, or both.

Clicking a court landmark means finding one exact pixel where two faint lines
cross. Clicking a court *line* means clicking anywhere along something you can
plainly see. On this footage -- white paint on pale pine, filmed obliquely -- the
second is far easier to do accurately, which is why the sports-field literature
(PnLCalib, ClearLines) fits points and lines together rather than points alone.

The algebra is the dual of the point case. If H maps court points to image
points, a court line l and its image l' satisfy

    H^T l'  =  lambda * l

so l x (H^T l') = 0, which is two independent linear constraints on H -- the same
count a point correspondence gives. Stack both kinds into one DLT system and
solve by SVD.

Two details that matter more than they look:

  normalisation  Court metres run to 28 and pixels to 1920, so an unnormalised
                 DLT is badly conditioned and the fit is visibly worse. Points
                 are Hartley-normalised; lines transform as T^-T, the inverse
                 transpose, because lines are covectors.
  row choice     The three rows of l x m are linearly dependent, and for a line
                 like x=0 (l = (1,0,0)) one of them is identically zero. Dropping
                 the row whose coefficient in that dependency is largest always
                 leaves two independent rows.
"""

import numpy as np


def line_through(p1, p2):
    """Homogeneous line through two points, scaled so (a, b) is a unit vector.

    The scaling is what makes l . x the signed perpendicular distance in pixels,
    which is the only residual for a line that means anything to a human.
    """
    a, b, c = np.cross([p1[0], p1[1], 1.0], [p2[0], p2[1], 1.0])
    n = float(np.hypot(a, b))
    if n < 1e-12:
        raise ValueError("the two points are the same, so they define no line")
    return np.array([a, b, c], np.float64) / n


def normalising_transform(points):
    """Hartley: centroid to the origin, mean distance from it to sqrt(2)."""
    p = np.asarray(points, np.float64).reshape(-1, 2)
    centre = p.mean(axis=0)
    spread = np.linalg.norm(p - centre, axis=1).mean()
    s = np.sqrt(2.0) / spread if spread > 1e-12 else 1.0
    return np.array([[s, 0, -s * centre[0]],
                     [0, s, -s * centre[1]],
                     [0, 0, 1.0]])


def _point_rows(X, x):
    """Two DLT rows for a point correspondence, court (X, Y) -> image (u, v)."""
    (Xc, Yc), (u, v) = X, x
    return [
        [-Xc, -Yc, -1, 0, 0, 0, u * Xc, u * Yc, u],
        [0, 0, 0, -Xc, -Yc, -1, v * Xc, v * Yc, v],
    ]


def _line_rows(l, lp):
    """Two independent rows of l x (H^T l') = 0."""
    a, b, c = l
    ap, bp, cp = lp
    # m = H^T l', written as coefficients on h = [h1..h9] row-major.
    m = [
        [ap, 0, 0, bp, 0, 0, cp, 0, 0],   # m1
        [0, ap, 0, 0, bp, 0, 0, cp, 0],   # m2
        [0, 0, ap, 0, 0, bp, 0, 0, cp],   # m3
    ]
    m = np.array(m, np.float64)
    rows = [b * m[2] - c * m[1],          # r1, coefficient a in the dependency
            c * m[0] - a * m[2],          # r2, coefficient b
            a * m[1] - b * m[0]]          # r3, coefficient c
    drop = int(np.argmax(np.abs([a, b, c])))
    return [r for i, r in enumerate(rows) if i != drop]


def solve(point_pairs=(), line_pairs=()):
    """Least-squares homography mapping court coordinates to image pixels.

    point_pairs: ((X, Y), (u, v)) in metres and pixels.
    line_pairs:  (((X1,Y1),(X2,Y2)), ((u1,v1),(u2,v2))) -- two points defining the
                 court line and two points clicked anywhere along its image.

    Each correspondence of either kind contributes two rows, so four of anything
    is the minimum.
    """
    point_pairs, line_pairs = list(point_pairs), list(line_pairs)
    if 2 * (len(point_pairs) + len(line_pairs)) < 8:
        return None

    court_pts = [P for P, _ in point_pairs] + [p for (seg, _) in line_pairs for p in seg]
    image_pts = [x for _, x in point_pairs] + [p for (_, seg) in line_pairs for p in seg]
    T, Tp = normalising_transform(court_pts), normalising_transform(image_pts)
    # Lines are covectors, so they carry the inverse transpose of the point map.
    Tinv_T, Tpinv_T = np.linalg.inv(T).T, np.linalg.inv(Tp).T

    def to_n(p, M):
        q = M @ [p[0], p[1], 1.0]
        return q[:2] / q[2]

    rows = []
    for P, x in point_pairs:
        rows += _point_rows(to_n(P, T), to_n(x, Tp))
    for seg, iseg in line_pairs:
        rows += _line_rows(Tinv_T @ line_through(*seg), Tpinv_T @ line_through(*iseg))

    A = np.array(rows, np.float64)
    _, s, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)

    # Uniqueness check. The last singular value is the solution direction and is
    # meant to be ~0; the *second* last is what says the answer is unique. With
    # exactly eight rows numpy returns only eight values -- the ninth is implicit
    # -- so pad before indexing, or you end up testing an irrelevant one and
    # accepting configurations that fit their own inputs to 0.00px and are 354px
    # wrong everywhere else. Relative, because the row scale is arbitrary.
    sigma = np.concatenate([s, np.zeros(9 - len(s))])
    if sigma[-2] < 1e-8 * sigma[0]:
        return None

    H = np.linalg.inv(Tp) @ Hn @ T
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def residuals(H, point_pairs=(), line_pairs=()):
    """Pixel error per correspondence, in the same units for both kinds.

    For a line, the error is how far the clicked endpoints sit from where H puts
    the court line -- perpendicular distance in pixels, so it is directly
    comparable to a point's reprojection error.
    """
    out = {"points": [], "lines": []}
    for P, x in point_pairs:
        q = H @ [P[0], P[1], 1.0]
        out["points"].append(float(np.hypot(q[0] / q[2] - x[0], q[1] / q[2] - x[1])))
    Hinv_T = np.linalg.inv(H).T
    for seg, iseg in line_pairs:
        lp = Hinv_T @ line_through(*seg)          # court line pushed into the image
        n = float(np.hypot(lp[0], lp[1]))
        if n < 1e-12:
            out["lines"].append(float("inf"))
            continue
        lp = lp / n
        out["lines"].append(float(np.mean([abs(lp @ [p[0], p[1], 1.0]) for p in iseg])))
    return out


def rms(H, point_pairs=(), line_pairs=()):
    r = residuals(H, point_pairs, line_pairs)
    e = np.array(r["points"] + r["lines"], np.float64)
    return float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")


def selftest():
    """Invent a camera, project through it, solve back, and check we recover it."""
    truth = np.array([[52.0, -9.0, 300.0], [7.0, 31.0, 560.0], [0.0021, -0.0011, 1.0]])

    def proj(p):
        q = truth @ [p[0], p[1], 1.0]
        return (q[0] / q[2], q[1] / q[2])

    corners = [(0.0, 0.0), (28.0, 0.0), (28.0, 15.0), (0.0, 15.0)]
    pts = [(c, proj(c)) for c in corners]

    # Court lines as segments, and image "clicks" taken at different points along
    # them than the segment endpoints -- clicking anywhere on the line must work,
    # which is the entire reason for doing this.
    segs = [((0, 0), (0, 15)), ((28, 0), (28, 15)), ((0, 0), (28, 0)), ((0, 15), (28, 15))]
    lines = []
    for (p1, p2) in segs:
        p1, p2 = np.array(p1, float), np.array(p2, float)
        a, b = p1 + 0.31 * (p2 - p1), p1 + 0.83 * (p2 - p1)
        lines.append(((tuple(p1), tuple(p2)), (proj(a), proj(b))))

    def seg_line(p1, p2, at=(0.31, 0.83)):
        p1, p2 = np.array(p1, float), np.array(p2, float)
        a, b = p1 + at[0] * (p2 - p1), p1 + at[1] * (p2 - p1)
        return ((tuple(p1), tuple(p2)), (proj(a), proj(b)))

    # Halfway and the far sideline: not parallel to each other, and neither runs
    # through the two corner points, so the eight constraints stay independent.
    mixed_lines = [seg_line((14, 0), (14, 15)), seg_line((0, 15), (28, 15))]

    centre = [((14.0, 7.5), proj((14.0, 7.5)))]
    for tag, P, L in [("4 points", pts, []), ("4 lines", [], lines),
                      ("3 points + 2 lines", pts[:3], mixed_lines),
                      ("1 point + 3 lines", centre, lines[:3]),
                      ("mixed, over-determined", pts, lines)]:
        H = solve(P, L)
        assert H is not None, f"{tag}: no solution"
        e = rms(H, P, L)
        # Compare the maps, not the matrices: scale is free.
        probe = [(3.0, 4.0), (14.0, 7.5), (24.0, 11.0), (27.0, 1.0)]
        gap = max(np.hypot(*(np.array(proj(p)) - np.array(
            (lambda q: (q[0] / q[2], q[1] / q[2]))(H @ [p[0], p[1], 1.0])))) for p in probe)
        print(f"  {tag:24s} rms {e:7.4f}px   worst point off-model {gap:7.4f}px")
        assert e < 1e-3 and gap < 1e-3, (tag, e, gap)

    # x=0 is the row that goes identically zero if you pick the rows naively.
    l = line_through((0, 0), (0, 15))
    assert abs(l[0]) > 0.99 and abs(l[1]) < 1e-9 and abs(l[2]) < 1e-9, l
    rows = np.array(_line_rows(l, line_through((10.0, 20.0), (10.0, 400.0))))
    assert np.linalg.matrix_rank(rows) == 2, "degenerate row choice for the x=0 line"
    print("  x=0 line keeps two independent rows")

    # Configurations that do not pin H down have to come back as None rather than
    # as a plausible-looking matrix. Both of these look like enough input.
    par = [(((0, y), (28, y)), (proj((0, y)), proj((28, y)))) for y in (0, 5, 10, 15)]
    assert solve([], par) is None, "four parallel lines were accepted"
    print("  four parallel lines rejected")

    # Eight constraints, but each point lies on one of the lines, so they repeat
    # each other. Found by writing this test wrong the first time.
    assert solve(pts[:2], lines[:2]) is None, "points lying on their own lines were accepted"
    print("  points sitting on the lines they are paired with rejected")

    # The one that got through the old check: two points and two lines that miss
    # them. Fits its own inputs to 0.00px, and is hundreds of pixels wrong away
    # from them, which is the whole reason the check is on sigma[-2] and relative.
    assert solve(pts[:2], mixed_lines) is None, "an eight-constraint ambiguity was accepted"
    print("  2 points + 2 lines rejected as ambiguous (fits perfectly, wrong elsewhere)")

    # A line residual has to be a real perpendicular distance in pixels.
    H = solve(pts, [])
    off = list(lines[0])
    shifted = tuple((u + 12.0, v) for (u, v) in off[1])
    r = residuals(H, [], [(off[0], shifted)])["lines"][0]
    print(f"  a 12px sideways nudge on a vertical line reads as {r:.2f}px")
    assert abs(r - 12.0) < 0.5, r

    print("\nselftest passed")


if __name__ == "__main__":
    selftest()
