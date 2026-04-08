"""Geometric self-intersection checker for deformed 2-D quad meshes."""

import numpy as np


# ---------------------------------------------------------------------------
# Segment intersection primitives
# ---------------------------------------------------------------------------

def _cross2d(ox, oy, ax, ay, bx, by):
    """Signed 2-D cross product of vectors (o→a) and (o→b)."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _segs_cross(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """Return True if segment p1-p2 properly intersects segment p3-p4.

    Uses the cross-product sign test.  Collinear / endpoint-touching
    segments return False (shared mesh edges are by definition ok).
    """
    d1 = _cross2d(p3x, p3y, p4x, p4y, p1x, p1y)
    d2 = _cross2d(p3x, p3y, p4x, p4y, p2x, p2y)
    d3 = _cross2d(p1x, p1y, p2x, p2y, p3x, p3y)
    d4 = _cross2d(p1x, p1y, p2x, p2y, p4x, p4y)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _quads_intersect(corners_a, corners_b):
    """Return True if two convex quad polygons intersect.

    Each argument is a ``(4, 2)`` array of ``[y, x]`` deformed corners
    in order TL, TR, BR, BL.

    Tests all 4 × 4 = 16 edge pairs for proper crossing.
    """
    edges_a = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edges_b = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edges_a:
        ay, ax = corners_a[i]
        by, bx = corners_a[j]
        for k, l in edges_b:
            cy, cx = corners_b[k]
            dy, dx = corners_b[l]
            if _segs_cross(ax, ay, bx, by, cx, cy, dx, dy):
                return True
    return False


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def has_quad_self_intersections(phi):
    """Return True if any two non-adjacent deformed quad cells intersect.

    Parameters
    ----------
    phi : ndarray, shape ``(2, H, W)``
        Corrected displacement field ``[dy, dx]``.

    Algorithm
    ---------
    1. Build deformed vertex positions ``Y = row + dy``, ``X = col + dx``.
    2. Compute one AABB per quad cell (H-1)×(W-1).
    3. For each pair of quads with overlapping AABBs and non-adjacent
       indices, run the full edge-edge intersection test.

    Two quads ``(r1,c1)`` and ``(r2,c2)`` are considered *adjacent* when
    ``max(|r1-r2|, |c1-c2|) <= 1`` — they share at least one vertex.
    """
    dy, dx = phi[0], phi[1]
    H, W = dy.shape
    nr, nc = H - 1, W - 1
    if nr <= 0 or nc <= 0:
        return False

    # Deformed coordinates
    rows = np.arange(H, dtype=float)[:, None]   # (H,1)
    cols = np.arange(W, dtype=float)[None, :]   # (1,W)
    Y = rows + dy   # (H, W)
    X = cols + dx   # (H, W)

    # Per-quad AABB: corners TL(r,c), TR(r,c+1), BR(r+1,c+1), BL(r+1,c)
    y_tl = Y[:-1, :-1]; x_tl = X[:-1, :-1]
    y_tr = Y[:-1, 1:];  x_tr = X[:-1, 1:]
    y_br = Y[1:,  1:];  x_br = X[1:,  1:]
    y_bl = Y[1:,  :-1]; x_bl = X[1:,  :-1]

    aabb_ymin = np.minimum(np.minimum(y_tl, y_tr), np.minimum(y_bl, y_br))  # (nr,nc)
    aabb_ymax = np.maximum(np.maximum(y_tl, y_tr), np.maximum(y_bl, y_br))
    aabb_xmin = np.minimum(np.minimum(x_tl, x_tr), np.minimum(x_bl, x_br))
    aabb_xmax = np.maximum(np.maximum(x_tl, x_tr), np.maximum(x_bl, x_br))

    # Flatten to list of (r, c) quads with their data
    # For large grids (nr*nc > ~10k) this loop can be slow, but it is
    # only called as a post-correction check so latency is acceptable.
    n_quads = nr * nc

    # Build flat corner arrays  shape (n_quads, 4, 2)
    corners = np.stack([
        np.stack([y_tl.ravel(), x_tl.ravel()], axis=1),
        np.stack([y_tr.ravel(), x_tr.ravel()], axis=1),
        np.stack([y_br.ravel(), x_br.ravel()], axis=1),
        np.stack([y_bl.ravel(), x_bl.ravel()], axis=1),
    ], axis=1)   # (n_quads, 4, 2)

    ymin_flat = aabb_ymin.ravel()
    ymax_flat = aabb_ymax.ravel()
    xmin_flat = aabb_xmin.ravel()
    xmax_flat = aabb_xmax.ravel()

    for i in range(n_quads):
        ri, ci = divmod(i, nc)
        for j in range(i + 1, n_quads):
            rj, cj = divmod(j, nc)
            # Skip adjacent quads (share at least one vertex)
            if abs(ri - rj) <= 1 and abs(ci - cj) <= 1:
                continue
            # AABB overlap test
            if (ymin_flat[i] > ymax_flat[j] or ymax_flat[i] < ymin_flat[j]
                    or xmin_flat[i] > xmax_flat[j] or xmax_flat[i] < xmin_flat[j]):
                continue
            # Full edge-edge test
            if _quads_intersect(corners[i], corners[j]):
                return True
    return False
