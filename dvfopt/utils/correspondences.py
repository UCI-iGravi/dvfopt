"""Point correspondence utilities."""

import matplotlib.pyplot as plt
import numpy as np


def remove_duplicates(list1: np.ndarray, list2: np.ndarray):
    """Remove duplicate elements from list1 and corresponding elements from list2."""
    seen = set()
    list1_unique = []
    list2_filtered = []

    for item1, item2 in zip(list1, list2):
        item1_tuple = tuple(item1)
        if item1_tuple not in seen:
            seen.add(item1_tuple)
            list1_unique.append(item1)
            list2_filtered.append(item2)

    return np.array(list1_unique), np.array(list2_filtered)


def orientation(p, q, r):
    """Calculate the orientation of the ordered triplet (p, q, r)."""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def on_segment(p, q, r):
    """Check if point q lies on the line segment pr."""
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False


def do_lines_intersect(p1, q1, p2, q2):
    """Check if two line segments (p1, q1) and (p2, q2) intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    if np.array_equal(p1, p2) or np.array_equal(p1, q2) or np.array_equal(q1, p2) or np.array_equal(q1, q2):
        return False

    if o1 != o2 and o3 != o4:
        return True

    return False


def swap_correspondences(mpts: np.ndarray, fpts: np.ndarray) -> np.ndarray:
    """Swap the correspondences of the moving and fixed points if they intersect."""
    swapped_fpts = np.copy(fpts)
    intersecting_pts = []

    for i in range(len(mpts)):
        line_segment1 = (mpts[i], fpts[i])
        for j in range(i + 1, len(mpts)):
            line_segment2 = (mpts[j], fpts[j])
            if do_lines_intersect(line_segment1[0], line_segment1[1], line_segment2[0], line_segment2[1]):
                swapped_fpts[i] = fpts[j]
                swapped_fpts[j] = fpts[i]
                intersecting_pts.append(fpts[i])
                intersecting_pts.append(fpts[j])
    return swapped_fpts, intersecting_pts


def detect_intersecting_segments(mpts: np.ndarray, fpts: np.ndarray) -> np.ndarray:
    """Detect intersecting line segments in the correspondences."""
    intersecting_indices = []
    intersecting_segments = []
    swapped_segments = []

    for i in range(len(mpts)):
        line_segment1 = (mpts[i], fpts[i])
        for j in range(i + 1, len(mpts)):
            line_segment2 = (mpts[j], fpts[j])
            if do_lines_intersect(line_segment1[0], line_segment1[1], line_segment2[0], line_segment2[1]):
                intersecting_indices.append((i, j))
                intersecting_segments.append((line_segment1, line_segment2))
                swapped_segments.append(((mpts[i], fpts[j]), (mpts[j], fpts[i])))
    intersecting_indices = np.array(intersecting_indices)
    return intersecting_indices, intersecting_segments, swapped_segments


def downsample_points(mpoints_path: str, fpoints_path: str, debug=False):
    """Create a subsample of points to test the Jacobian determinant computation."""
    SLICES = [349, 350, 351]

    mpoints = np.load(mpoints_path)
    fpoints = np.load(fpoints_path)
    mpoints, fpoints = remove_duplicates(mpoints, fpoints)
    fpoints, mpoints = remove_duplicates(fpoints, mpoints)

    mpoints = mpoints[np.isin(mpoints[:, 0], SLICES)]
    fpoints = fpoints[np.isin(fpoints[:, 0], SLICES)]

    mpoints[:, 0] -= 349
    fpoints[:, 0] -= 349

    mpoints = mpoints[::50]
    fpoints = fpoints[::50]

    max_ym = np.max(mpoints[:, 1])
    max_xm = np.max(mpoints[:, 2])
    max_yf = np.max(fpoints[:, 1])
    max_xf = np.max(fpoints[:, 2])
    mpoints[:, 1] = np.round(mpoints[:, 1] / max_ym * 19)
    mpoints[:, 2] = np.round(mpoints[:, 2] / max_xm * 39)
    fpoints[:, 1] = np.round(fpoints[:, 1] / max_yf * 19)
    fpoints[:, 2] = np.round(fpoints[:, 2] / max_xf * 39)

    mpoints, fpoints = remove_duplicates(mpoints, fpoints)
    fpoints, mpoints = remove_duplicates(fpoints, mpoints)

    fpoints, ipts = swap_correspondences(mpoints, fpoints)
    min_intersection_count = len(ipts)
    curr_fpts = fpoints
    while True:
        fpoints, ipts_temp = swap_correspondences(mpoints, fpoints)
        intersection_count = len(ipts_temp)
        if intersection_count < min_intersection_count:
            min_intersection_count = intersection_count
            curr_fpts = fpoints
        else:
            break
    fpoints = curr_fpts

    intersecting_indices, intersecting_segments, swapped_segments = detect_intersecting_segments(mpoints, fpoints)

    new_mpts = mpoints.copy()
    new_fpts = fpoints.copy()
    while len(intersecting_indices) != 0:
        new_mpts = [element for i, element in enumerate(new_mpts) if i not in intersecting_indices[:, 0]]
        new_fpts = [element for i, element in enumerate(new_fpts) if i not in intersecting_indices[:, 0]]
        intersecting_indices, intersecting_segments, swapped_segments = detect_intersecting_segments(new_mpts, new_fpts)
        num_intersections = len(intersecting_indices)
        if num_intersections != 0:
            intersecting_indices, intersecting_segments, swapped_segments = detect_intersecting_segments(new_mpts, new_fpts)
    mpoints = np.array(new_mpts)
    fpoints = np.array(new_fpts)

    if debug:
        for i in range(0, 3):
            curr_mpoints_slice = mpoints[mpoints[:, 0] == i][:, 1:]
            curr_fpoints_slice = fpoints[fpoints[:, 0] == i][:, 1:]

            print("mpoints shape:", curr_mpoints_slice.shape)
            print("fpoints shape:", curr_fpoints_slice.shape)
            plt.figure(figsize=(10, 10))
            plt.scatter(curr_mpoints_slice[:, 1], curr_mpoints_slice[:, 0], c='g')
            plt.scatter(curr_fpoints_slice[:, 1], curr_fpoints_slice[:, 0], c='r')
            for j in range(len(curr_mpoints_slice)):
                plt.plot([curr_mpoints_slice[j][1], curr_fpoints_slice[j][1]],
                         [curr_mpoints_slice[j][0], curr_fpoints_slice[j][0]], color='blue', alpha=0.75)

            plt.gca().invert_yaxis()
            plt.title("Slice " + str(i + 1))
            plt.show()
    return mpoints, fpoints
