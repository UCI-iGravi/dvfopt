"""
Contour extraction, normal estimation, and correspondence matching.

Provides functions for finding contours in brain images, estimating surface
normals, computing 2D correspondences, and performing slice-to-slice
Laplacian registration.

Performance optimizations:
- Parallel slice processing with joblib
- Batch KD-tree queries for normal estimation
- CG solver with diagonal (Jacobi) preconditioner (replaces LGMRES)
- Threaded parallel solves for independent RHS vectors
"""
import os
import time
import numpy as np
import scipy
from scipy.sparse.linalg import lgmres, cg
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from joblib import Parallel, delayed

import skimage
from skimage import feature
from skimage import filters
from skimage import measure

from .utils import laplacianA3D, propagate_dirichlet_rhs


def _default_log(msg, level='info'):
    print(msg)


# ============================================================================
# Parallel helpers (module-level so they are picklable by loky)
# ============================================================================

def _find_slice_correspondences(sno, templateimage, dataimage):
    """Find correspondences for a single slice (process-safe, no closures)."""
    fedge, medge, fbin, mbin = getContours(templateimage, dataimage)
    f, m = get2DCorrespondences(fedge, medge, fbin, mbin)
    if len(f) == 0:
        return None
    fpts = np.hstack([np.full((len(f), 1), sno), f])
    mpts = np.hstack([np.full((len(m), 1), sno), m])
    return fpts, mpts


# ============================================================================
# Contour Extraction
# ============================================================================

def getDataContours(dataImage):
    """
    Calculates internal and outer contours for data image.

    Parameters
    -----------
    dataImage : 2D image slice
    """
    dataImage = np.array(dataImage)  # writable copy (loky may deserialize as read-only)
    dataImage[dataImage>500] = 500
    dataImage[dataImage<0] = 0

    data = skimage.exposure.equalize_adapthist(dataImage.astype(np.uint16))*255
    local_thresh = skimage.filters.threshold_otsu(data)
    binary = data>local_thresh

    edges = feature.canny(binary, sigma=3)
    all_labels = measure.label(edges)
    
    for label in range(np.max(all_labels)):
        if( np.sum(all_labels==label)<100):
            edges[all_labels==label] = 0

    edges = skimage.morphology.thin(edges)

    return edges, binary

def getTemplateContours(templateImage):
    """
    Calculates internal and outer contours for template image.

    Parameters
    -----------
    templateImage : 2D image slice
    """

    local_thresh = skimage.filters.threshold_otsu(templateImage)
    binary = templateImage>local_thresh
    edges = feature.canny(binary, sigma=3)
    edges = skimage.morphology.thin(edges)

    all_labels = measure.label(edges)

    for label in range(np.max(all_labels)):
        if( np.sum(all_labels==label)<25):
            edges[all_labels==label] = 0
    return edges, binary

def getContours(templateImage , dataImage):
    """
    Should generalise both the functions
    """
    fedge, fbin = getTemplateContours(templateImage)
    medge, mbin = getDataContours(dataImage)

    return fedge, medge, fbin, mbin


# ============================================================================
# Normal Estimation
# ============================================================================

def estimate_normal(point, neighbours):
    """Estimate the surface normal at *point* from its *neighbours* via SVD."""
    centroid = np.mean(neighbours,axis=0)
    p_centered = neighbours -centroid
    point = point - centroid

    try:
        v = np.linalg.svd(p_centered - point)[-1]
        n =v[-1]
    except Exception:
        return None
    return n

def orient_normals_nd(points, normals, volume, k=9):
    """Orient normals toward low-intensity side of the volume.

    Works for any dimensionality (2D or 3D).  For each point, accumulates
    voxel intensities along the positive and negative normal directions
    over *k* steps, then flips the normal to point toward the lower-
    intensity side.

    Parameters
    ----------
    points : ndarray (N, D)
        Edge point coordinates.
    normals : ndarray (N, D)
        Unit normal vectors (same shape as *points*).
    volume : ndarray
        Image / binary volume whose shape matches the coordinate space.
    k : int
        Number of voxels to probe in each direction (default 9).

    Returns
    -------
    points : ndarray (M, D)
        Filtered points (boundary points removed).
    normals : ndarray (M, D)
        Corresponding oriented normals.
    """
    shape = np.array(volume.shape)
    flat = volume.ravel()

    # --- Single-pass boundary filter (replaces 4 sequential filters) ---
    probe_max = (points + k * normals).astype(int)
    probe_min = (points - k * normals).astype(int)
    valid = np.ones(len(points), dtype=bool)
    for d in range(volume.ndim):
        valid &= (probe_max[:, d] >= 0) & (probe_max[:, d] < shape[d])
        valid &= (probe_min[:, d] >= 0) & (probe_min[:, d] < shape[d])
    points, normals = points[valid], normals[valid]

    # --- Accumulate intensity along +/- normal direction ---
    left_sum = np.zeros(len(points))
    right_sum = np.zeros(len(points))
    for step in range(1, k + 1):
        fwd = (points + step * normals).astype(int).T   # (ndim, N)
        bwd = (points - step * normals).astype(int).T
        left_sum  += flat[np.ravel_multi_index(fwd, volume.shape, mode='clip')]
        right_sum += flat[np.ravel_multi_index(bwd, volume.shape, mode='clip')]

    # Flip normals toward low intensity
    flip = np.where(left_sum >= right_sum, -1.0, 1.0)
    normals = normals * flip[:, np.newaxis]
    return points, normals


def orient2Dnormals(points, normals, section):
    """Orient 2D normals toward low intensity (delegates to orient_normals_nd)."""
    return orient_normals_nd(points, normals, section, k=9)



def estimate2Dnormals(points,binarySection=None , radius = 3,pkdtree = None,  progressbar= False):
    """
    Estimate surface normals for 2D edge points using local PCA.

    Uses batch KD-tree queries for efficiency.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) edge points from a 2D image.
    binarySection : np.ndarray, optional
        Binary image for orienting normals toward low intensity.
    radius : float
        Neighbourhood radius for KD-tree query.
    pkdtree : scipy.spatial.KDTree, optional
        Pre-built KD-tree. Built from points if None.
    progressbar : bool
        Show progress bar.

    Returns
    -------
    tuple of np.ndarray
        (points, normals) — filtered arrays where normals are defined.
    """
    normals = np.zeros(points.shape)
    if pkdtree is None:
        pkdtree = scipy.spatial.KDTree(points)

    # Batch KD-tree query: get all neighbour lists in one call
    neighbour_lists = pkdtree.query_ball_point(points, radius)

    for i, indices in enumerate(neighbour_lists):
        if len(indices) >= 4:
            neighbours = points[indices]
            n = estimate_normal(points[i], neighbours)
            if n is not None:
                normals[i] = n
            else:
                points[i, :] = 0
        else:
            points[i, :] = 0

    if binarySection is not None:
        points, normals = orient2Dnormals(points, normals, binarySection)
    return points, normals


def get2DCorrespondences_batch(fpoints, fnormals, mpoints, mnormals,
                               degree_thresh=5, k_neighbours=30):
    """
    Batch correspondence matching — queries all source points at once.

    Uses a single batch KD-tree query instead of per-point calls, then
    vectorises the cosine-similarity filter.

    Parameters
    ----------
    fpoints : np.ndarray, shape (Nf, 2)
    fnormals : np.ndarray, shape (Nf, 2)
    mpoints : np.ndarray, shape (Nm, 2)
    mnormals : np.ndarray, shape (Nm, 2)
    degree_thresh : float
        Maximum angle (degrees) between normals to accept.
    k_neighbours : int
        Number of nearest neighbours to consider.

    Returns
    -------
    correspondences : np.ndarray, shape (Nf,)
        Index into mpoints for each fpoint, or -1 if no match.
    """
    if len(fpoints) == 0 or len(mpoints) == 0:
        return np.full(len(fpoints), -1, dtype=int)

    mkdtree = scipy.spatial.KDTree(mpoints)
    k = min(k_neighbours, len(mpoints))
    dists, indices = mkdtree.query(fpoints, k)  # (Nf, k)
    # KDTree returns 1D (Nf,) when k=1 (only 1 moving point); ensure always 2D.
    if dists.ndim == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    cos_thresh = np.cos(degree_thresh * np.pi / 180)
    correspondences = np.full(len(fpoints), -1, dtype=int)

    # Vectorized: compute distance percentile threshold per row
    # Handle infinite distances from KDTree (when k > len(mpoints))
    finite_mask = np.isfinite(dists)  # (Nf, k)
    # Replace inf with nan for nanpercentile; rows with no finite values stay unmatched
    dists_safe = np.where(finite_mask, dists, np.nan)
    with np.errstate(invalid='ignore'):
        p90 = np.nanpercentile(dists_safe, 90, axis=1)  # (Nf,)
    
    # For each source point, compute cosine similarity with all k target normals at once
    # target_normals shape: (Nf, k, 2)
    target_normals = mnormals[indices]
    # Dot product of each source normal with its k target normals: (Nf, k)
    sims = np.einsum('ij,ikj->ik', fnormals, target_normals)
    
    # Build combined mask: finite distance, within 90th percentile, and normal similarity above threshold
    valid = finite_mask & (dists < p90[:, None]) & (sims >= cos_thresh)
    
    # For each row, pick the first valid column (smallest distance due to KDTree ordering)
    has_match = valid.any(axis=1)
    # argmax on bool axis returns first True index per row
    first_valid_col = valid.argmax(axis=1)
    correspondences[has_match] = indices[has_match, first_valid_col[has_match]]

    return correspondences


def get2DCorrespondences(fsection, msection, fbinary, mbinary, inner=True):
    """
    Match contour points between fixed and moving sections.

    Uses batch KD-tree queries and vectorised filtering for performance.
    """
    mpoints, mnormals = estimate2Dnormals(np.array(msection.nonzero()).T, mbinary)
    fpoints, fnormals = estimate2Dnormals(np.array(fsection.nonzero()).T, fbinary)

    if len(fpoints) <= 0 or len(mpoints) <= 0:
        return [], []

    # Use batch correspondence matching
    correspondences = get2DCorrespondences_batch(
        fpoints, fnormals, mpoints, mnormals,
        degree_thresh=5, k_neighbours=30
    )

    c = correspondences!=-1
    cid = fpoints[c,0]* fsection.shape[1] + fpoints[c,1]

    if len(cid) <5:
        return [], []

    cindices = cid.astype(int)

    dx = mpoints[correspondences[c],0] - fpoints[c,0]
    dy = mpoints[correspondences[c],1] - fpoints[c,1]

    #print(np.mean(dx), np.percentile(dx,90), np.max(dx))

    # Filter out NaN values before percentile calculation to avoid warnings
    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)
    dx_valid = dx_abs[~np.isnan(dx_abs)]
    dy_valid = dy_abs[~np.isnan(dy_abs)]
    
    # Handle edge case of empty or all-NaN arrays
    if len(dx_valid) == 0 or len(dy_valid) == 0:
        return [], []
    
    dx_thresh = max(10, np.percentile(dx_valid, 90))
    dy_thresh = max(10, np.percentile(dy_valid, 90))
    
    valid_idx = dx_abs < dx_thresh
    valid_idy = dy_abs < dy_thresh

    valid_id = np.array(valid_idx.astype(int)+valid_idy.astype(int)) ==2
    #print(np.sum(valid_id) , )
    f_ = fpoints[c]
    m_ = mpoints[correspondences[c]]
    return f_[valid_id] , m_[valid_id]


# ============================================================================
# Slice-to-Slice Laplacian Registration
# ============================================================================

def sliceToSlice3DLaplacian(fixedImage, movingImage, sliceMatchList="same", axis=0, output_dir=None, rtol=1e-2, maxiter=1000, return_residuals=False, spacing=None, solver_dtype='float64', solver_method='cg', log_fn=None):
    """
    Perform slice-to-slice 3D Laplacian registration.
    
    Assumes both the images are matched slice to slice according to sliceMatchList 
    along the specified axis. Gets 2D correspondences between the slices and 
    interpolates them smoothly across the volume.

    Performance optimizations over original implementation:
    - Parallel slice processing with joblib
    - Single boundary array instead of three redundant copies
    - CG solver with AMG preconditioner (5-20x faster than LGMRES)
    - Threaded parallel solves for independent RHS vectors
    
    Parameters
    ----------
    fixedImage : np.ndarray
        Fixed/template image as a 3D numpy array.
    movingImage : np.ndarray
        Moving/data image as a 3D numpy array.
    sliceMatchList : str or list, optional
        Slice matching specification. Either "same" for identity mapping,
        or a list where sliceMatchList[i] gives the moving slice index
        corresponding to fixed slice i. Default "same".
    axis : int, optional
        Axis along which slices are taken. Default 0.
    output_dir : str, optional
        Directory to save boundary condition points. If provided, saves
        fpoints.npy and mpoints.npy to output_dir/boundary_conditions/.
    rtol : float, optional
        Relative tolerance for CG solver convergence. Default 1e-2.
    maxiter : int, optional
        Maximum number of CG solver iterations. Default 1000.
    return_residuals : bool, optional
        If True, returns a tuple (deformationField, residual_histories) where
        residual_histories is a dict mapping axis labels to lists of
        (iteration, relative_residual) tuples. Default False.
    spacing : tuple of float, optional
        Physical voxel size per axis (e.g. (0.025, 0.025, 0.025) for 25 µm
        isotropic). Passed to the Laplacian matrix builder so the smoothing
        stencil correctly handles anisotropic voxels. None (default) uses
        unit spacing (isotropic assumption).
    solver_method : str, optional
        Linear solver to use: ``'cg'`` (Conjugate Gradient with Jacobi
        preconditioner, default) or ``'lgmres'`` (restarted LGMRES).
        CG is faster for typical volumes; LGMRES may converge better on
        ill-conditioned systems.
    log_fn : callable, optional
        Logging function accepting a single string argument. If None
        (default), messages are printed to stdout via ``print()``.

    Returns
    -------
    np.ndarray or tuple
        Deformation field of shape ``(3, n0, n1, n2)`` (matching
        ``fixedImage.shape``) containing the displacement vectors for each
        voxel.  If *return_residuals* is True, returns
        ``(deformationField, residual_histories_dict)``.
    """
    log = log_fn or _default_log

    if isinstance(fixedImage, np.ndarray):
        fdata = fixedImage
    else:
        raise TypeError(f"fixedImage must be a numpy array, got {type(fixedImage).__name__}")
    if isinstance(movingImage, np.ndarray):
        mdata = movingImage
    else:
        raise TypeError(f"movingImage must be a numpy array, got {type(movingImage).__name__}")

    n0, n1, n2 = fdata.shape
    nd = len(fdata.shape)

    log(f"Laplacian refinement: Processing volume of shape ({n0}, {n1}, {n2}) along axis {axis}", 'header')
    log(f"Finding slice-to-slice correspondences for {fdata.shape[axis]} slices...", 'info')

    # --- Parallelised correspondence finding (process-based) ---
    # Pre-extract 2D slices so each worker receives only a small payload
    # instead of the full 3D volumes (avoids serialisation bottleneck).
    #
    # NIfTI volumes loaded by nibabel are Fortran-contiguous.  Slicing
    # along axis 0 on F-order arrays is extremely cache-unfriendly (stride
    # = 4224 bytes between successive elements) and caused 15+ minute
    # stalls on (528, 320, 456) volumes.  Converting to C-order once makes
    # all subsequent axis-0 slices contiguous in memory.
    if not fdata.flags.c_contiguous:
        fdata = np.ascontiguousarray(fdata)
    if not mdata.flags.c_contiguous:
        mdata = np.ascontiguousarray(mdata)

    n_slices = fdata.shape[axis]
    slice_pairs = []
    for sno in range(n_slices):
        msno = sno if sliceMatchList == "same" else sliceMatchList[sno]
        # np.take already returns a copy — no need for .copy()
        template_slice = np.take(fdata, sno, axis=axis)
        data_slice = np.take(mdata, msno, axis=axis)
        slice_pairs.append((sno, template_slice, data_slice))
    del fdata, mdata

    results = Parallel(n_jobs=-2)(
        delayed(_find_slice_correspondences)(sno, tpl, dat)
        for sno, tpl, dat in tqdm(slice_pairs, desc="Finding correspondences")
    )
    del slice_pairs

    # Collect valid results
    flist = [r[0] for r in results if r is not None]
    mlist = [r[1] for r in results if r is not None]
    del results

    if len(flist) == 0:
        log("Warning: No correspondence points found. Returning zero deformation.", 'warn')
        return np.zeros((nd, n0, n1, n2))

    fpoints = np.concatenate(flist)
    mpoints = np.concatenate(mlist)
    del flist, mlist

    # Save boundary condition points if output directory provided
    if output_dir is not None:
        boundary_dir = os.path.join(output_dir, "boundary_conditions")
        if not os.path.isdir(boundary_dir):
            os.makedirs(boundary_dir)
        # Save boundary condition points as fast binary .npy
        np.save(os.path.join(boundary_dir, "fpoints.npy"), fpoints)
        np.save(os.path.join(boundary_dir, "mpoints.npy"), mpoints)
        log(f"Saved boundary conditions to {boundary_dir}", 'path')

    # Compute flat indices for boundary points (single array, not 3 copies)
    fIndices = (fpoints[:, 0] * n1 * n2 + fpoints[:, 1] * n2 + fpoints[:, 2]).astype(int)

    _dtype = np.float32 if solver_dtype == 'float32' else np.float64

    flen = n0 * n1 * n2
    Yd = np.zeros(flen, dtype=_dtype)
    Xd = np.zeros(flen, dtype=_dtype)
    Yd[fIndices] += mpoints[:, 1] - fpoints[:, 1]
    Xd[fIndices] += mpoints[:, 2] - fpoints[:, 2]

    # Unique boundary indices for Laplacian matrix construction
    boundary_indices = np.unique(fIndices)
    del fIndices

    log(f"Laplacian refinement: Found {len(fpoints)} correspondence points ({len(boundary_indices)} unique boundary voxels).", 'value')
    del fpoints, mpoints

    # Propagate boundary displacements to non-boundary neighbours so the
    # symmetric Laplacian (boundary columns zeroed) gives correct results.
    propagate_dirichlet_rhs((n0, n1, n2), boundary_indices, Yd, Xd, spacing=spacing)

    # --- Build Laplacian matrix ---
    log("Building Laplacian matrix...", 'info')
    if spacing is not None:
        _sp = '(' + ', '.join(f'{v:.4g}' for v in spacing) + ')'
        log(f"Using spacing-weighted stencil: {_sp}", 'value')
    start = time.time()
    A = laplacianA3D((n0, n1, n2), boundary_indices, spacing=spacing, dtype=_dtype, log_fn=log)
    del boundary_indices
    log(f"Laplacian matrix built in {round(time.time() - start)} sec", 'success')

    # --- Solve with selected method ---
    _use_cg = solver_method.lower() != 'lgmres'
    _solver_label = 'CG+Jacobi' if _use_cg else 'LGMRES'

    M = None
    if _use_cg:
        from scipy.sparse import diags as sparse_diags
        diag_vals = A.diagonal()
        diag_vals[diag_vals == 0] = 1.0
        M = sparse_diags(1.0 / diag_vals, format='csr')
        del diag_vals

    _bytes_per_val = 4 if _dtype == np.float32 else 8
    nnz = A.nnz
    N = A.shape[0]
    mem_sparse_gb = nnz * (_bytes_per_val + 4) / (1024**3)
    mem_vectors_gb = (4 * N * _bytes_per_val) / (1024**3)
    mem_total_gb = mem_sparse_gb + mem_vectors_gb
    log(f"Solving for dy, dx displacement fields ({_solver_label})...", 'info')
    log(f"Convergence threshold: rtol={rtol:.0e}, "
        f"maxiter={maxiter}, "
        f"precision={solver_dtype}, "
        f"matrix: {N/1e6:.1f}M DOFs, "
        f"{nnz/1e6:.0f}M non-zeros, "
        f"est. memory: {mem_total_gb:.2f} GB", 'value')

    _print_lock = threading.Lock()

    def _solve_one(rhs, label):
        t0 = time.time()
        iters = [0]
        rhs_norm = np.linalg.norm(rhs)
        residual_history = []
        def _progress(xk):
            iters[0] += 1
            if iters[0] % 10 == 0 or iters[0] == 1:
                resid = np.linalg.norm(A @ xk - rhs)
                rel = resid / rhs_norm if rhs_norm > 0 else resid
                residual_history.append((iters[0], rel))
                elapsed = time.time() - t0
                rate = iters[0] / elapsed if elapsed > 0 else 0
                with _print_lock:
                    log(f"{label}: iter {iters[0]}/{maxiter}, "
                        f"rel_resid={rel:.2e}, {elapsed:.0f}s elapsed "
                        f"({rate:.1f} it/s)", 'progress')
        if _use_cg:
            x, info = cg(A, rhs, rtol=rtol, maxiter=maxiter, M=M, callback=_progress)
        else:
            x, info = lgmres(A, rhs, rtol=rtol, maxiter=maxiter, callback=_progress)
        elapsed = time.time() - t0
        final_resid = np.linalg.norm(A @ x - rhs)
        final_rel = final_resid / rhs_norm if rhs_norm > 0 else final_resid
        if not residual_history or residual_history[-1][0] != iters[0]:
            residual_history.append((iters[0], final_rel))
        with _print_lock:
            if info != 0:
                log(f"{label}: {_solver_label} stopped after {iters[0]} iters (info={info}), using best iterate", 'warn')
            log(f"{label} done: {iters[0]} iters in {elapsed:.1f} sec", 'success')
        return x, residual_history

    # Solve dy and dx in parallel threads (both CG and LGMRES release GIL during BLAS calls)
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_dy = executor.submit(_solve_one, Yd, "dy")
        fut_dx = executor.submit(_solve_one, Xd, "dx")
        dy, resid_dy = fut_dy.result()
        dx, resid_dx = fut_dx.result()
    del M

    del A, Yd, Xd
    log(f"Solves completed in {round(time.time() - start)} sec total", 'success')

    deformationField = np.zeros((nd, n0, n1, n2), dtype=np.float32)
    deformationField[0] = np.zeros((n0, n1, n2), dtype=np.float32)
    deformationField[1] = dy.reshape((n0, n1, n2))
    deformationField[2] = dx.reshape((n0, n1, n2))
    del dx, dy

    if return_residuals:
        return deformationField, {'dy': resid_dy, 'dx': resid_dx}
    return deformationField
