'''matrix_svd_control_options and the helpers it calls, copied from mrh/util/la.py
(and the two tolerances from mrh/util/params.py) for the LASSCF migration.
TODO (migration step 5): simplify; only LASCINoSymm._svd1 and LASCISymm._svd1 use it.
'''
import numpy as np
import scipy
import scipy.linalg

num_zero_atol = 1.0e-8
num_zero_rtol = 1.0e-5

def _interpret_shape (mat, spc, axis):
    Pbasis = P = None
    if isinstance (spc, np.ndarray) and spc.ndim == 2:
        Pbasis, P = spc.shape
    elif isinstance (spc, np.ndarray) and np.can_cast (spc, np.bool_):
        Pbasis = spc.size
        P = np.count_nonzero (spc)
    elif isinstance (spc, np.ndarray):
        P = spc.size
    if isinstance (mat, np.ndarray) and mat.ndim == 2:
        Pbasis = mat.shape[axis]
    if Pbasis is not None and P is None: P = Pbasis
    return Pbasis, P

def _symmadapt_subspace (space, symm):
    if symm is None or symm.ndim < 2: return space, symm
    if space.ndim > 1:
        space = space @ np.concatenate (symm, axis=1)
    else:
        space = np.concatenate (symm, axis=1)[space,:]
    symm = np.concatenate ([[idx,] * blk.shape[1] for idx, blk in enumerate (symm)]).astype (int)
    return space, symm

def _symmadapt_recurse_setup (symm_isvectorblock, mat, symm, space, space_isvectorblock, axis, Pbasis):
    if not symm_isvectorblock: return space, symm, mat, None
    symm_umat = np.concatenate (symm, axis=1)
    assert (symm_umat.shape == tuple((Pbasis, Pbasis))), "I can't guess how to map symmetry blocks to different bases"
    symm_lbls = np.concatenate ([idx * np.ones (blk.shape[1], dtype=int) for idx, blk in enumerate (symm)]).astype (int)
    if isinstance (mat, np.ndarray):
        symm_matr = mat @ symm_umat if axis else symm_umat.conjugate ().T @ mat
    else:
        symm_matr = mat * symm_umat if axis else symm_umat.conjugate ().T * mat
    if space is not None:
        # I have to turn the subspace into a vector block!
        if space_isvectorblock: symm_subs = symm_umat.conjugate ().T @ space
        else: symm_subs = symm_umat.conjugate ().T [:,space]
    # Be 200% sure that this recursion can't trigger this conditional block again!
    assert (not (isinstance (symm_lbls[0], np.ndarray))), 'Infinite recursion detected! Fix this bug!'
    return symm_subs, symm_lbls.astype (int), symm_matr, symm_umat

def _unpack_space_symm (space, symm, space_symmetry, space_isvectorblock, atol=num_zero_atol, rtol=num_zero_rtol):
    if space is None:
        symm_lbls = symm
        space = np.ones (M, dtype=np.bool_)
    elif space_symmetry is not None:
        symm_lbls = space_symmetry
    elif space_isvectorblock:
        space, symm_lbls = align_vecs (space, symm, rtol=rtol, atol=atol)
    else:
        symm_lbls = symm[space]
    return space, symm_lbls.astype (int)

def _add_symm_null (vecs, labels, null_lbls, symm_lbls, space, space_isvectorblock, Pbasis): 
    for lbl in null_lbls:
        nnull = symm_lbls==lbl
        if space_isvectorblock: 
            vecs_null = space[:,symm_lbls==lbl]
        else:
            vecs_null = np.eye (Pbasis)[:,space][:,symm_lbls==lbl]
        vecs = np.append (vecs, vecs_null, axis=1)
        labels = np.append (labels, [lbl for ix in range (nnull)]).astype (int)
    return vecs, labels

def matrix_svd_control_options (the_matrix, full_matrices=False, only_nonzero_vals=False, sort_vecs=-1,
    lspace=None, rspace=None, lsymm=None, rsymm=None, symmetry=None,
    lspace_symmetry=None, rspace_symmetry=None, strong_symm=False,
    num_zero_rtol=num_zero_rtol, num_zero_atol=num_zero_atol):
    ''' Perform SVD of a matrix using scipy's linalg driver with a lot of pre-/post-processing

        Args:
            the_matrix: ndarray of shape (M,N) or scalar
                If a scalar, is treated as I * the_matrix, where I has length P given by the
                number of rows in "lspace" if lspace is provided as a vector block or
                the number of rows in "rspace" otherwise.
                

        Kwargs:
            full_matrices: logical
                If true, lvecs and rvecs include the null space and have shapes (M,M) and (N,N).
                Otherwise, lvecs and rvecs omit the null space and have shapes (M,K) and (N,K).
            only_nonzero_vals: logical
                If true, the formal [max (M,N) - K] and numerical (sval == 0)
                null spaces are both omitted from the returned singular values
                and possibly (depending on the value of full_matrices)
                the left- and right-singular vectors: K = count_nonzero (svals).
                Otherwise: only the formal null space is omitted: K = min (M,N)
            sort_vecs: integer
                Defines overall sorting of non-degenerate eigenvalues. -1 means from largest
                to smallest; +1 means from smallest to largest. Play with other values at
                your peril
            lspace: index list for accessing Mprime elements array of shape (M,)
            or ndarray of shape (M,Mprime)
            or None
                Defines a subspace for the rows in which SVD is performed.
            rspace: index list for accessing Nprime elements array of shape (N,)
            or ndarray of shape (N,Nprime)
            or None
                Defines a subspace for the columns in which SVD is performed.
            lsymm: list of block labels of length M
            or list of non-square matrices of shape (M,P), sum_P = M
            or None
                Formal symmetry information for the rows of the matrix. Neither the matrix nor the
                subspace need to be symmetry adapted, and unless strong_symm=True,
                symmetry is used only to define a gauge convention within
                degenerate manifolds. Orthonormal linear combinations of coupled left- and right-singular
                vectors with the highest possible projections onto any symmetry
                block are sequentially generated using repeated singular value decompositions
                in successively smaller subspaces. Vectors within degenerate
                manifolds are therefore sorted from least to most symmetry-breaking;
                quantitatively symmetry-adapted vectors are grouped by block 
                with the blocks in a currently arbitrary order.
            rsymm: list of block labels of length N
            or list of non-square matrices of shape (N,P), sum_P = N
            or None
                Formal symmetry information for the columns of the matrix. See lsymm
            symmetry: list of block labels of length ?
            or list of non-square matrices of shape (?,P), sum_P = ?
            or None
                Convenience argument to set both lsymm and rsymm to the same value. Overridden
                by either lsymm or rsymm. User beware: no dimensionality checking.
            lspace_symmetry: list of block labels of length Mprime
            or list of non-square matrices of shape (Mprime,P), sum_P = Mprime
            or None
                Formal symmetry information of the row subspace. Set to None if lspace is None.
                If not None, "lsymm" is ignored (overrides "symmetry"). 
            rspace_symmetry: list of block labels of length Nprime
            or list of non-square matrices of shape (Nprime,P), sum_P = Nprime
            or None
                Formal symmetry information of the column subspace. Set to None if rspace is None.
                If not None, "rsymm" is ignored (overrides "symmetry").
            num_zero_atol: float
                Absolute tolerance for numpy's "isclose" function and its relatives.
            num_zero_rtol: float
                Relative tolerance for numpy's "isclose" function and its relatives.

        Returns:
            lvecs: ndarray of shape (M,M) or (M,K), K <= M (see full_matrices and only_nonzero_vals kwargs)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning (Mprime -> M)
            svals: ndarray of shape (K,), K <= min (M,N) (see only_nonzero_vals kwarg)
            rvecs: ndarray of shape (N,N) or (N,K), K <= N (see full_matrices and only_nonzero_vals kwargs)
                If a subspace is specified, the eigenvectors are transformed back into the original
                basis before returning (Nprime -> N)
            llabels: list of length M or K, K <= M
                Only returned if lsymm or symmetry is not None. Identifies symmetry block
                with the highest possible projection onto each left-singular vector.
                Does not guarantee that the vectors are symmetry-adapted.
            rlabels: list of length N or K, K <= N
                Only returned if rsymm or symmetry is not None. Identifies symmetry block
                with the highest possible projection onto each left-singular vector.
                Does not guarantee that the vectors are symmetry-adapted.


    '''

    # Interpret subspace information
    lspace = None if lspace is None else np.asarray (lspace)
    lspace_isvectorblock = False if lspace is None else lspace.ndim == 2
    rspace = None if rspace is None else np.asarray (rspace)
    rspace_isvectorblock = False if rspace is None else rspace.ndim == 2

    # Interpret subspace symmetry information
    if strong_symm:
        if lsymm is None and symmetry is None and lspace_symmetry is None: 
            raise RuntimeError ("Can't do SVD by symmetry blocks if no row symmetry information provided!")
        if rsymm is None and symmetry is None and rspace_symmetry is None: 
            raise RuntimeError ("Can't do SVD by symmetry blocks if no column symmetry information provided!")
    if lspace is None: lspace_symmetry = None
    if rspace is None: rspace_symmetry = None
    return_llabels = not (lsymm is None) or not (symmetry is None) or not (lspace_symmetry is None)
    return_rlabels = not (rsymm is None) or not (symmetry is None) or not (rspace_symmetry is None)
    lspace_symm_isvectorblock = False if lspace_symmetry is None else isinstance (rspace_symmetry[0], np.ndarray)
    rspace_symm_isvectorblock = False if rspace_symmetry is None else isinstance (rspace_symmetry[0], np.ndarray)
    if lspace_symmetry is not None: lsymm = None
    if rspace_symmetry is not None: rsymm = None

    # Interpret full-space symmetry information
    if lsymm is None and lspace_symmetry is None and symmetry is not None: lsymm = symmetry
    if rsymm is None and rspace_symmetry is None and symmetry is not None: rsymm = symmetry
    lsymm_isvectorblock = False if lsymm is None else isinstance (lsymm[0], np.ndarray)
    rsymm_isvectorblock = False if rsymm is None else isinstance (rsymm[0], np.ndarray)

    # If subspace symmetry is provided as a vector block, transform subspace into a symmetry-adapted form
    # No recursion necessary because the eigenvectors are meant to be provided in the full basis :)
    lspace, lspace_symmetry = _symmadapt_subspace (lspace, lspace_symmetry)
    rspace, rspace_symmetry = _symmadapt_subspace (rspace, rspace_symmetry)

    # Shape construction and zero matrix escape
    Mbasis, M = _interpret_shape (the_matrix, lspace, 0)
    Nbasis, N = _interpret_shape (the_matrix, rspace, 1)
    if Mbasis is None and Nbasis is None: raise RuntimeError ("Insufficient information to determine shape of matrix")
    if Nbasis is None: Nbasis = Mbasis
    if Mbasis is None: Mbasis = Nbasis
    if M is None: M = Mbasis
    if N is None: N = Nbasis
    K = min (M, N)

    # If symmetry information is provided as a vector block, transform into a symmetry-adapted basis and recurse
    if lsymm_isvectorblock or rsymm_isvectorblock:
        lsubs, lsymm, symm_matr, lumat = _symmadapt_recurse_setup (lsymm_isvectorblock, the_matrix, lsymm, lspace, lspace_isvectorblock, 0, Mbasis)
        rsubs, rsymm, symm_matr, rumat = _symmadapt_recurse_setup (rsymm_isvectorblock, symm_matr,  rsymm, rspace, rspace_isvectorblock, 1, Nbasis)
        rets = matrix_svd_control_options (symm_matr, only_nonzero_vals=only_nonzero_vals,
            full_matrices=full_matrices, sort_vecs=sort_vecs, strong_symm=strong_symm,
            lspace=lsubs, rspace=rsubs, lsymm=lsymm, rsymm=rsymm, symmetry=None,
            lspace_symmetry=lspace_symmetry, rspace_symmetry=rspace_symmetry,
            num_zero_atol=num_zero_atol, num_zero_rtol=num_zero_rtol)
        if lumat is not None: rets = [lumat @ x if idx == 0 else x for idx, x in enumerate (rets)]
        if rumat is not None: rets = [rumat @ x if idx == 2 else x for idx, x in enumerate (rets)]
        return rets

    # Zero matrix escape
    working_matrix = the_matrix if isinstance (the_matrix, np.ndarray) else np.eye (Mbasis)
    if lspace_isvectorblock: working_matrix = lspace.conjugate ().T @ working_matrix
    if lspace is not None and not lspace_isvectorblock: working_matrix = working_matrix[lspace,:]
    if rspace_isvectorblock: working_matrix = working_matrix @ rspace
    if rspace is not None and not rspace_isvectorblock: working_matrix = working_matrix[:,rspace]
    zero_matrix = (0 in (M,N)) or (np.amax (np.abs (working_matrix)) < num_zero_atol)
    zero_matrix = zero_matrix or ((scipy.linalg.norm (working_matrix) / working_matrix.size) < num_zero_atol)
    if zero_matrix:
        K = 0 if only_nonzero_vals else K
        llabels = lsymm
        lvecs = np.eye (Mbasis) 
        if lspace is not None:
            lvecs = lspace if lspace_isvectorblock else lvecs[:,lspace]
            if llabels is not None and not lspace_isvectorblock: llabels=llabels[lspace]
        if lspace_symmetry is not None: llabels = lspace_symmetry
        rlabels = rsymm
        rvecs = np.eye (Nbasis) 
        if rspace is not None:
            rvecs = rspace if rspace_isvectorblock else rvecs[:,rspace]
            if rlabels is not None and not rspace_isvectorblock: rlabels=rlabels[rspace]
        if rspace_symmetry is not None: rlabels = rspace_symmetry
        if not full_matrices:
            lvecs = lvecs[:,:K]
            rvecs = rvecs[:,:K]
            if return_llabels: llabels = llabels[:K]
            if return_rlabels: rlabels = rlabels[:K]
        if return_llabels and return_rlabels:
            return lvecs, np.zeros ((K)), rvecs, llabels, rlabels
        elif return_llabels:
            return lvecs, np.zeros ((K)), rvecs, llabels
        elif return_rlabels:
            return lvecs, np.zeros ((K)), rvecs, rlabels
        else:
            return lvecs, np.zeros ((K)), rvecs

    # Recurse from strong symmetry enforcement to SVD over individual symmetry blocks
    if strong_symm:


        # If a subspace is being diagonalized, recurse into symmetry blocks via the subspaces
        lspace, lsymm_lbls = _unpack_space_symm (lspace, lsymm, lspace_symmetry, lspace_isvectorblock, atol=num_zero_atol, rtol=num_zero_rtol)
        rspace, rsymm_lbls = _unpack_space_symm (rspace, rsymm, rspace_symmetry, rspace_isvectorblock, atol=num_zero_atol, rtol=num_zero_rtol)
            
        uniq_common_lbls = np.unique (np.append (lsymm_lbls, rsymm_lbls))
        uniq_null_lbls = [np.setdiff1d (np.unique (x), uniq_common_lbls) for x in (lsymm_lbls, rsymm_lbls)]
        svals = []
        vecs = [[], []]
        vecs_null = [[], []]
        labels = [[], []]
        labels_null = [[], []]
        vecs_blk = [[], []]
        for lbl in uniq_common_lbls:
            lsubs_blk = lspace[...,lsymm_lbls==lbl]
            rsubs_blk = rspace[...,rsymm_lbls==lbl]
            vecs_blk[0], svals_blk, vecs_blk[1] = matrix_svd_control_options (the_matrix,
                lsymm=None, rsymm=None, symmetry=None, strong_symm=False, lspace_symmetry=None, rspace_symmetry=None,
                lspace=lsubs_blk, rspace=rsubs_blk, only_nonzero_vals=only_nonzero_vals, full_matrices=full_matrices,
                sort_vecs=sort_vecs, num_zero_rtol=num_zero_rtol, num_zero_atol=num_zero_atol)
            nvals = len (svals_blk)
            nnull = [x.shape[1] - nvals for x in vecs_blk]
            svals.append (svals_blk)
            for i in range (2):
                vecs[i].append (vecs_blk[i][:,:nvals])
                vecs_null[i].append (vecs_blk[i][:,nvals:])
                labels[i].extend ([lbl for ix in range (nvals)])
                labels_null[i].extend ([lbl for ix in range (nnull[i])])
        svals = np.concatenate (svals)
        nvals = len (svals)
        vecs = [np.concatenate (x, axis=1) for x in vecs]
        labels = [np.asarray (x) for x in labels]
        vecs_null = [np.concatenate (x, axis=1) for x in vecs_null]
        labels_null = [np.asarray (x).astype (int) for x in labels_null]
        if sort_vecs:
            idx = svals.argsort ()[::sort_vecs]
            svals = svals[idx]
            vecs = [x[:,idx] for x in vecs]
            labels = [x[idx] for x in labels]
        if full_matrices:
            vecs = [np.append (x, y, axis=1) for x, y in zip (vecs, vecs_null)]
        if full_matrices or not only_nonzero_vals:
            labels = [np.append (x, y) for x, y in zip (labels, labels_null)]
            vecs[0], labels[0] = _add_symm_null (vecs[0], labels[0], uniq_null_lbls[0],
                lsymm_lbls, lspace, lspace_isvectorblock, Mbasis)
            vecs[1], labels[1] = _add_symm_null (vecs[1], labels[1], uniq_null_lbls[1],
                rsymm_lbls, rspace, rspace_isvectorblock, Nbasis)
        if not only_nonzero_vals and nvals < K: svals = np.append (svals, np.zeros (K - nvals))
        return vecs[0], svals, vecs[1], labels[0], labels[1]

    # Wrap in subspaces (Have to do both vector-block forms first)
    if lspace_isvectorblock:
        if isinstance (the_matrix, np.ndarray):
            the_matrix = lspace.conjugate ().T @ the_matrix 
        else:
            the_matrix = lspace.conjugate ().T * the_matrix
    if rspace_isvectorblock:
        if isinstance (the_matrix, np.ndarray):
            the_matrix = the_matrix @ rspace
        else:
            the_matrix = the_matrix * rspace
    elif rspace is not None:
        the_matrix = the_matrix[:,rspace]
    if not lspace_isvectorblock and lspace is not None:
        the_matrix = the_matrix[lspace,:]

    # Kernel
    lvecs, svals, r2q = scipy.linalg.svd (the_matrix, full_matrices=full_matrices)
    rvecs = r2q.conjugate ().T
    nsvals = len (svals)
    if only_nonzero_vals:
        idx = np.isclose (svals, 0, atol=num_zero_atol, rtol=0)
        svals = svals[~idx]
        if full_matrices:
            lvecs[:,:nsvals] = np.append (lvecs[:,:nsvals][:,~idx], lvecs[:,:nsvals][:,idx], axis=1)
            rvecs[:,:nsvals] = np.append (rvecs[:,:nsvals][:,~idx], rvecs[:,:nsvals][:,idx], axis=1)
        else:
            lvecs = lvecs[:,~idx]
            rvecs = rvecs[:,~idx]
        nsvals = len (svals)
    if sort_vecs:
        idx = (np.abs (svals)).argsort ()[::sort_vecs]
        svals = svals[idx]
        rvecs[:,:nsvals] = rvecs[:,:nsvals][:,idx]
        lvecs[:,:nsvals] = lvecs[:,:nsvals][:,idx]

    # Align and label eigenvectors using SUBSPACE symmetry information
    lvecs, rvecs, llabels, rlabels = align_degenerate_coupled_vecs (lvecs, svals, rvecs, lspace_symmetry, rspace_symmetry, rtol=num_zero_rtol, atol=num_zero_atol)

    # Wrap out subspaces
    if lspace_isvectorblock:
        lvecs = lspace @ lvecs
    elif lspace is not None:
        subs_lvecs = lvecs.copy ()
        lvecs = np.zeros ((Mbasis, lvecs.shape[1]), dtype=lvecs.dtype)
        lvecs[lspace,:] = subs_lvecs
    if rspace_isvectorblock:
        rvecs = rspace @ rvecs
    elif rspace is not None:
        subs_rvecs = rvecs.copy ()
        rvecs = np.zeros ((Nbasis, rvecs.shape[1]), dtype=rvecs.dtype)
        rvecs[rspace,:] = subs_rvecs

    # Align and label eigenvectors using FULL SPACE symmetry information
    lvecs, rvecs, llabels, rlabels = align_degenerate_coupled_vecs (lvecs, svals, rvecs, lsymm, rsymm, rtol=num_zero_rtol, atol=num_zero_atol)

    if return_llabels and return_rlabels:
        return lvecs, svals, rvecs, llabels, rlabels
    elif return_llabels:
        return lvecs, svals, rvecs, llabels
    elif return_rlabels:
        return lvecs, svals, rvecs, rlabels

    return lvecs, svals, rvecs

def lazyassign_vecs (vecs, row_labels, rtol=num_zero_rtol, atol=num_zero_atol, assert_tol=0, assign_thresh=1.0, return_weights=False):
    uniq_labels = np.unique (row_labels)
    wgts = np.stack ([(vecs[lbl==row_labels,:].conjugate () * vecs[lbl==row_labels,:]).sum (0) for lbl in uniq_labels], axis=-1)
    col_labels = uniq_labels[np.argmax (wgts, axis=-1)]
    wgts = np.amax (wgts, axis=-1)
    idx = np.argsort (wgts)[::-1]
    vecs = vecs[:,idx]
    wgts = wgts[idx]
    col_labels = col_labels[idx]
    nassign = np.count_nonzero ((assign_thresh-wgts) <= atol + rtol*assign_thresh)
    if return_weights: return vecs, col_labels, nassign, wgts
    return vecs, col_labels, nassign

def align_vecs (vecs, row_labels, rtol=num_zero_rtol, atol=num_zero_atol, assert_tol=0):
    uniq_labels = np.unique (row_labels)
    vecs, col_labels, i = lazyassign_vecs (vecs, row_labels, rtol=rtol, atol=atol, assert_tol=assert_tol)
    while i < vecs.shape[1]:
        try:
            svdout = [matrix_svd_control_options (1, lspace=(row_labels==lbl), rspace=vecs[:,i:],
                sort_vecs=-1, only_nonzero_vals=False, full_matrices=True) for lbl in uniq_labels]
        except (ValueError, np.linalg.linalg.LinAlgError) as e:
            print ("Missing zero-matrix escape? number of remaining vectors = {}".format (vecs.shape[1]-i))
            for lbl in uniq_labels:
                mat = vecs[row_labels==lbl,i:]
                try:
                    print ("max, norm of {} = {}, {}".format (lbl, np.amax (np.abs (mat)), scipy.linalg.norm (mat)))
                except:
                    fname = 'DEBUG_{}.npy'.format (lbl)
                    print ("Cannot calculate norm/max of vecs for {} for some reason; saving to {} instead".format (lbl, fname))
                    np.save (fname, mat)
            raise (e)
        # This argmax identifies the single best irrep assignment possible for all of vecs[:,i:]
        symm_label_idx = np.argmax ([svals[0] for lvecs, svals, rvecs in svdout])
        lvecs, svals, rvecs = svdout[symm_label_idx]
        j = i + np.count_nonzero (np.isclose (svals, svals[0], atol=atol, rtol=rtol))
        col_labels[i:j] = uniq_labels[symm_label_idx]
        if assert_tol and not np.all (np.isclose (svals, 0, atol=assert_tol, rtol=rtol) | np.isclose (svals, 1, atol=assert_tol, rtol=rtol)):
            raise RuntimeError ('Vectors not block-adapted in space {}; svals = {}'.format (col_labels[i], svals))
        # This puts the best-aligned vector at position i and causes all of vecs[:,j:] to be orthogonal to it
        vecs[:,i:] = rvecs
        assert (j > i), "{}: {}->{}".format (svals[0], i, j)
        i = j
        vecs[:,i:], col_labels[i:], di = lazyassign_vecs (vecs[:,i:], row_labels, rtol=rtol, atol=atol,
            assert_tol=assert_tol, assign_thresh=svals[0])
        i = i + di
        # This is a trick to grab a whole bunch of degenerate vectors at once (i.e., numerically symmetry-adapted vectors with the same sval)
        # It may improve numerical stability
    return vecs, col_labels

def align_degenerate_coupled_vecs (lvecs, svals, rvecs, lsymm, rsymm, rtol=num_zero_rtol, atol=num_zero_atol, assert_tol=0):
    nvals = len (svals)
    if lsymm is None and rsymm is None:
        return lvecs, rvecs, None, None
    if lsymm is None: lsymm = []
    if rsymm is None: rsymm = []
    uniq_labels = np.unique (np.append (lsymm, rsymm))
    idx_unchk = np.ones (nvals, dtype=np.bool_)
    llabels = np.empty (lvecs.shape[1], dtype=uniq_labels.dtype)
    rlabels = np.empty (rvecs.shape[1], dtype=uniq_labels.dtype)
    lv = lvecs[:,:nvals]
    rv = rvecs[:,:nvals]
    ll = llabels[:nvals]
    while np.count_nonzero (idx_unchk):
        chk_1st_sval = svals[idx_unchk][0]
        idx = np.isclose (svals, chk_1st_sval, rtol=rtol, atol=atol)
        if np.count_nonzero (idx) > 1:
            lv[:,idx], rv[:,idx], ll[idx] = align_coupled_vecs (
                lv[:,idx], rv[:,idx], lsymm, rsymm, rtol=rtol, atol=atol)
        else:
            proj = lv[:,idx] @ rv[:,idx].conjugate ().T
            symmweight = []
            for lbl in uniq_labels:
                row_idx = lsymm==lbl if len (lsymm) > 0 else np.ones (proj.shape[0], dtype=bool)
                col_idx = rsymm==lbl if len (rsymm) > 0 else np.ones (proj.shape[1], dtype=bool)
                symmweight.append (scipy.linalg.norm (proj[np.ix_(row_idx,col_idx)]))
            ll[idx] = uniq_labels[np.argmax (symmweight)]
        idx_unchk[idx] = False
    rlabels[:nvals] = llabels[:nvals]
    if lvecs.shape[1] > nvals and len (lsymm) > 0:
        lvecs[:,nvals:], llabels[nvals:] = lazyassign_vecs (lvecs[:,nvals:], lsymm, rtol=rtol, atol=atol)[:2]
    if rvecs.shape[1] > nvals and len (rsymm) > 0:
        rvecs[:,nvals:], rlabels[nvals:] = lazyassign_vecs (rvecs[:,nvals:], rsymm, rtol=rtol, atol=atol)[:2]
    if len (lsymm) == 0: llabels = None
    if len (rsymm) == 0: rlabels = None
    return lvecs, rvecs, llabels, rlabels

def align_coupled_vecs (lvecs, rvecs, lrow_labels, rrow_labels, rtol=num_zero_rtol, atol=num_zero_atol, assert_tol=0):
    assert (lvecs.shape[1] == rvecs.shape[1])
    npairs = lvecs.shape[1]
    col_labels = np.empty (npairs, dtype=rrow_labels.dtype)
    if lrow_labels is None: lrow_labels = []
    if rrow_labels is None: rrow_labels = []
    uniq_labels = np.unique (np.append (lrow_labels, rrow_labels))
    i = 0
    coupl = lvecs @ rvecs.conjugate ().T
    while i < npairs:
        svdout = []
        for lbl in uniq_labels:
            metric = coupl.copy ()
            if len (lrow_labels) > 0: metric[lrow_labels!=lbl,:] = 0
            if len (rrow_labels) > 0: metric[:,rrow_labels!=lbl] = 0
            svdout.append (matrix_svd_control_options (metric, sort_vecs=-1, only_nonzero_vals=False, full_matrices=True,
                lspace=lvecs[:,i:], rspace=rvecs[:,i:]))
        symm_label_idx = np.argmax ([svals[0] for lu, svals, ru in svdout])
        lvecs[:,i:], svals, rvecs[:,i:] = svdout[symm_label_idx]
        j = i + np.count_nonzero (np.isclose (svals, svals[0], atol=atol, rtol=rtol))
        col_labels[i:j] = uniq_labels[symm_label_idx]
        if assert_tol and not np.all (np.isclose (svals, 0, atol=assert_tol, rtol=rtol) | np.isclose (svals, 1, atol=assert_tol, rtol=rtol)):
            raise RuntimeError ('Vectors not block-adapted in space {}; svals = {}'.format (col_labels[i], svals))
        assert (j > i)
        i = j
    return lvecs, rvecs, col_labels

