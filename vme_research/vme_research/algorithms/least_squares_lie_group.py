###############################################################################
#
# Solve least squares problems on SE(3) and SO(3) using the Kabsch-Umeyama algorithm and variants
#
# History:
# 08-22-24 - Levi Burner - Created file, forms of this codes have appeared in random other projects
#
###############################################################################

import numpy as np

# Kabsch-Umeyama algorithm
# S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns,"
# in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 4, pp. 376-380, April 1991
# Estimate transform such that p_a = s * T_ab p_b by minimizing the least squares error via SVD
def least_squares_scale_SE3(p_a, p_b, assert_unique=True):
    E_p_a = p_a.mean(axis=0)
    E_p_b = p_b.mean(axis=0)

    p_a_centered = p_a - E_p_a
    p_b_centered = p_b - E_p_b

    Sigma = (p_a_centered.T @ p_b_centered) / p_b.shape[0]

    if assert_unique:
        assert np.linalg.matrix_rank(Sigma) >= 2

    U, d, V_T = np.linalg.svd(Sigma)

    # Got a left handed transform, fix it
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        V_T[2, :] = -V_T[2, :]

    R_ab = np.dot(U, V_T)

    V_b = np.var(p_b, axis=0)
    c_ab = d.sum() / V_b.sum()

    t_ab = E_p_a - c_ab * (R_ab @ E_p_b.T).T

    return c_ab, R_ab, t_ab

# Estimate transform such that p_a = R_ab p_b by minimizing the least squares error via SVD
# Based on the lemma in Umeyama's paper and specifically equation 31
# It is equivalent to above with a few less features
# It can be used to align rotation matrices by using
# the columns of the rotation matrices as points
# This is Wahba's problem
def least_squares_SO3(p_a, p_b):
    Sigma = (p_a.T @ p_b) / p_b.shape[0]

    U, d, V_T = np.linalg.svd(Sigma)

    # Got a left handed transform, fix it
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        V_T[2, :] = -V_T[2, :]

    R_ab = np.dot(U, V_T)
    return R_ab
