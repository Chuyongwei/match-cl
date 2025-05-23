import torch
import cv2
from transformations import quaternion_from_matrix
import numpy as np
# import pydegensac
# import pymagsac

def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t


def eval_nondecompose(p1s, p2s, E_hat, dR, dt, scores):

    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(p1s.dtype)
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E_hat, p1s_good, p2s_good)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            print("Failed in evaluation")
            print(E_hat)
            print(R)
            print(t)
            err_q = np.pi
            err_t = np.pi / 2
            #import pdb;pdb.set_trace()
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    # Change mask type
    mask = mask.flatten().astype(bool)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t


def eval_decompose_F(p1s, p2s, p1s_1, p2s_1, dR, dt, K1, K2, mask=None, method=cv2.LMEDS, probs=None,
                   weighted=False, use_prob=True, idx=None):

    # import wrappers
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    p1s_good_n = p1s_1[mask]
    p2s_good_n = p2s_1[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    R, t = None, None
    # cancel it because GCRANSAC
    # assert method.endswith("_F")

    if p1s_good.shape[0] >= 8:
        
        if method == "MAGSAC_F":
            # using magsac wrapper: 
            # Funda Mat: method="MAGSAC_F";
            #magsac 0.3
            # point = np.stack([p1s_good,p2s_good],axis=1).reshape(-1,1)
            # w1 = float(K1[0, 2] * 2 + 1.0)
            # h1 = float(K1[1, 2] * 2 + 1.0)
            # w2 = float(K2[0, 2] * 2 + 1.0)
            # h2 = float(K2[1, 2] * 2 + 1.0)
            # F,mask_new = pymagsac.findFundamentalMatrix(point,w1,h1,w2,h2,probabilities=None,use_magsac_plus_plus=False)
            #magsac 0.2
            F, mask_new = pymagsac.findFundamentalMatrix(p1s_good, p2s_good, False, 1)
            mask_new = mask_new.reshape(-1, 1).astype(np.uint8)
            E = None
        elif method == "DEGENSAC_F":
            # using pyransac by disabling degeneracy check
            # Better performance than opencv's ransac 
            threshold = 0.5
            F, mask_new = pydegensac.findFundamentalMatrix(p1s_good, p2s_good, threshold)
            mask_new = mask_new.reshape(-1, 1).astype(np.uint8)
            E = None
            # using opencv
            # F, mask_new = cv2.findFundamentalMat(
            #     p1s_good, p2s_good, cv2.FM_RANSAC, 3.0, 0.999)
            # E = None
        elif method == "GCRANSAC_F":
            w1 = int(K1[0, 2] * 2 + 1.0)
            h1 = int(K1[1, 2] * 2 + 1.0)
            w2 = int(K2[0, 2] * 2 + 1.0)
            h2 = int(K2[1, 2] * 2 + 1.0)
            import pygcransac
            F, mask_new = pygcransac.findFundamentalMatrix(
                p1s_good, p2s_good, h1, w1, h2, w2, threshold=0.5)
            mask_new = mask_new.reshape(-1, 1).astype(np.uint8)
            E = None
        else:
            raise ValueError("wrong method!")

        # convert to E if there is a F
        if F is not None:
            if F.shape[0] != 3:
                F = np.split(F, len(F) / 3)[0]
            # get E from F

            E = E.astype(np.float64)
            # mask_f = mask_new.flatten().astype(bool)

            # # # go back calibrated
            # p1s_good = (p1s_good - np.array([K1[0, 2], K1[1, 2]])) / K1[0,0]
            # p2s_good = (p2s_good - np.array([K2[0, 2], K2[1, 2]])) / K2[0,0]

        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good_n, p2s_good_n, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    # print("err_q: {} err_t: {}".format(err_q, err_t))

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t



def eval_decompose(p1s, p2s, dR, dt, mask=None, method=cv2.LMEDS, probs=None,
                   weighted=False, use_prob=True):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    R, t = None, None
    if p1s_good.shape[0] >= 5:
        if probs is None and method != "MLESAC":
            # Change the threshold from 0.01 to 0.001 can largely imporve the results
            # For fundamental matrix estimation evaluation, we also transform the matrix to essential matrix.
            # This gives better results than using findFundamentalMat
            E, mask_new = cv2.findEssentialMat(p1s_good, p2s_good, method=method, threshold=0.001)
            # import pymagsac
            # F, mask = pymagsac.findFundamentalMatrix(p1s_good, p2s_good, 3.0)


        else:
            pass
        if E is not None:
            new_RT = False
            # Get the best E just in case we get multipl E from
            # findEssentialMat
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _mask_new2 = cv2.recoverPose(
                    _E, p1s_good, p2s_good, mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    mask_new2 = _mask_new2
                    new_RT = True
            if new_RT:
                err_q, err_t = evaluate_R_t(dR, dt, R, t)
            else:
                err_q = np.pi
                err_t = np.pi / 2

        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated, R, t
