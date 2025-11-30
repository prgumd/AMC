import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import shutil
import subprocess
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.svm import LinearSVC
from tqdm import tqdm

from vme_research.hardware.record import Load, Record
from vme_research.algorithms.affine_flow import (draw_warped_patch_location,
                                                 draw_full_reverse_warp,
                                                 make_rot_times_affine,
                                                 AffineTrackRotInvariant)
from vme_research.algorithms.least_squares_lie_group import least_squares_SO3

from jax import config
config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
import logging
logger = logging.getLogger("jax._src.xla_bridge")
logger.setLevel(logging.ERROR)
from vme_research.algorithms.april_pose import AprilPose, get_pos_with_april_tag
from vme_research.algorithms.patch_track import (JHom4pTrackRotInvariant,
                                                 JRotTrackRotInvariant,
                                                 hom_4p_I_W_p_all_jit,
                                                 H_ji_4_point_dlt_jit)

from multiprocessing import Value
from vme_research.visualization.point_cloud import PointCloud
from vme_research.messaging.shared_ndarray import SharedNDArrayPipe



StabilizeFieldsOptions = {
    'name': 'Stabilize',
    'version': '0.0.1',
    'fields': [{'name': 'R_cprime_c0_t0', 'type': str(np.ndarray), 'split': False},
               {'name': 'R_cprimeprime_c0_t0', 'type': str(np.ndarray), 'split': False}],
    'append_fields': [],
}

SignalFieldsOptions = {
    'name': 'Signal',
    'version': '0.0.1',
    'fields': [{'name': 'x', 'type': str(np.ndarray), 'split': False}],
    'append_fields': []
}


# @jit(nopython=True)
def avg_frames(frames): # faster than np.mean
    frame_avg = np.zeros_like(frames[0])
    for i in range(len(frames)):
        frame_avg += frames[i]
    return frame_avg / len(frames)

def and_frames(frames):
    frame_and = np.ones_like(frames[0], dtype=bool)
    for i in range(len(frames)):
        frame_and = frame_and & frames[i]
    return frame_and

def basic_normal_flow(frame0, frame1, grad_frame=None):
    # Compute the normal flow between two frames
    # frame0 and frame1 are assumed to be grayscale
    if grad_frame is None: grad_frame = frame0
    grad_x = cv2.Sobel(grad_frame, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    grad_y = cv2.Sobel(grad_frame, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    norm_grad = np.sqrt(grad_x**2 + grad_y**2)
    dI = frame1 - frame0

    # Write this line but handle division by zero properly
    # nf = -(dI / norm_grad)[:, :, np.newaxis] * (np.dstack((grad_x, grad_y)) / norm_grad[:, :, np.newaxis])
    nf = np.zeros((*frame0.shape, 2), dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        nf[:, :, 0] = -(dI / norm_grad) * (grad_x / norm_grad)
        nf[:, :, 1] = -(dI / norm_grad) * (grad_y / norm_grad)

    # Zero out small gradients
    # TODO parameter
    nf[norm_grad < 15.0/255, :] = 0 # TODO, its not clear if this should be a threshold or a weighting
    nf[np.abs(dI) < 0.0/255, :] = 0
    return nf

def draw_flow_arrows(img, xx, yy, dx, dy, p_skip=15, mag_scale=1.0):
    xx     = xx[::p_skip, ::p_skip].flatten()
    yy     = yy[::p_skip, ::p_skip].flatten()
    flow_x = dx[::p_skip, ::p_skip].flatten()
    flow_y = dy[::p_skip, ::p_skip].flatten()

    for x, y, u, v in zip(xx, yy, flow_x, flow_y):
        if np.isnan(u) or np.isnan(v):
            continue
        cv2.arrowedLine(img,
                        (int(x), int(y)),
                        (int(x+mag_scale*u), int(y+mag_scale*v)),
                        (0, 0, 0),
                        tipLength=0.2)

def draw_nonzero_flow_arrows(img, xx, yy, dx, dy, p_skip=15, mag_scale=1.0, color=(0, 0, 0)):
    zero = (np.abs(dx) < 1e-5) & (np.abs(dy) < 1e-5)

    xx = xx[~zero].flatten()[::p_skip]
    yy = yy[~zero].flatten()[::p_skip]
    dx = dx[~zero].flatten()[::p_skip]
    dy = dy[~zero].flatten()[::p_skip]

    # xx     = xx[::p_skip, ::p_skip].flatten()
    # yy     = yy[::p_skip, ::p_skip].flatten()
    # flow_x = dx[::p_skip, ::p_skip].flatten()
    # flow_y = dy[::p_skip, ::p_skip].flatten()

    for x, y, u, v in zip(xx, yy, dx, dy):
        if np.isnan(u) or np.isnan(v):
            continue
        cv2.arrowedLine(img,
                        (int(x), int(y)),
                        (int(x+mag_scale*u), int(y+mag_scale*v)),
                        color,
                        tipLength=0.2)

def visualize_optical_flow(flowin, max_flow=None):
    flow=np.ma.array(flowin, mask=np.isnan(flowin))
    theta = np.mod(np.arctan2(flow[:, :, 0], flow[:, :, 1]) + 2*np.pi, 2*np.pi)

    flow_norms = np.linalg.norm(flow, axis=2)
    if max_flow is None:
        max_flow = np.max(np.ma.array(flow_norms, mask=np.isnan(flow_norms)))
    flow_norms_normalized = flow_norms / max_flow
    flow_norms_normalized = np.clip(flow_norms_normalized, 0, 1.0)
    flow_norms_normalized[np.isnan(flow_norms_normalized)] = 0

    flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * flow_norms_normalized
    flow_hsv[:, :, 2] = 255 * (flow_norms_normalized > 0)

    flow_hsv[np.logical_and(np.isnan(theta), np.isnan(flow_norms_normalized)), :] = 0

    return flow_hsv

# https://github.com/dhyuan99/VecKM_flow/blob/97a6cb9a650e612913a2f8cc7a89ff54482f94be/egomotion/s1_main.py#L31
def form_positivity_matrices_with_w(precomputed_A_x, precomputed_B_x, xy, d_normalized, w):
    A_x = precomputed_A_x[xy[:,1].astype(int), xy[:,0].astype(int)]
    B_x = precomputed_B_x[xy[:,1].astype(int), xy[:,0].astype(int)]

    dx_normalized = d_normalized[:, 0]
    dy_normalized = d_normalized[:, 1]

    g_x = np.stack((dx_normalized, dy_normalized), axis=-1)
    g_x = g_x / np.linalg.norm(g_x, axis=-1, keepdims=True)
    dt = 1.0
    n_x = g_x[:, 0] * (dx_normalized / dt) + g_x[:, 1] * (dy_normalized / dt)

    g_x_A_x = (g_x[:,None,:] @ A_x)[:,0,:]
    n_x_g_x_B_x_w = n_x.squeeze() - np.einsum('ij,ij->i', g_x, B_x @ w)

    return g_x_A_x, n_x_g_x_B_x_w

# https://github.com/dhyuan99/VecKM_flow/blob/97a6cb9a650e612913a2f8cc7a89ff54482f94be/egomotion/s1_main.py#L47C1-L67C21
# v0 is unused because the problem is convex
def svm_positivity(g_x_A_x, n_x_g_x_B_x_w, v0=None):
    #############################################################
    ##### Egomotion estimation from g_x, n_x, A_x, B_x ##########
    ############################################################
    sample_weights = np.abs(n_x_g_x_B_x_w)
    sign_w = np.sign(n_x_g_x_B_x_w)
    # sample_weights = np.abs(sign_w)

    X_balanced = np.concatenate([g_x_A_x, -g_x_A_x], axis=0)
    Y_balanced = np.concatenate([sign_w, -sign_w], axis=0)
    S_balanced = np.concatenate([sample_weights, sample_weights], axis=0)
    res = LinearSVC(fit_intercept=False, C=1).fit(
        X_balanced,
        Y_balanced, 
        sample_weight=S_balanced)
    sign_v = res.decision_function(g_x_A_x)
    v_c1_pred = res.coef_.squeeze()
    v_c1_pred = v_c1_pred / np.linalg.norm(v_c1_pred)
    # print(f"pct: {np.mean(np.sign(sign_v) == np.sign(sign_w)):.3f}", v_c1_pred, v_c1)
    return v_c1_pred

class EventNormalFlow:
    def __init__(self):
        self.frame0 = None
        self.framet = None
        self.grad_x = None
        self.grad_y = None
        self.norm_grad = None

        self.dx = 1
        self.dy = 1
        self.ksize = 3
        self.scale = 0.125

    def update(self, frame, saccade=False):
        # frame is assumed to be grayscale
        edge_thresh = 10.0 / 255.0
        dI_thresh = 2.1 / 255.0
        max_dt_frames = 30.0

        if self.frame0 is None or saccade:
            self.frame0 = frame
            self.framet = np.zeros_like(self.frame0)
            self.grad_x = cv2.Sobel(frame, cv2.CV_32F, self.dx, 0, ksize=self.ksize, scale=self.scale)
            self.grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, self.dy, ksize=self.ksize, scale=self.scale)
            self.norm_grad = np.sqrt(self.grad_x**2 + self.grad_y**2)
        else:
            self.framet += 1.0

        dI = frame - self.frame0
        sig_dI = np.abs(dI) > dI_thresh
        sig_grad = self.norm_grad >= edge_thresh
        sig_flow = sig_dI & sig_grad

        nf = np.zeros((*frame.shape, 2), dtype=np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            nf[:, :, 0] = -(dI / self.norm_grad) * (self.grad_x / self.norm_grad)
            nf[:, :, 1] = -(dI / self.norm_grad) * (self.grad_y / self.norm_grad)
            nf = nf / self.framet[:, :, None]

        # Use all the nf with a significant gradient, even if the flow is close to zero
        nf[~sig_grad] = 0
    
        # framets = self.framet[sig_flow]
        # if framets.shape[0] != 0:
        #     print('framet', np.min(framets), np.max(framets), np.mean(framets), np.median(framets))

        # Update the state
        new_grad_x = cv2.Sobel(frame, cv2.CV_32F, self.dx, 0, ksize=self.ksize, scale=self.scale)
        new_grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, self.dy, ksize=self.ksize, scale=self.scale)
        norm_new_grad = np.sqrt(new_grad_x**2 + new_grad_y**2)
        new_sig_grad = norm_new_grad >= edge_thresh

        # Update pixels where there was not a gradient but there is now
        # or where flow was estimated
        # TODO where data is old?
        timeout = self.framet >= max_dt_frames
        update = (~sig_grad & new_sig_grad) | sig_flow | timeout

        self.frame0[update] = frame[update]
        self.framet[update] = 0
        self.grad_x[update] = new_grad_x[update]
        self.grad_y[update] = new_grad_y[update]
        self.norm_grad[update] = norm_new_grad[update]

        # cv2.imshow('frame0', self.frame0)
        return nf

class SVMEgomotion:
    def __init__(self, K, dist, res):
        self.K = K
        self.dist = dist
        self.res = res

        # Get x,y coordinates of nf
        self.xx, self.yy = np.meshgrid(np.arange(self.res[0]), np.arange(self.res[1]))

        # Create the A_x and B_x matrices
        map1, map2 = cv2.initInverseRectificationMap(
            K, # Intrinsics
            dist, # Distortion
            np.eye(3), # Rectification
            np.eye(3), # New intrinsics
            res, # Size of the image
            cv2.CV_32FC1
        )
        self.map1 = map1
        self.map2 = map2

        self.precomputed_A_x = np.zeros((*map1.shape, 2, 3))
        self.precomputed_A_x[:, :, 0, 0] = -1
        self.precomputed_A_x[:, :, 1, 1] = -1
        self.precomputed_A_x[:, :, 0, 2] = map1
        self.precomputed_A_x[:, :, 1, 2] = map2

        self.precomputed_B_x = np.zeros((*map1.shape, 2, 3))
        self.precomputed_B_x[:, :, 0, 0] = map1 * map2
        self.precomputed_B_x[:, :, 0, 1] = -(np.square(map1) + 1)
        self.precomputed_B_x[:, :, 0, 2] = map2
        self.precomputed_B_x[:, :, 1, 0] = (np.square(map2) + 1)
        self.precomputed_B_x[:, :, 1, 1] = -map1 * map2
        self.precomputed_B_x[:, :, 1, 2] = -map1

    def estimate(self, nf_list, est_ttc=False):
        if type(nf_list) is not list:
            nf_list = [nf_list]

        nf_flat_list = []
        xx_flat_list = []
        yy_flat_list = []
        for nf in nf_list:
            # Remove flows with small magnitude
            nf_mag = np.linalg.norm(nf, axis=2)
            nf_sig = nf_mag > 0.0 # TODO should this be higher?

            xx = self.xx[nf_sig].flatten()
            yy = self.yy[nf_sig].flatten()
            nf = nf[nf_sig, :].reshape((-1, 2))

            nf_flat_list.append(nf)
            xx_flat_list.append(xx)
            yy_flat_list.append(yy)

        nf_flat = np.concatenate(nf_flat_list)
        xx_flat = np.concatenate(xx_flat_list)
        yy_flat = np.concatenate(yy_flat_list)

        if nf_flat.shape[0] < 10:
            return None, None, None

        # TODO, does the SVM problem involve some sense of weighting?
        # nf = nf / np.linalg.norm(nf, axis=1, keepdims=True)

        g_x_A_x, n_x_g_x_B_x_w = form_positivity_matrices_with_w(
            self.precomputed_A_x, self.precomputed_B_x, 
            np.stack((xx_flat, yy_flat), axis=1), nf_flat, 
            np.array([0.0, 0.0, 0.0]),
        )

        v_c1_pred = svm_positivity(g_x_A_x, n_x_g_x_B_x_w)

        normV_over_Z = None
        normV_over_Z_cloud = None
        if est_ttc:
            # Naively compute ||V||/Z_x using only the latest nf frame
            nf_flat_dec = nf_flat_list[-1]
            xx_flat_dec = xx_flat_list[-1]
            yy_flat_dec = yy_flat_list[-1]

            g_x_A_x_dec, n_x_g_x_B_x_w_dec = form_positivity_matrices_with_w(
                self.precomputed_A_x, self.precomputed_B_x, 
                np.stack((xx_flat_dec, yy_flat_dec), axis=1), nf_flat_dec, 
                np.array([0.0, 0.0, 0.0]),
            )

            if v_c1_pred is not None:
                nf_mag_flat = np.linalg.norm(nf_flat_dec, axis=1)
                normV_over_Z_flat = nf_mag_flat / (g_x_A_x_dec @ v_c1_pred)
                normV_over_Z = np.zeros((self.res[::-1]))
                normV_over_Z[yy_flat_dec.astype(int), xx_flat_dec.astype(int)] = normV_over_Z_flat

                map1_dec = self.map1[yy_flat_dec.astype(int), xx_flat_dec.astype(int)]
                map2_dec = self.map2[yy_flat_dec.astype(int), xx_flat_dec.astype(int)]

                # x_cloud = map1_dec / normV_over_Z_flat
                # y_cloud = map2_dec / normV_over_Z_flat
                # z_cloud = 1 / normV_over_Z_flat

                x_cloud = xx_flat_dec
                y_cloud = yy_flat_dec
                z_cloud = 1 / normV_over_Z_flat

                good = (z_cloud > 0) & (normV_over_Z_flat > 0.1)
                x_cloud = x_cloud[good]
                y_cloud = y_cloud[good]
                z_cloud = z_cloud[good]
                normV_over_Z_cloud = np.hstack((z_cloud[:, None], x_cloud[:, None], y_cloud[:, None]))

        return v_c1_pred, normV_over_Z, normV_over_Z_cloud

def warp_back_frames(R_cprime_c0_t0_list, R_cprimeprime_c0_t0,
                     K, K_inv,
                     frame_color_list=None, frame_gray_list=None,
                     valid=False, crop_rect=None):
    # Warp the frames back to the stabilized orientation
    # TODO when the rotation filter is off, old frames
    # do not need to be re-warped until the stabilized orientation changes
    warped_frames_color = []
    warped_frames_gray = []
    warped_frames_valid = []

    if frame_color_list is not None:
        frame_res = frame_color_list[-1]
    elif frame_gray_list is not None:
        frame_res = frame_gray_list[-1]

    frame_res = (frame_res.shape[1], frame_res.shape[0])
    frame_valid = np.ones(frame_res[::-1], dtype=np.float32)
    for i in range(len(R_cprime_c0_t0_list)):
        R_cprime_c0_t0 = R_cprime_c0_t0_list[i]
        R_cprime_cprimeprime = R_cprime_c0_t0 @ R_cprimeprime_c0_t0.T

        H_cprimenorm_cprimeprime = R_cprime_cprimeprime @ K_inv
        H_cprime_cprimeprime = np.vstack((K[0:2, :] @ H_cprimenorm_cprimeprime,
                                          H_cprimenorm_cprimeprime[2, :]))

        if frame_color_list is not None:
            warped_frame_color = cv2.warpPerspective(frame_color_list[i], H_cprime_cprimeprime, frame_res,
                                                     flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            
            # TODO don't warp pixels that are not used
            if crop_rect is not None:
                warped_frame_color[:crop_rect[1], :, :] = 0
                warped_frame_color[crop_rect[3]:, :, :] = 0
                warped_frame_color[:, :crop_rect[0], :] = 0
                warped_frame_color[:, crop_rect[2]:, :] = 0
            warped_frames_color.append(warped_frame_color)

        if frame_gray_list is not None:
            warped_frame_gray = cv2.warpPerspective(frame_gray_list[i], H_cprime_cprimeprime, frame_res,
                                                    flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
            # TODO don't warp pixels that are not used
            if crop_rect is not None:
                warped_frame_gray[:crop_rect[1], :] = 0
                warped_frame_gray[crop_rect[3]:, :] = 0
                warped_frame_gray[:, :crop_rect[0]] = 0
                warped_frame_gray[:, crop_rect[2]:] = 0
            warped_frames_gray.append(warped_frame_gray)

        if valid:
            warped_frame_valid = cv2.warpPerspective(frame_valid, H_cprime_cprimeprime, frame_res,
                                                     flags=cv2.INTER_NEAREST+cv2.WARP_INVERSE_MAP)
            # TODO don't warp pixels that are not used
            if crop_rect is not None:
                warped_frame_valid[:crop_rect[1], :] = 0
                warped_frame_valid[crop_rect[3]:, :] = 0
                warped_frame_valid[:, :crop_rect[0]] = 0
                warped_frame_valid[:, crop_rect[2]:] = 0
            warped_frame_valid = warped_frame_valid.astype(bool)
            warped_frames_valid.append(warped_frame_valid)

    # Average the warped frames and return
    avg_warped_color = None
    if frame_color_list is not None:
        avg_warped_color = avg_frames(warped_frames_color)

    avg_warped_gray = None
    if frame_gray_list is not None:
        avg_warped_gray = avg_frames(warped_frames_gray)

    and_warped_valid = None
    if valid:
        and_warped_valid = and_frames(warped_frames_valid)

    return avg_warped_color, avg_warped_gray, and_warped_valid

def update_rotation_filter(R_cprime_c0_t0, R_cprimeprime_c0_t0, tau, beta, dt):
    R_cprime_cprimeprime = R_cprime_c0_t0 @ R_cprimeprime_c0_t0.T
    w_cprime_cprimeprime = R.from_matrix(R_cprime_cprimeprime).as_rotvec()

    alpha = tau + beta * np.linalg.norm(w_cprime_cprimeprime)
    dw_cprime_cprimeprime = dt * alpha * w_cprime_cprimeprime
    dR_cprime_cprimeprime = R.from_rotvec(dw_cprime_cprimeprime).as_matrix()

    new_R_cprimeprime_c0_t0 = dR_cprime_cprimeprime @ R_cprimeprime_c0_t0

    return new_R_cprimeprime_c0_t0

class RotStabOmega:
    def __init__(self, template_frame, K, R_c_fc=np.eye(3), R_t0_tk=np.eye(3), R_cprimeprime_c0_t0=None,
                 margins=0.125, N_frames=6,
                 stride=1, max_steps=1000, delta_omega_stop=0.1 * np.pi / 180.0, # 0.1 degrees
                 blur_new_frame=True,
                 recorder=None,
                ):
        self.template_frame = template_frame
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.R_c_fc = R_c_fc.astype(np.float32) # Rotation of template to virtual camera frame (for precompensating with IMU)
        self.margins = margins
        self.N_frames = N_frames
        self.rect = None
        self.stride = stride
        self.max_steps = max_steps
        self.delta_omega_stop = delta_omega_stop
        self.blur_new_frame = blur_new_frame
        self.recorder = recorder

        # Portion of image to stabilize (centered rectangle)
        self.rect = [int(template_frame.shape[1] * margins),
                     int(template_frame.shape[0] * margins),
                     template_frame.shape[1] - int(template_frame.shape[1] * margins),
                     template_frame.shape[0] - int(template_frame.shape[0] * margins)]

        self.R_t0_tk = R_t0_tk # Rotation from the template to the beginning of time
        self.R_cprimeprime_c0_t0 = R_cprimeprime_c0_t0 # Stabilized rotation from current frame to beginning of time
        self.R_cprime_c0_t0_list = [] # List of rotations from current frame to template
        self.frame_color_list = [] # Frames corresponding to rotation list
        self.frame_gray_list = [] # Frames corresponding to rotation list

        self.avg_warped_color_list = [] # List of average warped color frames
        self.avg_warped_gray_list = [] # List of average warped gray frames
        self.and_warped_valid_list = [] # List of average warped valid frames
        self.avg_warped_color_sum = np.zeros((template_frame.shape[0], template_frame.shape[1], 3), dtype=np.float32)
        self.avg_warped_gray_sum = np.zeros_like(template_frame, dtype=np.float32)

        # Create a tracker with the template
        try:
            self.tracker = JRotTrackRotInvariant(
                            rect=self.rect,
                            template_image=template_frame,
                            R_c_fc=self.R_c_fc,
                            K=self.K,
                            delta_p_stop=self.delta_omega_stop,
                            stride=self.stride,
                            max_steps=self.max_steps,
                            blur_new_frame=self.blur_new_frame,
                        )
        except np.linalg.LinAlgError:
            print('RotStabOmega could not create tracker')
            self.tracker = None

    # Update the template to be the last frame passed to update
    def update_template(self):
        self.R_t0_tk = self.R_cprime_c0_t0_list[-1].T
        template_frame = self.frame_gray_list[-1]
        # cv2.imshow('RotStabOmega template', template_frame)

        # Create a tracker with the new template
        try:
            self.tracker = JRotTrackRotInvariant(
                            rect=self.rect,
                            template_image=template_frame,
                            R_c_fc=self.R_c_fc,
                            K=self.K,
                            delta_p_stop=self.delta_omega_stop,
                            stride=self.stride,
                            max_steps=self.max_steps,
                            blur_new_frame=self.blur_new_frame,
                        )
        except np.linalg.LinAlgError:
            print('RotStabOmega could not create tracker')
            self.tracker = None

    def update(self, frame_color, frame_gray, R_c_fc=None, R_cprime_c0_t0=None,
               rot_filt=True, rot_filt_tau=1 / 0.5, rot_filt_beta=40.0, rot_filt_dt=1/60.0,
               saccade=False,
               color=True, gray=False, valid=False,
               frame_time=None, frame_time_received=None):
        assert frame_color.dtype == np.float32
        assert frame_gray.dtype == np.float32
        if R_c_fc is not None:
            R_c_fc = R_c_fc.astype(np.float32)
        else:
            R_c_fc = np.eye(3, dtype=np.float32)

        # If a rotation is not provided, use the tracker to compute it
        if R_cprime_c0_t0 is None:
            w_cprime_c0 = np.array(self.tracker.update(frame_gray, R_c_fc).block_until_ready())
            # self.w_cprime_c0 = w_cprime_c0

            # R_cprime_c0 = H_tk_tk+delta
            # H_t0_tk+delta = H_t0_tk @ H_tk_tk+delta
            # R_cprime_c0_t0 = R_t0_tk @ R_cprime_c0
            R_cprime_c0 = R.from_rotvec(w_cprime_c0).as_matrix()
            R_cprime_c0_t0 = R_cprime_c0 @ self.R_t0_tk.T

        # Update and trim lists
        self.frame_color_list.append(frame_color)
        self.frame_gray_list.append(frame_gray)
        self.R_cprime_c0_t0_list.append(R_cprime_c0_t0)
        raw_frames_to_keep = self.N_frames if rot_filt else 1
        for l in [self.R_cprime_c0_t0_list, self.frame_color_list, self.frame_gray_list]:
            if len(l) > raw_frames_to_keep:
                l.pop(0)
                assert len(l) == raw_frames_to_keep

        if self.R_cprimeprime_c0_t0 is None:
            self.R_cprimeprime_c0_t0 = np.copy(R_cprime_c0_t0)
        if rot_filt:
            # Move R_cprimeprime_c0_t0 continuously towards the current orientation  
            # with quadratic control commanding first order dynamics on SO3
            self.R_cprimeprime_c0_t0 = update_rotation_filter(
                R_cprime_c0_t0, self.R_cprimeprime_c0_t0,
                tau=rot_filt_tau, beta=rot_filt_beta, dt=rot_filt_dt
            )

        # Snap stabilizing orientation to the current orientation
        # TODO generalize this beyond a saccade to the central vision
        if saccade:
            self.R_cprimeprime_c0_t0 = np.copy(R_cprime_c0_t0)
            if not rot_filt:
                self.avg_warped_color_list = []
                self.avg_warped_gray_list = []
                self.and_warped_valid_list = []
                self.avg_warped_color_sum = np.zeros((self.template_frame.shape[0], self.template_frame.shape[1], 3), dtype=np.float32)
                self.avg_warped_gray_sum = np.zeros_like(self.template_frame, dtype=np.float32)

        if self.recorder is not None:
            self.recorder.pub(frame_time,
                              (np.copy(R_cprime_c0_t0), np.copy(self.R_cprimeprime_c0_t0)),
                              t_received=frame_time_received)

        (avg_warped_color,
         avg_warped_gray,
         and_warped_valid) = warp_back_frames(
            self.R_cprime_c0_t0_list, self.R_cprimeprime_c0_t0,
            self.K, self.K_inv,
            frame_color_list=self.frame_color_list if color else None,
            frame_gray_list=self.frame_gray_list if gray else None,
            valid=valid, crop_rect=self.rect)

        if not rot_filt:
            if avg_warped_color is not None:
                self.avg_warped_color_list.append(avg_warped_color)
                self.avg_warped_color_sum += avg_warped_color
                if len(self.avg_warped_color_list) > self.N_frames:
                    self.avg_warped_color_sum -= self.avg_warped_color_list[0]
                    self.avg_warped_color_list.pop(0)
                avg_warped_color = self.avg_warped_color_sum / len(self.avg_warped_color_list)

            if avg_warped_gray is not None:
                self.avg_warped_gray_list.append(avg_warped_gray)
                self.avg_warped_gray_sum += avg_warped_gray
                if len(self.avg_warped_gray_list) > self.N_frames:
                    self.avg_warped_gray_sum -= self.avg_warped_gray_list[0]
                    self.avg_warped_gray_list.pop(0)
                avg_warped_gray = self.avg_warped_gray_sum / len(self.avg_warped_gray_list)

            if and_warped_valid is not None:
                self.and_warped_valid_list.append(and_warped_valid)
                if len(self.and_warped_valid_list) > self.N_frames:
                    self.and_warped_valid_list.pop(0)
                and_warped_valid = and_frames(self.and_warped_valid_list)

        return avg_warped_color, avg_warped_gray, and_warped_valid

# def normalize(x, min_x, max_x):
#     # min_x = np.min(x)
#     # max_x = np.max(x)
#     y = (x - min_x) / (max_x - min_x)
#     y = np.clip(y, 0, 1)
#     return y

def ttc_to_color(ttc, min_ttc=0.0, max_ttc=10.0):
    # print('ttc', ttc.shape, np.min(ttc), np.max(ttc))
    no_data = np.isnan(ttc) | np.isinf(ttc)
    # Normalize ttc to [0, 1]
    ttc = (ttc - min_ttc) / (max_ttc - min_ttc)
    ttc = np.clip(ttc, 0, 1)

    # Create a color map from ttc using matplotlib color maps
    # cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('turbo')
    color_map = cmap(1-ttc)
    color_map[no_data, :] = 0.0
    r = (color_map[:, :, 0] * 255).astype(np.uint8)
    g = (color_map[:, :, 1] * 255).astype(np.uint8)
    b = (color_map[:, :, 2] * 255).astype(np.uint8)
    # Basic red to green color map
    # ttc = 1-ttc
    # r = (ttc * 255).astype(np.uint8)
    # g = (255 - ttc * 255).astype(np.uint8)
    # b = np.zeros_like(r, dtype=np.uint8)
    # r[no_data] = 0
    # g[no_data] = 0
    # b[no_data] = 0
    return np.dstack((b, g, r))

def on_val_return_trackerbar(val, val0, val_ret):
    val_ret[0] = (val - val0) / 1000.0

def stabilize(sequence, cam_name,
              model, K_in, cam_dist, rotate, downsample,
              affine, rotstab,
              rotstab2, rotstab_gyro, 
              saccade,
              rotstab_video,
              export_cropped,
              export_stab_only,
              undistort_video,
              frame_filt,
              est_ttc, est_nf, est_foe, est_april,
              compute_metrics,
              benchmark,
              margins=0.125,
              tracker_stride=1,
              N_frames=6,
              start_t=None, end_t=None):
    num_stabilizers_enabled = np.sum([affine, rotstab, rotstab2, rotstab_gyro])
    assert num_stabilizers_enabled <= 1, "Only one stabilizer can be enabled at a time."

    cam_loader = Load(os.path.join(sequence, cam_name))
    cam_data = cam_loader.get_all()
    WIDTH = cam_loader.get_appended()['res'][0]
    HEIGHT = cam_loader.get_appended()['res'][1]
    n_frames = cam_data['t'].shape[0]
    expected_dt = np.mean(np.diff(cam_data['t'][n_frames // 4: 3*n_frames // 4]))
    expected_fps = int(np.round(1.0 / expected_dt))

    # delta_ts = np.diff(cam_data['t'])
    # plt.plot(delta_ts)
    # plt.show()
    # exit(0)

    K = np.copy(K_in)
    WIDTH = int(WIDTH / downsample)
    HEIGHT = int(HEIGHT / downsample)
    K[0, 0] /= downsample
    K[1, 1] /= downsample
    K[0, 2] /= downsample
    K[1, 2] /= downsample

    if model == "planar":
        cam_map1, cam_map2 = cv2.initUndistortRectifyMap(
            K_in,
            cam_dist,
            np.eye(3),
            K,
            (WIDTH, HEIGHT),
            cv2.CV_32FC1,
        )
    elif model == "fisheye":
        cam_map1, cam_map2 = cv2.fisheye.initUndistortRectifyMap(
            K_in,
            cam_dist,
            np.eye(3),
            K,
            (WIDTH, HEIGHT),
            cv2.CV_32FC1,
        )

    if rotate:
        K[0, 2] = (WIDTH  - 1) - K[0, 2]
        K[1, 2] = (HEIGHT - 1) - K[1, 2]
    K_inv = np.linalg.inv(K)

    tracker = None
    R_c_fc_best = np.eye(3)

    valid_test = np.ones((HEIGHT, WIDTH), dtype=bool)
    crop_rect = [int(WIDTH * margins), int(HEIGHT * margins),
                 WIDTH - int(WIDTH * margins), HEIGHT - int(HEIGHT * margins)]
    valid_test[:crop_rect[1], :] = 0
    valid_test[crop_rect[3]:, :] = 0
    valid_test[:, :crop_rect[0]] = 0
    valid_test[:, crop_rect[2]:] = 0
    max_N_valid = np.sum(valid_test)

    tracker_rotstab_reset_period = 5 * 1/60.0

    tracker_rotstab = None
    last_tracker_rostab_reset_time = None
    R_t0_tk = np.eye(3)
    R_cprimeprime_c0_t0 = np.eye(3)
    R_cprime_c0_t0_list = []
    frame_list = []

    rotstab_omega = None
    rotstab_omega_gyro = None

    if rotstab_gyro:
        flapper_gyro_loader = Load(os.path.join(sequence, 'flapper_data.npz/gyro'))
        # flapper_q_loader = Load(os.path.join(sequence, 'flapper_data.npz/q'))
        Load.time_synchronization(cam_loader, flapper_gyro_loader)#, flapper_q_loader)
        flapper_gyro = flapper_gyro_loader.get_all()
        flapper_t = flapper_gyro['t']
        flapper_omega = flapper_gyro['gyro']
        flapper_omega = np.stack((-flapper_omega[:, 1], flapper_omega[:, 2], flapper_omega[:, 0]), axis=1)
        flapper_omega *= np.pi / 180.0

        # Integrate omega
        flapper_R_cprime_c0 = [np.eye(3)]
        for i in range(0, flapper_omega.shape[0]-1):
            r = R.from_rotvec(flapper_omega[i-1] * (flapper_gyro['t'][i] - flapper_gyro['t'][i-1]))
            flapper_R_cprime_c0.append(r.as_matrix() @ flapper_R_cprime_c0[-1])
        flapper_R_cprime_c0 = np.array(flapper_R_cprime_c0)

    if rotstab_video:
        stabilized_frames_dir = os.path.join(sequence, cam_name + '_stabilized')
        if os.path.exists(stabilized_frames_dir):
            shutil.rmtree(stabilized_frames_dir)
        os.makedirs(stabilized_frames_dir)
        frame_out_i = 0

    if undistort_video:
        undistort_frames_dir = os.path.join(sequence, cam_name + '_undistort')
        if os.path.exists(undistort_frames_dir):
            shutil.rmtree(undistort_frames_dir)
        os.makedirs(undistort_frames_dir)
        frame_out_i_undistort = 0

    if est_april:
        marker_size_m = 19.25 * .0254
        april_tag_detector = AprilPose(K, family="tag36h11", marker_size_m=marker_size_m)    

    last_stabilized_frame = None
    last_valid = None

    start_i = 0
    end_i = len(cam_data['frame'])
    if start_t is not None:
        start_i = np.searchsorted(cam_data['t_received'] - cam_data['t_received'][0], start_t)
    if end_t is not None:
        end_i = np.searchsorted(cam_data['t_received'] - cam_data['t_received'][0], end_t)
    request_saccade = False

    svm_egomotion = SVMEgomotion(K, cam_dist, (WIDTH, HEIGHT))
    event_normal_flow = EventNormalFlow()
    pause = False
    nf_list = []

    stop = Value('i', 0)

    if est_ttc:
        pub_sub = SharedNDArrayPipe(sample_data=(np.zeros((3,)),), max_messages=10000000)
        max_points = int((2000.0 * 36) / downsample**2)
        cloud = PointCloud(stop, pub_sub, mode='drop',
                        x_scale=1280//downsample, y_scale=720//downsample, z_scale=10.0,
                        max_vis_points=max_points, max_buffer_points=max_points)
        cloud.start()

    # Create a time-to-contact color legend
    if est_ttc:
        min_ttc = 0.0
        max_ttc = 3.0
        ttc_colormap = np.zeros((HEIGHT, WIDTH))
        for r in range(HEIGHT):
            ttc_colormap[r, :] = np.linspace(min_ttc, max_ttc, WIDTH)
        ttc_colormap_image = ttc_to_color(ttc_colormap, min_ttc, max_ttc)
        cv2.imshow('ttc_colormap', ttc_colormap_image)

    if frame_filt:
        filtered_frame = None
        frame_alpha = 0.1

    stabilize_window_name = 'stabilized'
    cv2.namedWindow(stabilize_window_name)
    tshift = [0.0]
    val0 = 100
    t_shift_lambda = lambda val, tshift=tshift: on_val_return_trackerbar(val, val0, tshift)
    cv2.createTrackbar('t_shift (ms)', stabilize_window_name, val0, 2*val0, t_shift_lambda)

    recorders = []
    if compute_metrics:
        metrics_str = 'metrics'
        if affine: metrics_str += '_affine'
        elif rotstab: metrics_str += '_rotstab'
        elif rotstab2 and saccade: metrics_str += '_rotstab2_saccade'
        elif rotstab2: metrics_str += '_rotstab2'
        elif rotstab_gyro: metrics_str += '_rotstab_gyro'
        else: metrics_str += '_nostab'

        metrics_dir = os.path.join(sequence, metrics_str)
        foe_recorder = Record(os.path.join(metrics_dir, 'foe'), time_source=None, fields_options=SignalFieldsOptions)
        nf_mag_recorder = Record(os.path.join(metrics_dir, 'nf_mag'), time_source=None, fields_options=SignalFieldsOptions)
        april_pose_recorder = Record(os.path.join(metrics_dir, 'aprilpose'), time_source=None, fields_options=SignalFieldsOptions)
        image_rmse_recorder = Record(os.path.join(metrics_dir, 'image_rmse'), time_source=None, fields_options=SignalFieldsOptions)
        image_sharpness_recorder = Record(os.path.join(metrics_dir, 'image_sharpness'), time_source=None, fields_options=SignalFieldsOptions)
        stabilize_recorder = Record(os.path.join(metrics_dir, 'stabilize_data'), fields_options=StabilizeFieldsOptions)
        recorders.append(foe_recorder)
        recorders.append(nf_mag_recorder)
        recorders.append(april_pose_recorder)
        recorders.append(image_rmse_recorder)
        recorders.append(image_sharpness_recorder)
        recorders.append(stabilize_recorder)

    frames_stabilized = [] # For nonstab frame averaging

    loop_times = [time.perf_counter(),]
    for i in tqdm(range(start_i, end_i), total=end_i - start_i):
        frame_i = i - start_i
        frame_time = cam_data['t'][i]
        frame_time_received = cam_data['t_received'][i]
        frame_name = cam_data['frame'][i]

        frame = np.load(os.path.join(sequence, cam_name, frame_name))
        frame = cv2.remap(frame, cam_map1, cam_map2, cv2.INTER_LINEAR)
        if rotate:
            frame = np.flip(frame, axis=(0, 1))

        frame_float_color = frame.astype(np.float32) / 255.0
        frame_float = frame_float_color
        if len(frame.shape) > 2:
            frame_float = cv2.cvtColor(frame_float_color, cv2.COLOR_BGR2GRAY)

        if num_stabilizers_enabled == 0:
            frames_stabilized.append(frame_float_color)
            if len(frames_stabilized) > N_frames:
                frames_stabilized.pop(0)
            frame_stabilized = avg_frames(frames_stabilized)

            valid = np.ones_like(frame_float_color[:, :, 0], dtype=bool)

            rect = [int(frame_float_color.shape[1] * margins),
                    int(frame_float_color.shape[0] * margins),
                    frame_float_color.shape[1] - int(frame_float_color.shape[1] * margins),
                    frame_float_color.shape[0] - int(frame_float_color.shape[0] * margins)]

            frame_stabilized[:rect[1], :, :] = 0
            frame_stabilized[rect[3]:, :, :] = 0
            frame_stabilized[:, :rect[0], :] = 0
            frame_stabilized[:, rect[2]:, :] = 0
            valid[:rect[1], :] = 0
            valid[rect[3]:, :] = 0
            valid[:, :rect[0]] = 0
            valid[:, rect[2]:] = 0

            if compute_metrics:
                stabilize_recorder.pub(frame_time,
                                       (np.eye(3), np.eye(3)),
                                       t_received=frame_time_received)

        if affine and tracker is None:
            try:
                r = cv2.selectROI('select ROI', frame)
                rect = [r[0], r[1], r[0]+r[2], r[1]+r[3]]
                last = R.from_matrix(np.eye(3)).as_quat()
                template_q_c_to_fc = np.array([last[3], *last[0:3]])
                tracker_p = np.zeros((6,))
                tracker = AffineTrackRotInvariant(
                        patch_coordinates=rect,
                        template_image=frame_float,
                        template_q_c_to_fc=template_q_c_to_fc,
                        K=K,
                        delta_p_stop=0.1,
                        delta_p_mult=1.0,
                        visualize=False,
                        visualize_verbose=False,
                        wait_key=0,
                        stride=tracker_stride,
                        inverse=True,
                        max_update_time=0.02
                    )
            except np.linalg.LinAlgError:
                print('Could not create tracker')
                tracker_p = None
                tracker = None
                tracker2 = None

        if affine and tracker is not None:
            tracker_p = tracker.update(
                    p=tracker_p,
                    frame_gray=frame_float,
                    R_fc_to_c=R_c_fc_best.astype(np.float32),
                )

            frame_stabilized = draw_full_reverse_warp(frame, rect, tracker_p, R_c_fc_best, K)
            # frame_warped_back_cropped_old = frame_warped_back_cropped
            frame_warped_back_cropped = frame_stabilized[rect[1]:rect[3], rect[0]:rect[2]]
            # print(tracker_p)
            cv2.imshow('frame_warped_back', frame_stabilized)
            cv2.imshow('frame_warped_back_cropped', frame_warped_back_cropped)

        if rotstab and tracker_rotstab is None:
            rect = [int(frame.shape[1]*margins),
                    int(frame.shape[0]*margins),
                    frame.shape[1]-int(frame.shape[1]*margins),
                    frame.shape[0]-int(frame.shape[0]*margins)]

            try:
                tracker_rotstab = JHom4pTrackRotInvariant(
                                rect=rect,
                                template_image=frame_float,
                                R_c_fc=np.eye(3), #R_c_fc_best,
                                K=K,
                                delta_p_stop=0.05,
                                stride=tracker_stride,
                                max_steps=50,
                                blur_new_frame=True,
                            )
            except np.linalg.LinAlgError:
                print('Could not create tracker_rotstab')
                tracker_rotstab = None

        if rotstab and tracker_rotstab is not None:
            times = []
            times.append(time.time())
            tracker_rotstab_p = tracker_rotstab.update(frame_gray=frame_float, R_c_fc=R_c_fc_best.astype(np.float32))

            times.append(time.time())
            # Convert from Homography to closest rotation
            tracker_rotstab_p0_stacked = np.concatenate((tracker_rotstab.p0, np.ones((4,)))).reshape((3,4))
            tracker_rotstab_p_stacked  = np.concatenate((tracker_rotstab_p,  np.ones((4,)))).reshape((3,4))

            # print('p-p0', tracker_rotstab_p - tracker_rotstab.p0)

            tracker_rotstab_p0_stacked_spherical = (K_inv @ tracker_rotstab_p0_stacked) / (K_inv @ tracker_rotstab_p0_stacked)[2, :]
            tracker_rotstab_p_stacked_spherical  = (K_inv @ tracker_rotstab_p_stacked)  / (K_inv @ tracker_rotstab_p_stacked )[2, :]

            R_cprime_c0 = least_squares_SO3(tracker_rotstab_p_stacked_spherical.T, tracker_rotstab_p0_stacked_spherical.T)
            # R_cprime_c0 = H_tk_tk+delta
            # H_t0_tk+delta = H_t0_tk @ H_tk_tk+delta
            # R_cprime_c0_t0 = R_t0_tk @ R_cprime_c0
            R_cprime_c0_t0 = R_cprime_c0 @ R_t0_tk.T

            tracker_rotstab_p_stacked_spherical_cprime = R_cprime_c0_t0 @ tracker_rotstab_p0_stacked_spherical
            tracker_rotstab_p_stacked_cprime = K @ (tracker_rotstab_p_stacked_spherical_cprime / tracker_rotstab_p_stacked_spherical_cprime[2, :])

            # tracker_rotstab_p_list.append(np.copy(tracker_rotstab_p_stacked_cprime[0:2, :].flatten()))
            R_cprime_c0_t0_list.append(np.copy(R_cprime_c0_t0))
            frame_list.append(frame_float_color)

            # Trim and check lengths
            if len(R_cprime_c0_t0_list) > N_frames:
                R_cprime_c0_t0_list = R_cprime_c0_t0_list[-N_frames:]
            if len(frame_list) > N_frames:
                frame_list = frame_list[-N_frames:]
            assert len(R_cprime_c0_t0_list) == len(frame_list)

            # Move R_cprimeprime_c0_t0 continuously towards the current orientation with a time constant of 1/5 seconds
            alpha = (1/expected_fps) / 0.2
            R_cprimeprime_c0_t0 = R.from_rotvec(alpha * R.from_matrix(R_cprime_c0_t0 @ R_cprimeprime_c0_t0.T).as_rotvec()).as_matrix() @ R_cprimeprime_c0_t0
            times.append(time.time())

            if compute_metrics:
                stabilize_recorder.pub(frame_time,
                                       (np.copy(R_cprime_c0_t0), np.copy(R_cprimeprime_c0_t0)),
                                       t_received=frame_time_received)

            warped_back_frames = []
            for i in range(len(frame_list)):
                # times2 = []
                # times2.append(time.time())
                R_cprime_c0_t0 = R_cprime_c0_t0_list[i]
                # tracker_rotstab_p_stacked_spherical_cprime = R_cprime_c0_t0 @ tracker_rotstab_p0_stacked_spherical
                # tracker_rotstab_p_stacked_cprime = K @ (tracker_rotstab_p_stacked_spherical_cprime / tracker_rotstab_p_stacked_spherical_cprime[2, :])

                # tracker_rotstab_p_stacked_spherical_cprimeprime = R_cprimeprime_c0_t0 @ tracker_rotstab_p0_stacked_spherical
                # tracker_rotstab_p_stacked_cprimeprime = K @ (tracker_rotstab_p_stacked_spherical_cprimeprime / tracker_rotstab_p_stacked_spherical_cprimeprime[2, :])

                R_cprime_cprimeprime = R_cprime_c0_t0 @ R_cprimeprime_c0_t0.T

                tmp = R_cprime_cprimeprime @ K_inv
                K_cropped = np.ascontiguousarray(K[0:2, :])
                top = K_cropped @ tmp
                bot = tmp[2, :]
                M = np.vstack((top, np.atleast_2d(bot)))

                # times2.append(time.time())
                # warped_back_frames.append(np.array(hom_4p_I_W_p_all_jit(
                #     frame_list[i], tracker_rotstab_p_stacked_cprime[0:2].flatten(),
                #     np.eye(3), K, 1, True, tracker_rotstab_p_stacked_cprimeprime[0:2].flatten())))
                warped_back_frames.append(cv2.warpPerspective(frame_list[i], M, (frame_list[i].shape[1], frame_list[i].shape[0]),
                                                              flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                                                              borderMode=cv2.BORDER_REPLICATE))
                # times2.append(time.time())
                # print('deltas2', times2[-1] - times2[0], np.diff(times2))

            times.append(time.time())
            # frame_avg_warped = np.mean(warped_back_frames, axis=0)
            frame_stabilized = avg_frames(warped_back_frames)
            valid = np.ones_like(frame_stabilized[:, :, 0], dtype=bool)
            times.append(time.time())
            cv2.imshow('tracker_rotstab fixed template moving frame', frame_stabilized)

        if rotstab2 and rotstab_omega is None:
            rotstab_omega = RotStabOmega(frame_float, K,
                                         recorder=stabilize_recorder if compute_metrics else None,
                                         margins=margins,
                                         stride=tracker_stride,
                                         N_frames=N_frames)

        if rotstab2 and rotstab_omega is not None:
            # Decide when to saccade
            # TODO parameter for valid_ratio
            if saccade and last_valid is not None:
                max_valid = (rotstab_omega.rect[2] - rotstab_omega.rect[0]) * (rotstab_omega.rect[3] - rotstab_omega.rect[1])
                valid_ratio = np.sum(last_valid) / max_valid
                if valid_ratio < 0.9:
                    request_saccade = True
                    last_stabilized_frame = None
                    filtered_frame = None

            # Stabilize the frame
            frame_stabilized, _, valid = rotstab_omega.update(frame_float_color, frame_float,
                                                               rot_filt=not saccade,
                                                               saccade=request_saccade,
                                                               valid=not benchmark,
                                                               frame_time=frame_time,
                                                               frame_time_received=frame_time_received)

        if rotstab_gyro and rotstab_omega_gyro is None:
            rotstab_omega_gyro = RotStabOmega(frame_float, K,
                                              recorder=stabilize_recorder if compute_metrics else None,
                                              margins=margins,
                                              stride=tracker_stride,
                                              N_frames=N_frames)

        if rotstab_gyro and rotstab_omega_gyro is not None:
            # Decide when to saccade
            # TODO parameter for valid_ratio
            if saccade and last_valid is not None:
                max_valid = ((rotstab_omega_gyro.rect[2] - rotstab_omega_gyro.rect[0])
                             * (rotstab_omega_gyro.rect[3] - rotstab_omega_gyro.rect[1]))
                valid_ratio = np.sum(last_valid) / max_valid
                if valid_ratio < 0.9:
                    request_saccade = True
                    last_stabilized_frame = None
                    filtered_frame = None

            flapper_R_i = np.searchsorted(flapper_t + tshift[0], frame_time)
            R_cprime_c0_t0 = flapper_R_cprime_c0[flapper_R_i]

            # Stabilize the frame
            frame_stabilized, _, valid = rotstab_omega_gyro.update(frame_float_color, frame_float,
                                                                   R_cprime_c0_t0=R_cprime_c0_t0,
                                                                   rot_filt=not saccade,
                                                                   saccade=request_saccade,
                                                                   valid=not benchmark,
                                                                   frame_time=frame_time,
                                                                   frame_time_received=frame_time_received)
                                                                   #rot_filt_tau=1/0.5, rot_filt_beta=0.0)

        if not frame_filt or filtered_frame is None:
            filtered_frame = frame_stabilized
        else:
            filtered_frame = frame_alpha * frame_stabilized + (1 - frame_alpha) * filtered_frame

        if not benchmark:
            if last_valid is None:
                last_valid = valid
            all_valid = valid & last_valid

        if est_nf:
            # TODO parameter for nf estimation type
            if True:
                if last_stabilized_frame is not None:
                    nf = basic_normal_flow(cv2.cvtColor(last_stabilized_frame, cv2.COLOR_BGR2GRAY), 
                                           cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY),
                                           grad_frame=cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY))
                else:
                    nf = np.zeros_like(frame_stabilized[:, :, 0:2])

            if False:
                nf = event_normal_flow.update(cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY), saccade=request_saccade)

            # Test normal flow estimation
            # Keep normal flows where pixels were valid in this frame and the last
            # and erode the borders to avoid frame border effects
            # TODO parameter
            all_valid_new = cv2.erode(all_valid.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            nf = all_valid_new[:, :, None] * nf

            if compute_metrics:
                # Remove flows with small magnitude
                nf_mag = np.linalg.norm(nf, axis=2)
                nf_sig = nf_mag > 0.0
                nf_valid = nf[all_valid_new.astype(bool) & nf_sig]
                N_all_valid = nf_valid.shape[0]
                if N_all_valid > 0:
                    nf_rms = np.sqrt(np.mean(nf_valid**2))
                    nf_log = np.array((nf_rms, N_all_valid, max_N_valid))
                    nf_mag_recorder.pub(frame_time, (nf_log,), t_received=frame_time_received)

            # Keep a list of the last 6 normal flow fields
            # TODO parameter
            if request_saccade: nf_list = []
            nf_list.append(nf)
            if len(nf_list) > 6:
                nf_list.pop(0)

            # Estimate FOE and depth from normal flow using the SVM algorithm
            if est_foe:
                V, normV_over_Z, normV_over_Z_cloud = svm_egomotion.estimate(nf_list, est_ttc=est_ttc)
                if compute_metrics and V is not None:
                    foe_recorder.pub(frame_time, (V,), t_received=frame_time_received)
            else:
                V = None

        if est_april:
            filtered_frame_uint8 = (255 * filtered_frame).astype(np.uint8)
            detections = april_tag_detector.find_tags(cv2.cvtColor(filtered_frame_uint8, cv2.COLOR_BGR2GRAY))
            assert len(detections) <= 1, "Only one AprilTag should be detected at a time."
            if len(detections) > 0:
                t_wc, R_wc = get_pos_with_april_tag(detections[0])
                if compute_metrics:
                    rec_vec = np.concatenate((t_wc, R_wc.flatten()))
                    april_pose_recorder.pub(frame_time, (rec_vec,), t_received=frame_time_received)

        # Compute image RMSE and sharpness
        if compute_metrics:
            # RMSE between filtered_frame and last_stabilized_frame
            # Include only the valid regions
            if last_stabilized_frame is not None:
                filtered_frame_valid = filtered_frame[all_valid]
                last_stabilized_frame_valid = last_stabilized_frame[all_valid]
                N_all_valid = filtered_frame_valid.shape[0]
                rmse = np.sqrt(np.mean((filtered_frame_valid - last_stabilized_frame_valid) ** 2))
                rmse_log = np.array((rmse, N_all_valid, max_N_valid))
                image_rmse_recorder.pub(frame_time, (rmse_log,), t_received=frame_time_received)
    
            # Sharpness of the filtered frame measured with the first derivative (gradient)
            grad_x = cv2.Sobel(filtered_frame, cv2.CV_64F, 1, 0, ksize=3, scale=0.125)
            grad_y = cv2.Sobel(filtered_frame, cv2.CV_64F, 0, 1, ksize=3, scale=0.125)
            grad_x_valid = grad_x[valid]
            grad_y_valid = grad_y[valid]
            N_sharpness_valid = grad_x_valid.shape[0]
            sharpness_rms = np.sqrt(np.mean((grad_x_valid**2 + grad_y_valid**2)))
            sharpness_log = np.array((sharpness_rms, N_sharpness_valid, max_N_valid))
            # filtered_frame_gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
            # filtered_frame_valid = filtered_frame_gray[valid]
            # Ibar = np.mean(filtered_frame_valid)
            # rms_contrast = np.sqrt(np.mean((filtered_frame_valid - Ibar)**2))
            # N_sharpness_valid = filtered_frame_valid.shape[0]
            # sharpness_log = np.array((rms_contrast, N_sharpness_valid, max_N_valid))
            image_sharpness_recorder.pub(frame_time, (sharpness_log,), t_received=frame_time_received)

        # Update last_X variables, the algorithm is complete
        last_stabilized_frame = filtered_frame
        last_valid = valid
        request_saccade = False

        # Live Visualization
        if not benchmark:
            frame_stabilized_up = cv2.resize(filtered_frame, 
                                            (int(frame_stabilized.shape[1] * downsample),
                                            int(frame_stabilized.shape[0] * downsample)),
                                            interpolation=cv2.INTER_NEAREST)

            if est_nf:
                # Draw the normal flow arrows on the frame
                draw_nonzero_flow_arrows(frame_stabilized_up,
                                        *np.meshgrid((np.arange(0, frame_stabilized.shape[1]) * downsample),
                                                    np.arange(0, frame_stabilized.shape[0]) * downsample),
                                        nf[:, :, 0], nf[:, :, 1],
                                        p_skip=1, mag_scale=2.0 * downsample, color=(0, 0, 1))

                if V is not None:
                    V_unorm = K @ (V / V[2])
                    V_unorm = V_unorm * downsample
                    V_unorm[2] = 1.0
                    c_x = np.clip(V_unorm[0], 0, int(frame_stabilized.shape[1] * downsample)-1)
                    c_y = np.clip(V_unorm[1], 0, int(frame_stabilized.shape[0] * downsample)-1)
                    color = (0, 1, 0) if V[2] > 0 else (1, 0, 0)
                    cv2.circle(frame_stabilized_up, (int(c_x), int(c_y)), 5, color, -1)

                    # Visualize ttc/depth
                    if est_ttc:
                        with np.errstate(divide='ignore', invalid='ignore'):
                            abs_ttc = np.abs(1.0 / normV_over_Z)
                        depth_im = ttc_to_color(abs_ttc, min_ttc, max_ttc)
                        depth_im_resized = cv2.resize(depth_im, 
                                                    (int(frame_stabilized.shape[1] * downsample),
                                                    int(frame_stabilized.shape[0] * downsample)),
                                                    interpolation=cv2.INTER_NEAREST)
                        cv2.imshow('depth', depth_im_resized)
                        pub_sub.pub((normV_over_Z_cloud,))

            if est_april:
                april_tag_detector.draw_detections(frame_stabilized_up, detections, scale=downsample)

            # Show the stabilized frame
            cv2.imshow(stabilize_window_name, frame_stabilized_up)

        if rotstab_video:
            frame_stabilized_up_uint8 = (255*frame_stabilized_up).astype(np.uint8)
            if export_stab_only:
                frame_out = frame_stabilized_up_uint8
            elif export_cropped:
                crop_rect_video = (int(crop_rect[0]*downsample), int(crop_rect[1]*downsample), int(crop_rect[2]*downsample), int(crop_rect[3]*downsample))
                frame_stabilized_up_uint8_cropped = frame_stabilized_up_uint8[crop_rect_video[1]:crop_rect_video[3], crop_rect_video[0]:crop_rect_video[2], :]
                frame_out = frame_stabilized_up_uint8_cropped
            else:
                frame_up = cv2.resize(frame,
                                    (int(frame_stabilized.shape[1] * downsample),
                                    int(frame_stabilized.shape[0] * downsample)),
                                    interpolation=cv2.INTER_NEAREST)
                frame_out = np.hstack((frame_up, frame_stabilized_up_uint8))
            # cv2.imshow('video', frame_out)
            frame_name = os.path.join(stabilized_frames_dir, 'frame_{:06d}.jpg'.format(frame_out_i))
            cv2.imwrite(frame_name, frame_out)
            frame_out_i += 1
            # cv2.imshow('video_frame', frame_out)

        if undistort_video:
            frame_up = cv2.resize(frame,
                                (int(frame_stabilized.shape[1] * downsample),
                                int(frame_stabilized.shape[0] * downsample)),
                                interpolation=cv2.INTER_NEAREST)
            frame_name = os.path.join(undistort_frames_dir, 'frame_{:06d}.jpg'.format(frame_out_i_undistort))
            cv2.imwrite(frame_name, frame_up)
            frame_out_i_undistort += 1
            # cv2.imshow('video_frame', frame_out)

        if benchmark:
            key = None
        if pause:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)

        frame_time = frame_i / expected_fps
        if key == ord('q'):
            break
        elif key == ord('p'):
            pause = ~pause
        elif key == ord(' '):
            pause = not pause
        elif key == ord('s'):
            request_saccade = True
            last_stabilized_frame = None
        elif (key == ord('r')
              or last_tracker_rostab_reset_time is None
              or frame_time - last_tracker_rostab_reset_time > tracker_rotstab_reset_period):
            last_tracker_rostab_reset_time = frame_time

            # Reset stabilization with moving reference frame and to a switching template
            frame_list = []
            # tracker_rotstab_p_list = []
            R_cprime_c0_t0_list = []
            # If there is an existing tracker, update the accumulate warp that maintains consistency when the template switches
            if rotstab and tracker_rotstab is not None:
                # Solve for R_t_k-1_t_k with DLT
                # H_t0_tk = H_t0_tk @ H_tk-1_tk
                tracker_rotstab_p0_stacked = np.concatenate((tracker_rotstab.p0, np.ones((4,)))).reshape((3,4))
                tracker_rotstab_p_stacked  = np.concatenate((tracker_rotstab_p,  np.ones((4,)))).reshape((3,4))
                tracker_rotstab_p0_stacked_spherical = (K_inv @ tracker_rotstab_p0_stacked) / (K_inv @ tracker_rotstab_p0_stacked)[2, :]
                tracker_rotstab_p_stacked_spherical  = (K_inv @ tracker_rotstab_p_stacked)  / (K_inv @ tracker_rotstab_p_stacked )[2, :]

                R_t_k_t_kminus1 = least_squares_SO3(tracker_rotstab_p_stacked_spherical.T, tracker_rotstab_p0_stacked_spherical.T)
                R_t_k_t_kminus1 = R_c_fc_best @ R_t_k_t_kminus1
                R_t0_tk = R_t0_tk @ R_t_k_t_kminus1.T

                # print('reset tracker_rotstab', time_source.time())
                tracker_rotstab = None
            # If there is no tracker (or it was destroyed above)
            if rotstab and tracker_rotstab is None:
                rect = [int(frame.shape[1]*margins),
                        int(frame.shape[0]*margins),
                        frame.shape[1]-int(frame.shape[1]*margins),
                        frame.shape[0]-int(frame.shape[0]*margins)]

                try:
                    tracker_rotstab = JHom4pTrackRotInvariant(
                                 rect=rect,
                                 template_image=frame_float,
                                 R_c_fc=np.eye(3), #R_c_fc_best,
                                 K=K,
                                 delta_p_stop=0.05,
                                 stride=tracker_stride,
                                 max_steps=50,
                                 blur_new_frame=True,
                                )
                except np.linalg.LinAlgError:
                    print('Could not create tracker_rotstab')
                    tracker_rotstab = None

            # Reset rotstab's template but keep the absolute and filtered orientation states
            if rotstab2 and rotstab_omega is not None:
                rotstab_omega.update_template()

        loop_times.append(time.perf_counter())

    if benchmark:
        loop_times = np.array(loop_times)[50:] # Skip the first 50 frames to avoid counting startup effects
        avg_fps = 1.0/((loop_times[-1] - loop_times[0]) / loop_times.shape[0])
        dt = np.gradient(loop_times)
        max_fps = 1.0 / dt.min()
        min_fps = 1.0 / dt.max()
        med_fps = 1.0 / np.median(dt)
        print('FPS', avg_fps, med_fps, min_fps, max_fps)

        import matplotlib.pyplot as plt
        plt.plot(np.gradient(loop_times)) # Skip the first 50 frames to avoid counting startup effects
        plt.show()

    for signal_recorder in recorders:
        if signal_recorder is not None:
            signal_recorder.close()

    if args.rotstab_video:
        print('Encoding stabilized video')
        proc = subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel' , 'warning',
                               '-framerate', str(expected_fps), '-pattern_type',
                               'glob', '-i', os.path.join(stabilized_frames_dir, 'frame_*.jpg'),
                               '-c:v', 'libx264', os.path.join(sequence, cam_name + '_stabilized.mp4')])

    if args.undistort_video:
        print('Encoding undistort video')
        proc = subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel' , 'warning',
                               '-framerate', str(expected_fps), '-pattern_type',
                               'glob', '-i', os.path.join(undistort_frames_dir, 'frame_*.jpg'),
                               '-c:v', 'libx264', os.path.join(sequence, cam_name + '_undistort.mp4')])

def run_args(args):
    if args.flapper:
        calibration_loader = Load('data/calibration_joint_25_08_18')
        model = "fisheye"
        K        = np.array(calibration_loader.get_appended()['K'])
        cam_dist = np.array(calibration_loader.get_appended()['dist'])
    else:
        cam_loader = Load(os.path.join(args.sequence, args.cam))
        cam_data = cam_loader.get_appended()
        model = "planar"
        K = np.array(cam_data["K"])
        cam_dist = np.array(cam_data["dist"])

    stabilize(args.sequence, args.cam,
            model, K, cam_dist, args.rotate, args.downsample,
            args.affine, args.rotstab,
            args.rotstab2, args.rotstab_gyro,
            args.saccade,
            args.rotstab_video,
            args.export_cropped,
            args.export_stab_only,
            args.undistort_video,
            args.frame_filt,
            args.est_ttc, args.est_nf, args.est_foe, args.april,
            args.metrics,
            args.benchmark,
            margins=args.margins,
            tracker_stride=args.stride,
            N_frames=args.N_frames,
            start_t=args.start_t,
            end_t=args.end_t)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    cv2.setNumThreads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, default=None)
    parser.add_argument('--cam', type=str, default='cam_front')
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--margins', type=float, default=0.125)
    parser.add_argument('--stride', type=float, default=1)
    parser.add_argument('--downsample', type=float, default=1.0)
    parser.add_argument('--N_frames', type=int, default=6)
    parser.add_argument('--flapper', action='store_true')
    parser.add_argument('--affine', action='store_true')
    parser.add_argument('--rotstab', action='store_true')
    parser.add_argument('--rotstab2', action='store_true')
    parser.add_argument('--rotstab_gyro', action='store_true')
    parser.add_argument('--saccade', action='store_true')
    parser.add_argument('--rotstab_video', action='store_true')
    parser.add_argument('--export_cropped', action='store_true')
    parser.add_argument('--export_stab_only', action='store_true')
    parser.add_argument('--undistort_video', action='store_true')
    parser.add_argument('--frame_filt', action='store_true')
    parser.add_argument('--est_ttc', action='store_true')
    parser.add_argument('--est_nf', action='store_true')
    parser.add_argument('--est_foe', action='store_true')
    parser.add_argument('--april', action='store_true')
    parser.add_argument('--metrics', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--start_t', type=float, default=None)
    parser.add_argument('--end_t', type=float, default=None)
    args = parser.parse_args()
    run_args(args)
