from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from vme_research.hardware.record import Load
from vme_research.algorithms.least_squares_lie_group import least_squares_scale_SE3
from vme_research.algorithms.sample_buffer import SampleBuffer
from scipy.spatial.transform import Rotation as R
from scipy import signal
import os
from pprint import pprint
import json

sequence_folder = 'data/2025_08_21_collection_vicon2'
no_vicon = False

sequence_folder = 'data/2025_08_21_collection_outside'
no_vicon = True

folder_settings = json.load(open(os.path.join(sequence_folder, 'analysis_settings.json')))
flapper_vicon_name = folder_settings['flapper_vicon_name']
calibration_folder = os.path.join('data', folder_settings['calibration'])

sequences = list(sorted(glob(os.path.join(sequence_folder, 'sequence_*'))))
sequence = sequences[0]

methods = ['nostab', 'rotstab2', 'rotstab2_saccade']#, 'rotstab_gyro']
# metrics_datas = ['aprilpose', 'foe', 'image_mse', 'image_sharpness', 'stabilize_data']
metrics_datas = ['image_rmse', 'image_sharpness', 'stabilize_data', 'nf_mag']

calibration_loader = Load(calibration_folder)
model = "fisheye"
K        = np.array(calibration_loader.get_appended()['K'])
cam_dist = np.array(calibration_loader.get_appended()['dist'])
t_vc_c = np.array(calibration_loader.get_appended()['t_vc_c'])
R_vc_c = np.array(calibration_loader.get_appended()['R_vc_c'])

results_table_data = {}
for sequence in sequences:
    settings = folder_settings.get(os.path.basename(sequence))
    seq_nam = os.path.basename(sequence)

    loaders = {}
    for method in methods:
        loaders[method] = {}
    loaders_list = []
    for method in methods:
        for metric_data in metrics_datas:
            loader = Load(os.path.join(sequence, f'metrics_{method}/{metric_data}'))
            data = loader.get_all()
            # print(os.path.join(sequence, f'metrics_{method}/{metric_data}'), data['t'].shape, data['t_received'].shape)
            loaders[method][metric_data] = loader
            loaders_list.append(loader)
            # Load.time_synchronization(stabilize_data_loader, flapper_gyro_loader, flapper_q_loader)

    if not no_vicon:
        vicon_loader = Load(os.path.join(sequence, 'flapper_data.npz/vicon'))
        loaders['vicon'] = vicon_loader
        loaders_list.append(vicon_loader)

    gyro_loader = Load(os.path.join(sequence, 'flapper_data.npz/gyro'))
    loaders['gyro'] = gyro_loader
    loaders_list.append(gyro_loader)

    camera_loader = Load(os.path.join(sequence, 'cam_front'))
    loaders['cam_front'] = camera_loader
    loaders_list.append(camera_loader)

    Load.time_synchronization(loaders_list[0], *loaders_list[1:])

    t_start = camera_loader.get_all()['t'][settings['frame_i_start']]
    t_end = camera_loader.get_all()['t'][settings['frame_i_end']]
    # print(sequence, 't_start', t_start, 't_end', t_end)

    if not no_vicon:
        if 'avgV' not in results_table_data:
            results_table_data['avgV'] = {}
        if 'avgw_vicon' not in results_table_data:
            results_table_data['avgw_vicon'] = {}
        vicon_data = vicon_loader.get_all()
        vicon_t = vicon_data['t']
        vicon_flapper = vicon_data[folder_settings['flapper_vicon_name']]

        vicon_start_i = np.searchsorted(vicon_t, t_start, side='left')
        vicon_end_i = np.searchsorted(vicon_t, t_end, side='right')
        vicon_t = vicon_t[vicon_start_i:vicon_end_i]
        vicon_flapper = vicon_flapper[vicon_start_i:vicon_end_i]

        nan_ratio = np.sum(np.isnan(vicon_flapper[:, 0])) / vicon_flapper.shape[0]
        assert nan_ratio < 0.02 # Vicon loses track occasionally, make sure its rare

        no_data = np.isnan(vicon_flapper).any(axis=1)
        vicon_t = vicon_t[~no_data]
        vicon_flapper = vicon_flapper[~no_data]
        assert not np.any(np.isnan(vicon_flapper))

        t_wo = vicon_flapper[:, 0:3]
        q_wo_wxyz = vicon_flapper[:, 3:7]

        # Estimate omega from q_wo_wxyz
        # convert q_wo_wxyz to scipy rotation objects
        R_wo = R.from_quat(q_wo_wxyz, scalar_first=True).as_matrix()
        vicon_w_t = vicon_t[:-1]
        vicon_w = np.zeros((R_wo.shape[0]-1, 3))
        for i in range(R_wo.shape[0]-1):
            dR_o1o2 = R_wo[i, :, :].T @ R_wo[i + 1, :, :]
            vicon_w[i, :] = R.from_matrix(dR_o1o2).as_rotvec() * 180 / np.pi / (vicon_t[i + 1] - vicon_t[i])
        
        norm_omega = np.linalg.norm(vicon_w, axis=1)
        avg_omega = np.sum(norm_omega * np.gradient(vicon_w_t)) / (vicon_w_t[-1] - vicon_w_t[0])
        results_table_data['avgw_vicon'][seq_nam] = avg_omega.item()

        # Estimate V
        v_wo = np.gradient(t_wo, vicon_t, axis=0)
        norm_vwo = np.linalg.norm(v_wo, axis=1)
        avg_v = np.sum(norm_vwo * np.gradient(vicon_t)) / (vicon_t[-1] - vicon_t[0])

        results_table_data['avgV'][seq_nam] = avg_v.item()

        # plt.figure(figsize=(12, 12))
        # plt.subplot(4, 1, 1)
        # plt.plot(vicon_t, t_wo[:, 0], label='t_wo_x')
        # plt.plot(vicon_t, t_wo[:, 1], label='t_wo_y')
        # plt.plot(vicon_t, t_wo[:, 2], label='t_wo_z')
        # plt.title('Translation (world)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Position (m)')
        # plt.legend()

        # plt.subplot(4, 1, 2)
        # plt.plot(vicon_t, v_wo[:, 0], label='v_wo_x')
        # plt.plot(vicon_t, v_wo[:, 1], label='v_wo_y')
        # plt.plot(vicon_t, v_wo[:, 2], label='v_wo_z')
        # plt.title('Velocity (world)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Velocity (m/s)')
        # plt.legend()

        # plt.subplot(4, 1, 3)
        # plt.plot(vicon_t, q_wo_wxyz[:, 0], label='q_wo_w')
        # plt.plot(vicon_t, q_wo_wxyz[:, 1], label='q_wo_x')
        # plt.plot(vicon_t, q_wo_wxyz[:, 2], label='q_wo_y')
        # plt.plot(vicon_t, q_wo_wxyz[:, 3], label='q_wo_z')
        # plt.title('Orientation (world)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Quaternion')
        # plt.legend()

        # plt.subplot(4, 1, 4)
        # plt.plot(vicon_w_t, vicon_w[:, 0], label='vicon_w_x')
        # plt.plot(vicon_w_t, vicon_w[:, 1], label='vicon_w_y')
        # plt.plot(vicon_w_t, vicon_w[:, 2], label='vicon_w_z')
        # plt.title('Angular Velocity (world)')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Angular Velocity (deg/s)')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()

    if 'avgw' not in results_table_data:
        results_table_data['avgw'] = {}
    gyro_data = gyro_loader.get_all()
    omega_t = gyro_data['t']
    omega = gyro_data['gyro']
    assert not np.any(np.isnan(omega))

    omega_start_i = np.searchsorted(omega_t, t_start, side='left')
    omega_end_i = np.searchsorted(omega_t, t_end, side='right')
    omega_t = omega_t[omega_start_i:omega_end_i]
    omega = omega[omega_start_i:omega_end_i]

    norm_omega = np.linalg.norm(omega, axis=1)
    avg_omega = np.sum(norm_omega * np.gradient(omega_t)) / (omega_t[-1] - omega_t[0])
    results_table_data['avgw'][seq_nam] = avg_omega.item()

    if 'avgw_image' not in results_table_data:
        results_table_data['avgw_image'] = {}
    if 'avgw_stab' not in results_table_data:
        results_table_data['avgw_stab'] = {}
    for method in methods:
        stab_loader = loaders[method]['stabilize_data']

        R_t = stab_loader.get_all()['t']
        R_cprime_c0 = stab_loader.get_all()['R_cprime_c0_t0']
        R_cprimeprime_c0 = stab_loader.get_all()['R_cprimeprime_c0_t0']
        assert not np.any(np.isnan(R_cprime_c0))
        assert not np.any(np.isnan(R_cprimeprime_c0))

        start_i = np.searchsorted(R_t, t_start, side='left')
        end_i = np.searchsorted(R_t, t_end, side='right')
        R_t = R_t[start_i:end_i]
        R_cprime_c0 = R_cprime_c0[start_i:end_i]
        R_cprimeprime_c0 = R_cprimeprime_c0[start_i:end_i]

        w_t = R_t[:-1]
        image_w = np.zeros((R_t.shape[0]-1, 3))
        stab_w = np.zeros((R_t.shape[0]-1, 3))
        for i in range(R_t.shape[0]-1):
            dR_o1o2 = R_cprime_c0[i, :, :] @ R_cprime_c0[i + 1, :, :].T
            image_w[i, :] = R.from_matrix(dR_o1o2).as_rotvec() * 180 / np.pi / (R_t[i + 1] - R_t[i])

            dR_o1o2 = R_cprimeprime_c0[i, :, :] @ R_cprimeprime_c0[i + 1, :, :].T
            stab_w[i, :] = R.from_matrix(dR_o1o2).as_rotvec() * 180 / np.pi / (R_t[i + 1] - R_t[i])

        if method not in results_table_data['avgw_image']:
            results_table_data['avgw_image'][method] = {}
        if method not in results_table_data['avgw_stab']:
            results_table_data['avgw_stab'][method] = {}

        norm_omega = np.linalg.norm(image_w, axis=1)
        avg_omega = np.sum(norm_omega * np.gradient(w_t)) / (w_t[-1] - w_t[0])
        results_table_data['avgw_image'][method][seq_nam] = avg_omega.item()

        norm_omega = np.linalg.norm(stab_w, axis=1)
        avg_omega = np.sum(norm_omega * np.gradient(w_t)) / (w_t[-1] - w_t[0])
        results_table_data['avgw_stab'][method][seq_nam] = avg_omega.item()
        # print(results_table_data['avgw_image'][method][seq_nam])
        # print(results_table_data['avgw_stab'][method][seq_nam])

        # Plot all results in 3 plots
        # plt.figure(figsize=(12, 12))
        # plt.subplot(3, 1, 1)
        # plt.plot(w_t, image_w[:, 0], label='image_w_x')
        # plt.plot(w_t, stab_w[:, 0], label='stab_w_x')
        # plt.ylabel('Angular Velocity (deg/s)')
        # plt.legend()

        # plt.subplot(3, 1, 2)
        # plt.plot(w_t, image_w[:, 1], label='image_w_y')
        # plt.plot(w_t, stab_w[:, 1], label='stab_w_y')
        # plt.ylabel('Angular Velocity (deg/s)')
        # plt.legend()

        # plt.subplot(3, 1, 3)
        # plt.plot(w_t, image_w[:, 2], label='image_w_z')
        # plt.plot(w_t, stab_w[:, 2], label='stab_w_z')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Angular Velocity (deg/s)')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()

    # Plot omega in 3 plots
    # plt.figure(figsize=(12, 12))
    # plt.subplot(3, 1, 1)
    # plt.plot(omega_t, omega[:, 0], label='omega_x')
    # plt.ylabel('Angular Velocity (deg/s)')
    # plt.legend()

    # plt.subplot(3, 1, 2)
    # plt.plot(omega_t, omega[:, 1], label='omega_y')
    # plt.ylabel('Angular Velocity (deg/s)')
    # plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(omega_t, omega[:, 2], label='omega_z')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angular Velocity (deg/s)')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    # exit(0)

    if 'mse' not in results_table_data:
        results_table_data['mse'] = {}
    for method in methods:
        image_error_loader = loaders[method]['image_rmse']
        i_start = np.searchsorted(image_error_loader.get_all()['t'], t_start, side='left')
        i_end = np.searchsorted(image_error_loader.get_all()['t'], t_end, side='right')

        stat_data = image_error_loader.get_all()['x']
        assert not np.any(np.isnan(stat_data))
        stat_data = stat_data[i_start:i_end, :]

        rmse = stat_data[:, 0].flatten()
        N_valid = stat_data[:, 1].flatten()
        N_max = stat_data[:, 2].flatten()

        mse = np.square(rmse)
        mse_all_frames = np.sum(mse * N_valid) / np.sum(N_valid)
        rmse_all_frames = np.sqrt(mse_all_frames)
        # print(sequence, method, rmse_all_frames)

        if method not in results_table_data['mse']:
            results_table_data['mse'][method] = {}
        results_table_data['mse'][method][seq_nam] = rmse_all_frames.item()

    if 'sharpness' not in results_table_data:
        results_table_data['sharpness'] = {}
    if 'valid' not in results_table_data:
        results_table_data['valid'] = {}
    for method in methods:
        loader = loaders[method]['image_sharpness']
        i_start = np.searchsorted(loader.get_all()['t'], t_start, side='left')
        i_end = np.searchsorted(loader.get_all()['t'], t_end, side='right')

        stat_data = loader.get_all()['x']
        assert not np.any(np.isnan(stat_data))
        stat_data = stat_data[i_start:i_end, :]

        sharpness_rmse = stat_data[:, 0].flatten()
        N_valid = stat_data[:, 1].flatten()
        N_max = stat_data[:, 2].flatten()

        sharpness_mse = np.square(sharpness_rmse)
        sharpness_mse_all_frames = np.sum(sharpness_mse * N_valid) / np.sum(N_valid)
        sharpness_rmse_all_frames = np.sqrt(sharpness_mse_all_frames)

        if method not in results_table_data['sharpness']:
            results_table_data['sharpness'][method] = {}
        results_table_data['sharpness'][method][seq_nam] = sharpness_rmse_all_frames.item()

        if method not in results_table_data['sharpness']:
            results_table_data['sharpness'][method] = {}
        results_table_data['sharpness'][method][seq_nam] = sharpness_rmse_all_frames.item()

        if method not in results_table_data['valid']:
            results_table_data['valid'][method] = {}
        results_table_data['valid'][method][seq_nam] = 100*np.mean(N_valid / N_max).item()

    if 'nf_mag' not in results_table_data:
        results_table_data['nf_mag'] = {}
    for method in methods:
        loader = loaders[method]['nf_mag']
        i_start = np.searchsorted(loader.get_all()['t'], t_start, side='left')
        i_end = np.searchsorted(loader.get_all()['t'], t_end, side='right')

        stat_data = loader.get_all()['x']
        assert not np.any(np.isnan(stat_data))
        stat_data = stat_data[i_start:i_end, :]

        nf_mag_rms = stat_data[:, 0].flatten()
        N_valid = stat_data[:, 1].flatten()
        N_max = stat_data[:, 2].flatten()

        nf_mag_ms = np.square(nf_mag_rms)
        nf_mag_mse_all_frames = np.sum(nf_mag_ms * N_valid) / np.sum(N_valid)
        nf_mag_rmse_all_frames = np.sqrt(nf_mag_mse_all_frames)

        if method not in results_table_data['nf_mag']:
            results_table_data['nf_mag'][method] = {}
        results_table_data['nf_mag'][method][seq_nam] = nf_mag_rmse_all_frames.item()

pprint(results_table_data)

# Make a CSV for the data
csv_strings = []
sep='&'
for metric in ['mse', 'sharpness', 'nf_mag']:
    for method in results_table_data[metric]:
        csv_string = f"{metric},{method}"
        for seq_name, value in results_table_data[metric][method].items():
            csv_string = f"{csv_string},{value:0.3f}"
        csv_strings.append(csv_string)

if not no_vicon:
    for metric in ['avgV']:
        csv_string = f"{metric}"
        for seq_name, value in results_table_data[metric].items():
            csv_string = f"{csv_string},{value:0.3f}"
        csv_strings.append(csv_string)

avgw_metrics = ['avgw']
if not no_vicon: avgw_metrics.append('avgw_vicon')
for metric in avgw_metrics:
    csv_string = f"{metric}"
    for seq_name, value in results_table_data[metric].items():
        csv_string = f"{csv_string},{value:3.0f}"
    csv_strings.append(csv_string)

for metric in ['avgw_image', 'avgw_stab']:
    for method in results_table_data[metric]:
        csv_string = f"{metric},{method}"
        for seq_name, value in results_table_data[metric][method].items():
            csv_string = f"{csv_string},{value:3.0f}"
        csv_strings.append(csv_string)

for metric in ['valid']:
    for method in results_table_data[metric]:
        csv_string = f"{metric},{method}"
        for seq_name, value in results_table_data[metric][method].items():
            csv_string = f"{csv_string},{value:3.1f}"
        csv_strings.append(csv_string)

# pprint(csv_strings)
csv_file = "\n".join(csv_strings)
with open(os.path.join(sequence_folder, 'table_results.csv'), 'w') as f:
    f.write(csv_file)

# Append \\ and print out for copy and pasting
latex_strings = [s.replace(",", " & ") for s in csv_strings]
latex_strings = [s + " \\\\" for s in latex_strings]
for s in latex_strings:
    print(s)

# pprint(latex_file, width=300)

    # pprint(loaders)

    # vicon_data = vicon_loader.get_all()
    # vicon_t = vicon_data['t']
    # vicon_flapper_pos = vicon_data[flapper_vicon_name][:, :3]
    # vicon_flapper_qwxyz = vicon_data[flapper_vicon_name][:, 3:7]
    # vicon_buffer = SampleBuffer()
    # for t, x in zip(vicon_t, vicon_flapper_pos):
    #     vicon_buffer.append(t, x)

    # def interp_buffer(times, buf):
    #     samples = []
    #     for t in times:
    #         _, x = buf.get(t)
    #         samples.append(x)
    #     return np.array(samples)

    # Compute the vicon velocity using finite diff
    # vicon_flapper_vel = np.gradient(vicon_flapper_pos, vicon_t, axis=0)
    # vicon_flapper_ego_t = []
    # vicon_flapper_ego = []
    # for i in range(0, vicon_flapper_vel.shape[0]):
    #     if not np.any(np.isnan(vicon_flapper_qwxyz[i])):
    #         R_w_vc = R.from_quat(vicon_flapper_qwxyz[i], scalar_first=True).as_matrix()
    #         R_w_c = R_w_vc @ R_vc_c
    #         V_ego_w = vicon_flapper_vel[i] #/ np.linalg.norm(vicon_flapper_vel[i]) if np.linalg.norm(vicon_flapper_vel[i]) > 0 else np.zeros(3)
    #         V_ego_c = R_w_c.T @ V_ego_w
    #         vicon_flapper_ego_t.append(vicon_t[i])
    #         vicon_flapper_ego.append(V_ego_c)
    # vicon_flapper_ego_t = np.array(vicon_flapper_ego_t)
    # vicon_flapper_ego = np.array(vicon_flapper_ego)

    # vicon_ego_buffer = SampleBuffer()
    # for t, x in zip(vicon_flapper_ego_t, vicon_flapper_ego):
    #     vicon_ego_buffer.append(t, x)

    # Start with getting the apriltag positions on a graph
    # aligned_aprils = {}
    # for method in methods:
    #     april_data = loaders[method]['aprilpose'].get_all()
    #     april_pos_t = april_data['t']
    #     april_pos = april_data['x'][:, :3]

    #     vicon_flapper_pos_interp = interp_buffer(april_pos_t, vicon_buffer)

    #     c_ab, R_ab, t_ab = least_squares_scale_SE3(vicon_flapper_pos_interp, april_pos)
    #     april_pos_aligned = c_ab * (R_ab @ april_pos.T).T + t_ab
    #     aligned_aprils[method] = april_pos_aligned

    # plt.figure(figsize=(10, 6))
    # for i in range(3):
    #     plt.subplot(3, 1, i + 1)
    #     plt.plot(vicon_data['t'], vicon_flapper_pos[:, i], label=f'Vicon {["x", "y", "z"][i]}')
    #     plt.legend()
    # for method in methods:
    #     for i in range(3):
    #         plt.subplot(3, 1, i + 1)
    #         april_data = loaders[method]['aprilpose'].get_all()
    #         plt.plot(april_data['t'], aligned_aprils[method][:, i], label=f'{method} {["x", "y", "z"][i]}')
    #         plt.legend()
    # plt.subplot(3, 1, 1)
    # plt.title('AprilTag Positions')
    # plt.tight_layout()

    # Plot FOE
    # plt.figure(figsize=(10, 6))
    # for i in range(3):
    #     plt.subplot(3, 1, i + 1)
    #     plt.plot(vicon_flapper_ego_t, vicon_flapper_ego[:, i], label=f'Vicon {["x", "y", "z"][i]}')
    #     plt.legend()
    # for method in methods:
    #     for i in range(3):
    #         plt.subplot(3, 1, i + 1)
    #         foe_data = loaders[method]['foe'].get_all()
    #         vicon_flapper_ego_interp = interp_buffer(foe_data['t'], vicon_ego_buffer)
    #         foe_times_V = foe_data['x'][:, i] * np.linalg.norm(vicon_flapper_ego_interp, axis=1)
    #         plt.plot(foe_data['t'], foe_times_V, label=f'{method} {["x", "y", "z"][i]}')
    #         plt.legend()
    # plt.subplot(3, 1, 1)
    # plt.title('FOE * ||V||')
    # plt.tight_layout()

    # plt.figure(figsize=(10, 6))
    # for method in methods:
    #     data = loaders[method]['image_rmse'].get_all()
    #     plt.subplot(2, 1, 1)
    #     plt.plot(data['t'], data['x'][:, 0], label=f'{method} mse')
    #     plt.ylim([0, None])
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(data['t'], data['x'][:, 1] / data['x'][:, 2], label=f'{method} coverage')
    #     plt.ylim([-0.1, 1.1])
    #     plt.legend()
    # plt.subplot(2, 1, 1)
    # plt.title('Image MSE')
    # plt.tight_layout()

    # plt.figure(figsize=(10, 6))
    # for method in methods:
    #     data = loaders[method]['image_sharpness'].get_all()
    #     plt.subplot(2, 1, 1)
    #     plt.plot(data['t'], data['x'][:, 0], label=f'{method} sharpness')
    #     plt.ylim([0, None])
    #     plt.legend()
    #     plt.subplot(2, 1, 2)
    #     plt.plot(data['t'], data['x'][:, 1] / data['x'][:, 2], label=f'{method} coverage')
    #     plt.ylim([-0.1, 1.1])
    #     plt.legend()
    # plt.subplot(2, 1, 1)
    # plt.title('Image Sharpness')
    # plt.tight_layout()

    # plt.show()

