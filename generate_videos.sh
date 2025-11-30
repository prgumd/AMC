#!/bin/bash
set -e

downsample=4
stride=1
N_frames=6
margins=0.125

# Prepare output folders
rm -rf data/2025_08_21_collection_outside_videos/sequence_000000
rm -rf data/2025_08_21_collection_outside_videos/sequence_000001
rm -rf data/2025_08_21_collection_outside_videos/sequence_000002
rm -rf data/2025_08_21_collection_outside_videos/sequence_000003
rm -rf data/2025_08_21_collection_outside_videos/sequence_000004
mkdir -p data/2025_08_21_collection_outside_videos/sequence_000000
mkdir -p data/2025_08_21_collection_outside_videos/sequence_000001
mkdir -p data/2025_08_21_collection_outside_videos/sequence_000002
mkdir -p data/2025_08_21_collection_outside_videos/sequence_000003
mkdir -p data/2025_08_21_collection_outside_videos/sequence_000004

# Generate undisorted, unstabilized video and undistorted stabilized video
N_frames=6
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000000 --rotstab_video --undistort_video --export_stab_only --rotstab2 & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000001 --rotstab_video --undistort_video --export_stab_only --rotstab2 & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000002 --rotstab_video --undistort_video --export_stab_only --rotstab2 & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000003 --rotstab_video --undistort_video --export_stab_only --rotstab2 & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000004 --rotstab_video --undistort_video --export_stab_only --rotstab2 & PID4=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_rotstab_6
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_rotstab_6
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_rotstab_6
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_rotstab_6
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_rotstab_6

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_undistort data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_undistort
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_undistort data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_undistort
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_undistort data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_undistort
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_undistort data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_undistort
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_undistort data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_undistort

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_rotstab_6.mp4
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_rotstab_6.mp4
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_rotstab_6.mp4
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_rotstab_6.mp4
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_rotstab_6.mp4

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_undistort.mp4 data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_undistort.mp4
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_undistort.mp4 data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_undistort.mp4
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_undistort.mp4 data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_undistort.mp4
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_undistort.mp4 data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_undistort.mp4
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_undistort.mp4 data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_undistort.mp4

# Generate stabilized video N_frames = 1
N_frames=1
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000000 --rotstab_video --export_stab_only --rotstab2 & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000001 --rotstab_video --export_stab_only --rotstab2 & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000002 --rotstab_video --export_stab_only --rotstab2 & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000003 --rotstab_video --export_stab_only --rotstab2 & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000004 --rotstab_video --export_stab_only --rotstab2 & PID4=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_rotstab_1
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_rotstab_1
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_rotstab_1
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_rotstab_1
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_rotstab_1

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_rotstab_1.mp4
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_rotstab_1.mp4
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_rotstab_1.mp4
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_rotstab_1.mp4
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_rotstab_1.mp4

# Generate stabilized video N_frames = 6 and saccade
N_frames=6
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000000 --rotstab_video --export_stab_only --rotstab2 --saccade & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000001 --rotstab_video --export_stab_only --rotstab2 --saccade & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000002 --rotstab_video --export_stab_only --rotstab2 --saccade & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000003 --rotstab_video --export_stab_only --rotstab2 --saccade & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --sequence data/2025_08_21_collection_outside/sequence_000004 --rotstab_video --export_stab_only --rotstab2 --saccade & PID4=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_rotstab_6_saccade
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_rotstab_6_saccade
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_rotstab_6_saccade
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_rotstab_6_saccade
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_stabilized data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_rotstab_6_saccade

mv data/2025_08_21_collection_outside/sequence_000000/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000000/cam_front_stabilized_rotstab_6_saccade.mp4
mv data/2025_08_21_collection_outside/sequence_000001/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000001/cam_front_stabilized_rotstab_6_saccade.mp4
mv data/2025_08_21_collection_outside/sequence_000002/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000002/cam_front_stabilized_rotstab_6_saccade.mp4
mv data/2025_08_21_collection_outside/sequence_000003/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000003/cam_front_stabilized_rotstab_6_saccade.mp4
mv data/2025_08_21_collection_outside/sequence_000004/cam_front_stabilized.mp4 data/2025_08_21_collection_outside_videos/sequence_000004/cam_front_stabilized_rotstab_6_saccade.mp4
