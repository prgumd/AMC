#!/bin/bash
set -e

downsample=4
stride=1
N_frames=6
margins=0.125

# N_frames=1
# python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000001 --start_t 5
# exit

N_frames=1
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000000 & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000001 & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000002 & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000003 & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000004 & PID4=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000005 & PID5=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000006 & PID6=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000007 & PID7=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000008 & PID8=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000009 & PID9=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9

N_frames=6
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000000 --rotstab2 & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000001 --rotstab2 & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000002 --rotstab2 & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000003 --rotstab2 & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000004 --rotstab2 & PID4=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000005 --rotstab2 & PID5=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000006 --rotstab2 & PID6=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000007 --rotstab2 & PID7=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000008 --rotstab2 & PID8=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000009 --rotstab2 & PID9=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9

python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000000 --rotstab2 --saccade & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000001 --rotstab2 --saccade & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000002 --rotstab2 --saccade & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000003 --rotstab2 --saccade & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000004 --rotstab2 --saccade & PID4=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000005 --rotstab2 --saccade & PID5=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000006 --rotstab2 --saccade & PID6=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000007 --rotstab2 --saccade & PID7=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000008 --rotstab2 --saccade & PID8=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_vicon2/sequence_000009 --rotstab2 --saccade & PID9=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9
