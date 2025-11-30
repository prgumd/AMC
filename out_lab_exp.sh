#!/bin/bash
set -e

downsample=4
stride=1
N_frames=6
margins=0.125

N_frames=1
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000000 & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000001 & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000002 & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000003 & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000004 & PID4=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4

N_frames=6
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000000 --rotstab2 & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000001 --rotstab2 & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000002 --rotstab2 & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000003 --rotstab2 & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000004 --rotstab2 & PID4=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4

python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000000 --rotstab2 --saccade & PID0=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000001 --rotstab2 --saccade & PID1=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000002 --rotstab2 --saccade & PID2=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000003 --rotstab2 --saccade & PID3=$!
python3 rotation_stabilize.py --flapper --rotate --downsample $downsample --stride $stride --N_frames $N_frames --margins $margins --est_nf --metrics --sequence data/2025_08_21_collection_outside/sequence_000004 --rotstab2 --saccade & PID4=$!
wait $PID0 $PID1 $PID2 $PID3 $PID4
