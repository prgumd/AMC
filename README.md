# Artifical Microsaccade Compensation: Stable Vision for an Ornithopter

[Webpage](https://prg.cs.umd.edu/AMC), [arXiv](https://arxiv.org/abs/2512.03995)

## Contents
* `rotation_stabilize.py`: Rotation stabilization algorithm
* `vme_research/algorithms/patch_track.py`: Inverse compositional LK on SO(3) with updates in so(3) and line search
* `base_exp.sh`: Process raw data from motion capture sequences for `analysis.py`
* `out_lab_exp.sh`: Process raw data from non-motion capture sequences for `analysis.py`
* `analysis.py`: Compute final metrics and generate tables (comment out top lines to choose motion capture or non-motion capture sequences)
* `generate_videos_vicon.sh`: Generate videos (excluding Adobe) (motion capture sequences) 
* `generate_videos.sh`: Generate videos (excluding Adobe) (non-motion capture sequences) 

## Data
The raw record data is available [here](https://drive.google.com/file/d/1XEgPUvLfYYPCyPr9qqqxEewofeAc-RFO/view?usp=sharing). Extract it into the root of this repository

## Dependencies
* `vme_research`: A minimal version of an internal library used by the group for robotics research. To install run: `pip install -e ./vme_research` from the root of this repository.
* numpy, scipy, matplotlib, jax (cpu)
* OpenCV
