# DuckieBot Follower

Authored by: António Ribeiro, Filipe Campos and Francisco Cerqueira
Project for the Intelligent Robotics course of M.EIC at FEUP, 2023/2024

## Structure

The project is divided into 2 sections:
- `gym-duckietown` - Contains the gym environment for the duckietown world. Inside this directory, the main files are located in lib.
- `duckietown_world` - Contains the duckietown world simulator, we only modify the maps and add ArUco markers to the back of DuckieBots

## Requirements

Pre-requisites
- conda
- git & git-lfs


If the duckietown_world is missing (due to size constraints in moodle), just reset git using:
```
git reset --hard origin/main 
```


Obtain lfs files
```
git lfs pull
```

```
conda create --name ri-assign4 python=3.8
conda activate ri-assign4

git submodule init
git submodule update
pip install -r gym-duckietown/feup-ri-assign4-model/requirements.txt 

cd gym_duckietown
pip install -e .

cd ..
cd duckietown_world
pip install -e .
```

## Running 

```
cd gym_duckietown
python3 follower.py --env-name Duckietown --map-name ETH_small_intersect --method {yolo or aruco}
```