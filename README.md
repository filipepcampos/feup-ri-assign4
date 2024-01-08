Pre-requisites
- conda
- git & git-lfs

Obtain lfs files
```
git lfs pull
```

Setup conda
```
conda create --name ri-assign3 python=3.8
conda activate ri-assign3
```

Install gym-duckietown
```
cd gym-duckietown
pip install -e .
pip install pyglet==1.5.11  # otherwise the simulator crashes
```

If using Ubuntu 22.04
```
conda install -c conda-forge libstdcxx-ng
```

Install duckietown-world
```
cd duckietown-world
pip install -e .
```

Run movement script
```
cd gym-duckietown
./movement.py --env-name Duckietown --map-name ETH_small_intersect
```
