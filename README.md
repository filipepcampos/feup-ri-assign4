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

Install duckietown-world
```
```