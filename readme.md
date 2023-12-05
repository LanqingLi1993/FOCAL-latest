## Requirements
pytorch==1.6.0, mujoco-py==2.0.2.13
All the requirments are specified in requirements.txt

## Data Collection
Run the following script in a bash.
```
for seed in {1..40}
do
	python train_data_collection.py --env-type cheetah_vel --save-models 1 --log-tensorboard 1 --seed $seed
done
```
Data collection program uses SAC algorithm to collect offline data, hyperparameters can be accessed and modified at data\_collection\_config

## Mujoco Environment Installation
Ant and Cheetah environments need mujoco210. Refer to https://github.com/openai/mujoco-py for more details about mujoco210 installation.

Walker and Hopper environments need mujoco131. Download mjpro131 and mjkey from https://www.roboti.us/download.html, extract them into ```~/.mujoco/mjpro131```, and set ```export MUJOCO_PY_MJPRO_PATH=~/.mujoco/mjpro131```, then mujoco131 is ready to go.

Be aware that the environment variable of mujoco131 ```MUJOCO_PY_MJPRO_PATH``` is different from mujoco210 ```MUJOCO_PY_MUJOCO_PATH```. Please discern them to avoid potential errors.

## Run FOCAL
```
python train_offline_FOCAL.py --env-type cheeta_vel
```
Change the argument `--env-type` to choose a different environment:

Environment | Argument
------------|------------
Half-Cheetah-Vel | cheetah\_vel
Half-Cheetah-Dir | cheetah\_dir
Ant-Dir | ant\_dir

Hyperparameters of FOCAL can be modified at offline\_rl\_config
