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
