# rlblocks
Simple RL library

```
# Humanoid-v4
python train/train_gym_env.py --env Humanoid-v4 --replay-buffer-ep-num 5000 --episode-len 1000 --replay-buffer-pre-fill 100 --train-epochs 100000 --train-steps-per-epoch 25 --exploration-prob 0.8 --lambdaa 1 --visualize-every 10 --visualize 1 --device cuda:1 --tb ../logs/humanoid-v4/exp3
python train/train_gym_env.py --env Humanoid-v4 --replay-buffer-ep-num 5000 --episode-len 1000 --replay-buffer-pre-fill 500 --train-epochs 100000 --train-steps-per-epoch 25 --exploration-prob 0.8 --exploration-std 0.1 --lambdaa 1 --reward-scale 0.01 --visualize-every 10 --visualize 1 --device cuda:0 --load-state-norm ../norm/humanoid-v4.pkl --tb ../logs/humanoid-v4/exp17 --save-dir ../logs/humanoid-v4/exp17 --save-episodes-dir ../logs/humanoid-v4/exp17

# 'BipedalWalker-v3' ep len 1600
python train/train_gym_env.py --env BipedalWalker-v3 --replay-buffer-ep-num 5000 --episode-len 800 --replay-buffer-pre-fill 100 --train-epochs 100000 --train-steps-per-epoch 100 --exploration-prob 0.8 --lambdaa 1 --visualize-every 10 --visualize 1 --device cuda:0 --tb ../logs/bipedal-walker-v3/exp6

# HalfCheetah-v4
python train/train_gym_env.py --env HalfCheetah-v4 --replay-buffer-ep-num 5000 --episode-len 1000 --replay-buffer-pre-fill 20 --train-epochs 100000 --train-steps-per-epoch 250 --exploration-prob 0.8 --lambdaa 1 --visualize-every 10 --visualize 1 --device cuda:0 --tb ../logs/half-cheetah-v4/exp19

# InvertedPendulum-v4
python train/train_gym_env.py --env InvertedPendulum-v4 --replay-buffer-ep-num 5000 --episode-len 400 --replay-buffer-pre-fill 100 --train-epochs 100000 --train-steps-per-epoch 50 --exploration-prob 0.8 --lambdaa 8 --visualize-every 10 --visualize 1 --device cuda:0 --tb ../logs/inv-pendulum-v2/exp26
```
