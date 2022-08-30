# rlblocks
Simple RL library

```
# 'BipedalWalker-v3' ep len 1600
python train/train_gym_env.py --env BipedalWalker-v3 --replay-buffer-ep-num 1000 --episode-len 800 --replay-buffer-pre-fill 100 --train-epochs 10000 --train-steps-per-epoch 25 --exploration-prob 0.8 --visualize-every 10 --visualize 1 --device cuda:0 --tb ../logs/bipedal-walker/exp6
```
