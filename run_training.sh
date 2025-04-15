#!/bin/bash
clear
ITERATION=$1
echo $VIRTUAL_ENV
python rl_lstm_trader.py --timesteps=20000000 --discount_factor=0.99 --eval_frequency=1500 --checkpoint_frequency=100 --iteration="$ITERATION" --parallel_envs=96 > training_"$ITERATION".log

