#!/bin/bash
clear
ITERATION=$1
PARALLEL_ENVS=$2
echo $VIRTUAL_ENV
python rl_lstm_trader.py --timesteps=20000000 --discount_factor=0.99 --eval_frequency=750 --checkpoint_frequency=50 --iteration="$ITERATION" --parallel_envs="$PARALLEL_ENVS" > training_"$ITERATION".log

