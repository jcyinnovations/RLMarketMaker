#!/bin/bash

#source ~/stablebaselines/bin/activate
#echo $VIRTUAL_ENV
ITERATION=$1
while read epoch; do
  echo "Epoch: $epoch. Iteration: $ITERATION"
  uv run python rl_lstm_eval.py --iteration="$ITERATION" --checkpoint="$epoch" >evals/iteration-"$ITERATION"/eval_"$ITERATION"_"$epoch".jsonl
done <target_checkpoints.log