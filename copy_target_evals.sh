#!/bin/bash

#source ~/stablebaselines/bin/activate
#echo $VIRTUAL_ENV
ITERATION=$1
while read epoch; do
  echo "Epoch: $epoch. Iteration: $ITERATION"
  cp evals/iteration-"$ITERATION"/eval_"$ITERATION"_"$epoch".jsonl download/evals/iteration-"$ITERATION"/eval_"$ITERATION"_"$epoch".jsonl
  cp models/iteration-"$ITERATION"/checkpoints/rppo_trading_model_"$epoch"_steps.zip download/models/iteration-"$ITERATION"/
done <target_checkpoints.log