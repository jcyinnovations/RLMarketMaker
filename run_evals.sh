#!/bin/bash

#source ~/stablebaselines/bin/activate
#echo $VIRTUAL_ENV
ITERATION=$1
for FILE in ./models/iteration-"$ITERATION"/checkpoints/*.zip; 
  do echo -e $FILE;
  if [[ "$FILE" =~ .*rppo_trading_model_([0-9]*)_steps ]]; then
    epoch=${BASH_REMATCH[1]}
    echo "Epoch: $epoch. Iteration: $ITERATION"
    uv run python rl_lstm_eval.py --iteration="$ITERATION" --checkpoint="$epoch" >evals/iteration-"$ITERATION"/eval_"$ITERATION"_"$epoch".jsonl
  fi
done

