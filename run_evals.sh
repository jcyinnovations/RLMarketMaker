#!/bin/bash

source ~/stablebaselines/bin/activate
echo $VIRTUAL_ENV

for FILE in ./models/iteration-18/checkpoints/*.zip; 
  do echo -e $FILE;
  if [[ "$FILE" =~ .*rppo_trading_model_([0-9]*)_steps ]]; then
    epoch=${BASH_REMATCH[1]}
    echo "Epoch: $epoch"
    python rl_lstm_eval.py --iteration=18 --checkpoint="$epoch" >eval_18_"$epoch".jsonl
  fi
done

