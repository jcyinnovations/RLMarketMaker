#!/bin/bash

while read p; do
  echo "$p.log"
done <target_checkpoints.log
