#!/bin/bash

mkdir -p train_data
mkdir -p eval_data

for i in `seq -f "%03g" 0 500` ; do
   python3 create_pretraining_data.py \
      --input_file=cleanup_scripts/results/part-00$i-of-00500 \
      --output_file=train_data/part-00$i-of-00500 \
      --vocab_file=vocab.txt \
      --do_lower_case=True \
      --max_seq_length=512 \
      --max_predictions_per_seq=76 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=10
done

for i in `seq -f "%03g" 501 531`; do
   python3 create_pretraining_data.py \
      --input_file=cleanup_scripts/results/part-00$i-of-00500 \
      --output_file=eval_data/part-00$i-of-00500 \
      --vocab_file=vocab.txt \
      --do_lower_case=True \
      --max_seq_length=512 \
      --max_predictions_per_seq=76 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=10
done