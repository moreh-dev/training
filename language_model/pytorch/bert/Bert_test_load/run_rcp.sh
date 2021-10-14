#!/bin/bash

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${LOGDIR:=./results}"
: "${CLEAR_CACHES:=1}"

## DL params
BATCHSIZE=3
LR=4.0e-4
GRADIENT_STEPS=1
#MAX_STEPS=${MAX_STEPS:-22000}
MAX_STEPS=${MAX_STEPS:-333333}
WARMUP_PROPORTION=0.0
PHASE=2
MAX_SAMPLES_TERMINATION=4500000
# EXTRA_PARAMS="--unpad"
EXTRA_PARAMS=""
# EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-3000000}
# EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-500000}
EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-100000}
EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-100000}
# EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-40000}
# EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-40000}
USE_DDP=${USE_DDP:-1}

LOG_FREQ=${LOG_FREQ:-100}

PHASE2="\
    --train_batch_size=$BATCHSIZE \
    --learning_rate=${LR:-0.00035} \
    --opt_lamb_beta_1=${OPT_LAMB_BETA_1:-0.9} \
    --opt_lamb_beta_2=${OPT_LAMB_BETA_2:-0.999} \
    --warmup_proportion=${WARMUP_PROPORTION:-0.0} \
    --warmup_steps=${WARMUP_STEPS:-0.0} \
    --start_warmup_step=${START_WARMUP_STEP:-0.0} \
    --max_steps=$MAX_STEPS \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --input_dir=/home/share/dataset/mlperf/bert/train_data \
    --init_checkpoint=cks/model.ckpt-28252.pt 
    "

PHASE=${PHASE:-2}

echo "***** Running Phase $PHASE *****"

# Run fixed number of training samples
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -u run_pretraining.py \
python -m torch.distributed.launch --nproc_per_node=1 run_pretraining.py \
    $PHASE2 \
    --do_train \
    --skip_checkpoint \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=${TARGET_MLM_ACCURACY:-0.712} \
    --weight_decay_rate=${WEIGHT_DECAY_RATE:-0.01} \
    --max_samples_termination=${MAX_SAMPLES_TERMINATION} \
    --eval_iter_start_samples=${EVAL_ITER_START_SAMPLES} --eval_iter_samples=${EVAL_ITER_SAMPLES} \
    --eval_batch_size=16 --eval_dir=/home/share/dataset/mlperf/bert/eval_data \
    --cache_eval_data \
    --output_dir=results \
    --fused_gelu_bias --dense_seq_output ${EXTRA_PARAMS} \
    --gradient_accumulation_steps=${GRADIENT_STEPS} \
    --log_freq=${LOG_FREQ} \
    --bert_config_path=cks/bert_config.json 


    #--local_rank=-1 \
