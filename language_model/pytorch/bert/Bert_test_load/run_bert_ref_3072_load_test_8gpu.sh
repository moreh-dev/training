#!/bin/bash

# Vars with defaults
: "${NEXP:=5}"
: "${CLEAR_CACHES:=1}"

DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}
BENCHMARK=${BENCHMARK:-"BERT"}
BENCHMARK_NAME="BERT"
LOGDIR=${LOGDIR:-"./results/$BENCHMARK"}
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -p $(dirname "${LOGFILE_BASE}")

#BATCHSIZE=256
#GRADIENT_STEPS=8
NUM_SAMPLES_TO_CHECKPOINT=12500

## DL params

WARMUP_PROPORTION=0.0
PHASE=2
MAX_SAMPLES_TERMINATION=4500000
# EXTRA_PARAMS="--unpad"
EXTRA_PARAMS=""
# EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-3000000}
# EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-500000}
EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-1}
EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-1}
# EVAL_ITER_START_SAMPLES=${EVAL_ITER_START_SAMPLES:-40000}
# EVAL_ITER_SAMPLES=${EVAL_ITER_SAMPLES:-40000}
USE_DDP=${USE_DDP:-0}

#Batch 3072 reference
BATCHSIZE=3072
LR=0.002
EPSILON=0.000001
MAX_STEPS=1141
WARMUP_STEPS=100
START_WARMUP_STEP=0
OPT_LAMB_BETA_1=0.66
OPT_LAMB_BETA_2=0.998
WEIGHT_DECAY_RATE=0.01
GRADIENT_STEPS=96

#hparam set
BATCHSIZE=${BATCHSIZE}
LR=${LR}
EPSILON=${EPSILON}
MAX_STEPS=${MAX_STEPS}
WARMUP_STEPS=${WARMUP_STEPS}
START_WARMUP_STEP=${START_WARMUP_STEP}
OPT_LAMB_BETA_1=${OPT_LAMB_BETA_1}
OPT_LAMB_BETA_2=${OPT_LAMB_BETA_2}
WEIGHT_DECAY_RATE=${WEIGHT_DECAY_RATE}
GRADIENT_STEPS=${GRADIENT_STEPS}

RANDOM=12345

NUM_GPU=8
NUM_GPU=${NUM_GPU:-"1"}

LOG_FILE="${LOGFILE_BASE}_moreh_${BATCHSIZE}_${GRADIENT_STEPS}_load_test_${NUM_GPU}gpu_.log"

#LOG_FREQ=${LOG_FREQ:-100}
LOG_FREQ=10

echo "hparms: "
echo "{ batch: $BATCHSIZE"
echo "  learning_rate: $LR"
echo "  epsilon: $EPSILON"
echo "  training_steps: $MAX_STEPS"
echo "  warmup_steps: $WARMUP_STEPS"
echo "  start_warmup_step: $START_WARMUP_STEP"
echo "  lamb_beta_1: $OPT_LAMB_BETA_1"
echo "  lamb_beta_2: $OPT_LAMB_BETA_2"
echo "  weight_decay_rate: $WEIGHT_DECAY_RATE"
echo "  GRADIENT_STEPS: $GRADIENT_STEPS"
echo "}"

echo "num gpu: $NUM_GPU"


PHASE2="\
    --train_batch_size=$BATCHSIZE \
    --learning_rate=${LR:-4e-3} \
    --opt_lamb_beta_1=${OPT_LAMB_BETA_1:-0.9} \
    --opt_lamb_beta_2=${OPT_LAMB_BETA_2:-0.999} \
    --warmup_proportion=${WARMUP_PROPORTION:-0.0} \
    --warmup_steps=${WARMUP_STEPS:-0.0} \
    --start_warmup_step=${START_WARMUP_STEP:-0.0} \
    --max_steps=$MAX_STEPS \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --input_dir=/home/share/dataset/mlperf/new_bert/bert_data/2048_shards_uncompressed \
    --init_checkpoint=./cks/model.ckpt-28252.pt \
    "

PHASE=${PHASE:-2}

echo "***** Running Phase $PHASE *****"

MOREH_ENABLE_LAZY_MODE=1 MOREH_NUM_DEVICES=${NUM_GPU} python3 -u run_pretraining.py \
    $PHASE2 \
    --do_train \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=${TARGET_MLM_ACCURACY:-0.720} \
    --weight_decay_rate=${WEIGHT_DECAY_RATE:-0.01} \
    --max_samples_termination=${MAX_SAMPLES_TERMINATION} \
    --eval_iter_start_samples=${EVAL_ITER_START_SAMPLES} --eval_iter_samples=${EVAL_ITER_SAMPLES} \
    --eval_batch_size=16 --eval_dir=/home/share/dataset/mlperf/new_bert/bert_data/eval_set_uncompressed \
    --cache_eval_data \
    --output_dir=results \
    --fused_gelu_bias --dense_seq_output ${EXTRA_PARAMS} \
    --gradient_accumulation_steps=${GRADIENT_STEPS} \
    --log_freq=${LOG_FREQ} \
    --local_rank=-1 \
    --seed=$RANDOM \
    --bert_config_path=cks/bert_config.json \
    --skip_checkpoint |& tee $LOG_FILE
    #--fp16 \
    #--num_samples_per_checkpoint=${NUM_SAMPLES_TO_CHECKPOINT} \
    #--resume_from_checkpoint $@

