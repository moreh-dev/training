DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}
BENCHMARK=${BENCHMARK:-"unet_3d"}
BENCHMARK_NAME="unet3d"
LOGDIR=${LOGDIR:-"./results/$BENCHMARK"}
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
GPU=${GPU:-1}
mkdir -p $(dirname "${LOGFILE_BASE}")
touch "./results/unet3d.log"

MOREH_NUM_DEVICES=8 python main.py \
    --data_dir /home/share/dataset/mlperf/3d-unet/data \
    --epochs 10000 \
    --evaluate_every 20 \
    --start_eval_at 1000 \
    --quality_threshold 0.908 \
    --batch_size 16 \
    --optimizer sgd \
    --ga_steps 1 \
    --learning_rate 0.8 \
    --seed 0 \
    --lr_warmup_epochs 200 \
    -v \
    --log_dir $LOGDIR |& tee ${LOGFILE_BASE}_moreh.log