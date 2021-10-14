DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}
BENCHMARK=${BENCHMARK:-"single_stage_detector"}
BENCHMARK_NAME="ssd"
LOGDIR=${LOGDIR:-"./results/$BENCHMARK"}
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -p $(dirname "${LOGFILE_BASE}")

MOREH_ENABLE_DDP_MODE=1 MOREH_NUM_DEVICES=8 python train.py \
    --epochs 80 \
    --lr 0.001484375 \
    --warmup 5.25 \
    --warmup-factor 0 \
    --weight-decay 1.6e-4 \
    --no-save \
    --threshold=0.23 \
    --data /scratch/ssd \
    --batch-size 512 \
    --log-interval 100 \
    --lr-decay-schedule 44 55 \
    --val-interval 0 \
    --val-epochs 40 50 55 60 65 70 75 80 \
    --seed $RANDOM |& tee ${LOGFILE_BASE}_moreh_512_DDP_half_lr.log
