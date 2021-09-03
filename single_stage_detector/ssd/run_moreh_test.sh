DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}
BENCHMARK=${BENCHMARK:-"single_stage_detector"}
BENCHMARK_NAME="ssd"
LOGDIR=${LOGDIR:-"./results/$BENCHMARK"}
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -p $(dirname "${LOGFILE_BASE}")

MOREH_NUM_DEVICES=8 python train.py \
    --epochs 120 \
    --lr 0.00296875 \
    --warmup 5.25 \
    --warmup-factor 0 \
    --weight-decay 1.6e-4 \
    --no-save \
    --threshold=0.23 \
    --data /home/share/dataset/mlperf/ssd/ \
    --batch-size 512 \
    --log-interval 100 \
    --lr-decay-schedule 44 55 \
    --val-interval 0 \
    --val-epochs 120 \
    --seed 0 |& tee ${LOGFILE_BASE}_moreh_test.log
