DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}
BENCHMARK=${BENCHMARK:-"single_stage_detector"}
BENCHMARK_NAME="ssd"
LOGDIR=${LOGDIR:-"./results/$BENCHMARK"}
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
GPU=${GPU:-1}
mkdir -p $(dirname "${LOGFILE_BASE}")

python -m torch.distributed.launch --nproc_per_node=$GPU train.py \
    --epochs 120 \
    --lr 2.5e-3 \
    --warmup 2.619685 \
    --warmup-factor 0 \
    --weight-decay 5e-4 \
    --no-save \
    --threshold=0.23 \
    --data /home/share/dataset/mlperf/ssd/ \
    --batch-size 128 \
    --log-interval 100 \
    --lr-decay-schedule 40 50 \
    --val-interval 0 \
    --val-epochs 40 50 55 60 65 70 75 80 \
    --seed $RANDOM |& tee ${LOGFILE_BASE}_nvidia.log