DATESTAMP=${DATESTAMP:-`date +'%y%m%d%H%M%S%N'`}
BENCHMARK=${BENCHMARK:-"single_stage_detector"}
BENCHMARK_NAME="ssd"
LOGDIR=${LOGDIR:-"./results/$BENCHMARK"}
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
GPU=${GPU:-4}
mkdir -p $(dirname "${LOGFILE_BASE}")

python -m torch.distributed.launch --nproc_per_node=$GPU train.py \
    --epochs 120 \
    --lr 0.00296875 \
    --warmup 5.25 \
    --warmup-factor 0 \
    --weight-decay 1.6e-4 \
    --no-save \
    --threshold=0.23 \
    --data /home/share/dataset/mlperf/ssd/ \
    --batch-size 128 \
    --log-interval 100 \
    --lr-decay-schedule 44 55 \
    --val-interval 0 \
    --val-epochs 40 50 55 60 65 70 75 80 \
    --fp16 True \
    --seed $RANDOM |& tee ${LOGFILE_BASE}_nvidia_512_fp16.log