python3 src/main.py \
    --dataset-path ./dataset \
    --layers 8 \
    --dim 320 \
    --heads 4 \
    --emb-dropout 0.0 \
    --msa-dropout 0.0 \
    --mlp-dropout 0.0 \
    --sync-lambda 0.0 \
    --droppath 0.2 \
    --init-seed 0 \
    --mixup-seed 0 \
    --dropout-seed 0 \
    --input-features 1434 \
    --input-length 29 \
    --learning-rate 1e-3 \
    --weight-decay 0.05 \
    --adam-b1 0.9 \
    --adam-b2 0.999 \
    --adam-eps 1e-8 \
    --max-norm 1.0 \
    --train-batch-size 1024 \
    --valid-batch-size 128 \
    --warmup-epochs 10 \
    --training-epochs 1000 \
    --log-interval 50 \
    --name $(basename $0 | cut -f 1 -d '.') \
    --ipaddr $(curl -s ifconfig.me) \
    --hostname $(hostname) \
    $@
