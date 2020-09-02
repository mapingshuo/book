
python train.py \
    --enable_ce \
    --batch_size=16 \
    --max_step=40 \
    --base_optimizer="sgd" \
    --use_gradient_merge="true"

