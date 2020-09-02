
python train_dyn_rnn.py \
    --num_epochs=1  \
    --enable_ce \
    --use_gradient_merge="true" \
    --batch_size=16 \
    --max_step=40 \
    --base_optimizer="adagrad" \
    --embedding_type="dense"


