
python train_dyn_rnn.py \
    --num_epochs=1  \
    --enable_ce \
    --batch_size=64 \
    --max_step=9 \
    --base_optimizer="adagrad" \
    --embedding_type="dense"


