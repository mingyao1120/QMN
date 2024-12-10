seed=32
lambda=0
ratio=0.90
margin=0.115
vtc_loss=0.3

CUDA_VISIBLE_DEVICES=0 python train.py  \
    --vtc-loss-cof $vtc_loss \
    --cc-loss-cof $cc_loss \
    --margin-dis $margin \
    --vote true \
    --seed $seed \
    --Lambda $lambda \
    --ratio $ratio \
    --config-path config/tacos/main.json \
    --log_dir LOG_DIR \
    --tag TAG 

