seed=32
lambda=0.146
ratio=0.80

margin=0.115
vtc_loss=0.3
cc_loss=0

CUDA_VISIBLE_DEVICES=0 python train.py  \
    --vtc-loss-cof $vtc_loss \
    --cc-loss-cof $cc_loss \
    --margin-dis $margin \
    --vote true \
    --seed $seed \
    --Lambda $lambda \
    --ratio $ratio \
    --config-path config/charades/main_i3d.json \
    --log_dir LOG_DIR \
    --tag TAG 
