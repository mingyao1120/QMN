seed=24 # 24
lambda=0.133
ratio=0.95
base_margin=0.1
margin=1
vtc_loss=0.4
cc_loss=0

CUDA_VISIBLE_DEVICES=1 python train.py  \
    --vtc-loss-cof $vtc_loss \
    --cc-loss-cof $cc_loss \
    --margin-dis $margin \
    --vote true \
    --seed $seed \
    --Lambda $lambda \
    --ratio $ratio \
    --config-path config/activitynet/main.json \
    --log_dir LOG_DIR \
    --tag TAG 
