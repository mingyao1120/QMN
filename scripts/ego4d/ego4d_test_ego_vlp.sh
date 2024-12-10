seed=32
lambda=0
ratio=0.80

margin=0.115
vtc_loss=0.3


CUDA_VISIBLE_DEVICES=1 python train.py  \
    --vtc-loss-cof $vtc_loss \
    --cc-loss-cof $cc_loss \
    --margin-dis $margin \
    --vote true \
    --seed $seed \
    --Lambda $lambda \
    --ratio $ratio \
    --config-path config/ego4d/main_egovlp.json \
    --log_dir LOG_DIR \
    --tag TAG 

