#!/media/lscsc/nas/qianqian/anaconda3/envs/da/bin/python

# initialization

# CUDA_VISIBLE_DEVICES=1 python -u MH_MTDA_initialization.py --epoch 15 

# self-training

CUDA_VISIBLE_DEVICES=0 python -u MH_MTDA_self_training.py --pseudo_update_interval 2 --epoch 2 --isst




