#!/media/lscsc/nas/qianqian/anaconda3/envs/da/bin/python
#Jilin-1 imagery
# initialization
CUDA_VISIBLE_DEVICES=0 python -u MH_MTDA_initialization.py --csv_filename guangzhou_Jilin.csv --data_folder {your dataset path}/city_wise_png_ext --epoch 15 

# self-training
CUDA_VISIBLE_DEVICES=0 python -u MH_MTDA_self_training.py --csv_filename guangzhou_Jilin.csv --data_folder {your dataset path}/city_wise_png_ext --epoch 2 --ckpt_path ./MTDA_weights/{your .pt from initialization}.pt

# Google Earth imagery
# initialization
# CUDA_VISIBLE_DEVICES=0 python -u MH_MTDA_initialization.py --csv_filename guangzhou_Google.csv --dset city_wise_png --data_folder ../../../dataset/city_wise_png_ext --class_num 16 --epoch 15 --net resnet18

# self-training
# CUDA_VISIBLE_DEVICES=0 python -u MH_MTDA_self_training.py --csv_filename guangzhou_Google.csv --dset city_wise_png --data_folder ../../../dataset/city_wise_png_ext --class_num 16 --pseudo_update_interval 3 --isst --epoch 4 --ckpt_path ./MTDA_weights/Stage2_step1_city_wise_png_13.pt --net resnet18 --backbonelr 0.2





