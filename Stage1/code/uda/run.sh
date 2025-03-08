#!/media/lscsc/nas/qianqian/anaconda3/envs/da/bin/python
# Jilin-1 imagery
CUDA_VISIBLE_DEVICES=0 python image_source.py --gpu_id 0 --seed 2025 --da uda --output ckps/source/ --dset city_wise_png_jilin --data_folder {your dataset path}/city_wise_png_ext_jilin --max_epoch 50 --s 2

CUDA_VISIBLE_DEVICES=0 python image_target.py --gpu_id 0 --seed 2025 --da uda --output ckps/target/ --dset city_wise_png_jilin --data_folder {your dataset path}/city_wise_png_ext_jilin --s 0 --cls_par 0.3 --ssl 0.6 --output_src ckps/source 



# Google Earth imagery
# CUDA_VISIBLE_DEVICES=0 python image_source.py --gpu_id 0 --seed 2024 --da uda --output ckps/source/ --dset city_wise_png --data_folder ../../../../../dataset/city_wise_png_ext --max_epoch 50 --s 0

# CUDA_VISIBLE_DEVICES=0 python image_target.py --gpu_id 0 --seed 2024 --da uda --output ckps/target/ --dset city_wise_png --data_folder ../../../../../dataset/city_wise_png_ext --s 0 --cls_par 0.3 --ssl 0.6 --output_src ckps/source 

