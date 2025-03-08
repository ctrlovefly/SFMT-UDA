#!/media/lscsc/nas/qianqian/anaconda3/envs/da/bin/python
# Jilin-1
CUDA_VISIBLE_DEVICES=0 python3 bridge.py --s 0 --dset city_wise_png_jilin --data_folder ../../../dataset/city_wise_png_ext_jilin --output_suffix _Jilin --net resnet50 --batch_size 64

# Google Earth imagery
# CUDA_VISIBLE_DEVICES=0 python3 bridge.py --s 0 --dset city_wise_png --data_folder ../../../dataset/city_wise_png_ext --output_suffix _Google --net resnet50 --batch_size 64 
