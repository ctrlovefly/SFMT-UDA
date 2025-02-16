#!/media/lscsc/nas/qianqian/anaconda3/envs/da/bin/python

# CUDA_VISIBLE_DEVICES=0 python image_source.py --gpu_id 0 --seed 2025 --da uda --output ckps/source/ --dset city_wise_png_jilin --max_epoch 50 --s 2

# CUDA_VISIBLE_DEVICES=1 python image_target.py --gpu_id 1 --seed 2025 --da uda --output ckps/target_nosimilar/ --dset city_wise_png_jilin --s 0 --cls_par 0.3 --ssl 0.6 --output_src ckps/source 

# CUDA_VISIBLE_DEVICES=1 python image_target.py --gpu_id 1 --seed 2025 --da uda --output ckps/target_nosimilar/ --dset city_wise_png_jilin --s 2 --cls_par 0.3 --ssl 0.6 --output_src ckps/source 

CUDA_VISIBLE_DEVICES=1 python image_target.py --gpu_id 1 --seed 2025 --da uda --output ckps/target/ --dset city_wise_png_jilin --s 0 --cls_par 0.3 --ssl 0.6 --output_src ckps/source 

CUDA_VISIBLE_DEVICES=1 python image_target.py --gpu_id 1 --seed 2025 --da uda --output ckps/target/ --dset city_wise_png_jilin --s 2 --cls_par 0.3 --ssl 0.6 --output_src ckps/source 

