#!/media/lscsc/nas/qianqian/anaconda3/envs/da/bin/python

CUDA_VISIBLE_DEVICES=1 python3 bridge.py --s 0 --dset city_wise_png_jilin --net resnet50 --batch_size 64