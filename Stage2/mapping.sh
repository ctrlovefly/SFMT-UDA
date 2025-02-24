#!/media/lscsc/nas/qianqian/anaconda3/envs/da/bin/python


CUDA_VISIBLE_DEVICES=1 python /media/lscsc/nas/qianqian/UDA/SFMTDA/Stage2/mapping.py --outpath classified_output_SFMT-UDA_cs.tif --method SFMT-UDA

CUDA_VISIBLE_DEVICES=1 python /media/lscsc/nas/qianqian/UDA/SFMTDA/Stage2/mapping.py --outpath classified_output_CoNMix_cs.tif --method CoNMix

CUDA_VISIBLE_DEVICES=1 python /media/lscsc/nas/qianqian/UDA/SFMTDA/Stage2/mapping.py --outpath classified_output_aggregation_cs.tif --method aggregation

CUDA_VISIBLE_DEVICES=1 python /media/lscsc/nas/qianqian/UDA/SFMTDA/Stage2/mapping.py --outpath classified_output_source-only_cs.tif --method source-only

