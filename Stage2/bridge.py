#############################
#241023 multihead KD
#
#############################
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from helper.data_list import ImageList, ImageList_idx
from torch.utils.data import DataLoader
# from bridge_MTDA import data_load,image_train
import network, loss
import numpy as np 
import random 
import os.path as osp
import os
import pandas as pd
from torchvision import transforms

def new_softmax(logits, temperature=1):

  '''
    Annealing the temperature of the softmax.
  '''
  logits = logits/temperature
  return np.exp(logits)/np.sum(np.exp(logits))

def soft_labels(loader, netF, netB, netC, flag=False):
    start_test = True
    print("Finding Accuracy and Pseudo Label")
    with torch.no_grad():
        iter_test = iter(loader)
        all_idx = []
        for i in tqdm(range(len(loader))):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            idx = data[2]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
                all_idx = idx.int()
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_idx = torch.cat((all_idx, idx.int()), 0)
    return all_output,all_idx.numpy()

def image_train(resize_size=256, crop_size=224, alexnet=False):
#   if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
#   if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
    # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    root_dir=args.data_folder
    
    txt_src = open(args.s_dset_path).readlines()#读取数据路径list
    txt_test = open(args.test_dset_path).readlines()

    full_paths_txt_src = [os.path.join(root_dir, line.strip()) for line in txt_src]
    full_paths_txt_test = [os.path.join(root_dir, line.strip()) for line in txt_test]#组织好每张图的路径

    if args.trte == "val":
        dsize = len(full_paths_txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(full_paths_txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(full_paths_txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(full_paths_txt_src, [tr_size, dsize - tr_size])
        tr_txt = full_paths_txt_src

    dsets["source_tr"] = ImageList_idx(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList_idx(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx(full_paths_txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def test_target(args):
    # 获取数据
    dset_loaders = data_load(args) # text loaders 
    # 定义teachers 模型
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  
    elif args.net[0:4] == 'deit':
        if args.net == 'deit_s':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        elif args.net == 'deit_b':
            netF = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).cuda()
        netF.in_features = 1000
    else:
        netF = network.ViT().cuda()
        
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/target_F_par_0.3_ssl_0.6.pt' # 全部是除了s之外的targe的模型參數
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/target_B_par_0.3_ssl_0.6.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/target_C_par_0.3_ssl_0.6.pt'   
    netC.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    print("Models loaded")
    # Teachers' logits extractions
    logits,idx = soft_labels(dset_loaders['test'], netF, netB, netC, False)
    print(type(logits))
    print(logits.shape)
    # log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    return logits,idx


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage2preparepseudolabels')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dset', type=str, default='city_wise_png_jilin', choices=['city_wise_png', 'city_wise_png_jilin'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2025, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='STDA_weights')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--bsp', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)
    parser.add_argument('--cls_par', type=float, default=0.2)

    parser.add_argument('--list_folder', type=str, default='./data/')
    parser.add_argument('--data_folder', type=str, default='../../../dataset/city_wise_png_ext_jilin')


    args = parser.parse_args()
    # 主要改这里，其他基本不用改
    args.source_name = 'guangzhou_source'
    if args.dset == 'city_wise_png':
        if args.source_name == 'wuhan':
            names = [ 'wuhan', 'guangzhou', 'hefei', 'hongkong', 'nanchang','nanjing', 'shanghai']
            args.class_num = 16
        else:
            names = ['guangzhou', 'hefei', 'hongkong', 'nanchang', 'nanjing', 'shanghai', 'wuhan']
            args.class_num = 16
    elif args.dset == 'city_wise_png_jilin':
        if args.source_name == 'guangzhou':
            names = ['guangzhou', 'guangzhou_source', 'changsha']
            args.class_num = 15
        elif args.source_name == 'changsha':
            names = ['changsha', 'guangzhou_source', 'guangzhou']
            args.class_num = 15
        else:
            names = ['guangzhou_source', 'guangzhou', 'changsha']
            args.class_num = 15

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = args.list_folder 

    print(print_args(args))

    args.name_src = names[args.s][0].upper()
    args.save_dir = osp.join('csv_pseudo_labels/', args.dset) #保存位置
    if not osp.exists(args.save_dir):
        os.system('mkdir -p ' + args.save_dir)

    for i in range(len(names)):# 对每一对源域进行生成
        if i == args.s:
            continue
        args.t = i

        args.name = names[args.s].upper() + names[args.t].upper() #修改

        args.output_dir_src = osp.join(args.output, 'STDA', args.dset+'_nosimilar', args.name.upper()) #預訓練模型

        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt' #數據
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        logits,idx = test_target(args)
        txt_test = open(args.test_dset_path).readlines()
        img_path = []
        label = []
        for i in list(idx):
            image_path, lbl = txt_test[i].split(' ')
            img_path.append(image_path)
            label.append(int(lbl))

        # 将 logits 转换为 NumPy 数组
        logits_np = logits.cpu().numpy()
        # 将每一行的 logits 转换为字符串形式（以逗号分隔）
        logits_str = [','.join(map(str, logit)) for logit in logits_np]

        dict = {'Domain': args.t, 'Image Path': img_path, 'Actual Label': label, 'Pseudo Label': logits_str} # 目标domain的都存储在args.s名字的csv当中了

        df = pd.DataFrame(dict)
        df.to_csv(osp.join(args.save_dir, names[args.s]+'_nosimilar.csv'), mode = 'a', header=False, index=False)
        # df.to_csv(osp.join(args.save_dir, names[args.s]+'.csv'), mode = 'a', header=False, index=False)



# hinton KD training 使用平均的soft labels

# init hydra parameters

# hydra KD training 使用一一对应的soft labels

