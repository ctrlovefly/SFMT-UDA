import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import pdb
import matplotlib.pyplot as plt
import seaborn as sns

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    root_dir=args.data_folder
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9*dsize)
    # print(dsize, tr_size, dsize - tr_size)
    tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])


    full_paths_tr_txt = [os.path.join(root_dir, line.strip()) for line in tr_txt]
    full_paths_te_txt = [os.path.join(root_dir, line.strip()) for line in te_txt]
    full_paths_txt_test = [os.path.join(root_dir, line.strip()) for line in txt_test]


    
    dsets["source_tr"] = ImageList(full_paths_tr_txt, transform=image_train())
    # print(len(dsets["source_tr"]))
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(full_paths_te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs*3, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(full_paths_txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, 
        num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    # plot_confusion_matrix(torch.squeeze(predict).float(), all_label.to('cpu'), 16,'output_plot_'+args.name+'_mtx_src.png')


    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        # Calculate accuracy
        overall_accuracy = matrix.diagonal().sum() / matrix.sum() * 100
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print("Confusion Matrix:\n")
        print(matrix)
        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        print(f"aacc:{aacc},acc:{acc}")
        # matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        # matrix = matrix[np.unique(all_label).astype(int),:]
        # acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        # aacc = acc.mean()
        # aa = [str(np.round(i, 2)) for i in acc]
        # acc = ' '.join(aa)
        return accuracy*100, mean_ent
    else:
        return accuracy*100, mean_ent

def train_source(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            # print("***********")
            # print(len(dset_loaders["source_tr"]))
            # print(iter_source)
            # print("***********")
            # pdb.set_trace()
            inputs_source, labels_source = next(iter_source)
            # pdb.set_trace()
        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                if iter_num == max_iter:
                    acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
                else:
                    acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC

def test_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.dset=='VISDA-C':
        acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc) + '\n' + acc_list
    else:
        acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
        log_str = '\nTask: {}, Accuracy = {:.2f}%'.format(args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    

def plot_confusion_matrix(pseudo_labels, true_labels, num_classes, output_path):
    # 生成转移矩阵
    transfer_matrix = confusion_matrix(true_labels, pseudo_labels, labels=np.arange(num_classes))

    # 打印转移矩阵
    print("Transition Matrix (Confusion Matrix):")
    print(transfer_matrix)
    
    # 可视化转移矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(transfer_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes))
    plt.xlabel('Pseudo Labels')
    plt.ylabel('True Labels')
    plt.title('Pseudo vs True Labels Transition Matrix')

    # 保存图像到文件而不是显示
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # 关闭当前图形以释放资源
    plt.close()

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage1Step1')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='city_wise_png_jilin', choices=['city_wise_png', 'city_wise_png_jilin'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet34, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2025, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    
    parser.add_argument('--list_folder', type=str, default='../data/')
    parser.add_argument('--data_folder', type=str, default='../../../../../dataset/city_wise_png_ext_jilin')

    args = parser.parse_args()

    if args.dset == 'city_wise_png':
        names = ['guangzhou', 'hefei', 'hongkong', 'nanchang', 'nanjing', 'shanghai', 'wuhan']
        args.class_num = 16
    elif args.dset == 'city_wise_png_jilin':
        names = ['guangzhou_source', 'guangzhou', 'changsha']
        args.class_num = 15

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    folder = args.list_folder
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s].upper())
    args.name_src = names[args.s].upper()#这里有[0]
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)
    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()

    train_source(args)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test_2.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s].upper() + names[args.t].upper()
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        test_target(args)
        
    args.out_file.close()