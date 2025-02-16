#############################
#241103 伪源域对齐
#
#############################

import torch
import torch.nn as nn
import network  # Assuming the definition of ResBase, feat_bootleneck, feat_classifier is in network.py
import argparse
import os.path as osp
from test_model import MultiHeadResNet50
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from helper.mixup_utils import progress_bar
import os
from utils import InfiniteDataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
os.environ['TORCH_HOME'] = './pypth'

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
        # wandb.log({'MISC/LR': param_group['lr']})
    return optimizer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

class ImageListWithLogits_SingleDomain(Dataset):
    def __init__(self, csv_file, target_id, transform=None):
        self.data = pd.read_csv(csv_file,header=None)
        self.data = self.data[self.data.iloc[:, 0] == target_id]
        self.data['absolute_idx'] = self.data.index
        self.data = self.data.reset_index(drop=True)
        self.transform = transform
        self.root_path='../../../dataset/city_wise_png_ext_jilin'
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 读取图像路径和标签
        domain = self.data.iloc[idx, 0]  # Domain 列
        img_path = self.data.iloc[idx, 1]  # Image Path 列
        actual_label = self.data.iloc[idx, 2]  # Actual Label 列
                
        # 处理 logits 列，去掉引号并转换为 float tensor
        logits_str = self.data.iloc[idx, 3].replace('“', '').replace('”', '')  # 去掉可能的引号
        logits = torch.tensor(list(map(float, logits_str.split(','))), dtype=torch.float32)  # Pseudo Label 列 (logits)
        
        # logits = torch.tensor(list(map(float, self.data.iloc[idx, 3].split(','))), dtype=torch.float32)  # Pseudo Label 列 (logits)
        full_img_path = os.path.join(self.root_path, img_path)

        # 图像预处理
        image = self.load_image(full_img_path)  # 实现图像加载的函数
        if self.transform:
            image = self.transform(image)

        absolute_idx = self.data.iloc[idx, 4]
        return image, actual_label, logits, domain, absolute_idx
    
    def load_image(self, img_path):
        # 实现图像加载的逻辑（可以使用 PIL、OpenCV 等）
        image = Image.open(img_path).convert('RGB')
        return image
        
    def get_high_confidence_data(self, top_percent=0.8):
        """
        获取按置信度排名前 top_percent 的高置信度样本（DataFrame 格式）

        参数:
        top_percent (float): 要返回的高置信度样本的比例（0 到 1 之间），默认值为 0.7

        返回:
        high_confidence_data (pd.DataFrame): 包含前 top_percent 的高置信度样本的数据，按置信度从高到低排序
        """
        confidence_data = []

        for idx in range(len(self.data)):
            # 读取 logits 并计算置信度
            logits_str = self.data.iloc[idx, 3].replace('“', '').replace('”', '')
            logits = torch.tensor(list(map(float, logits_str.split(','))), dtype=torch.float32)
            confidence_score = logits.softmax(dim=0).max().item()  # 计算置信度分数

            # 将置信度分数与样本数据一起存储
            row_data = self.data.iloc[idx].copy()
            row_data['confidence_score'] = confidence_score
            confidence_data.append(row_data)

        # 转换为 DataFrame 并按置信度从高到低排序
        confidence_df = pd.DataFrame(confidence_data)
        confidence_df = confidence_df.sort_values(by='confidence_score', ascending=False)

        # 确定置信度排名的百分比阈值
        threshold_index = int(len(confidence_df) * top_percent)
        high_confidence_data = confidence_df.head(threshold_index)

        return high_confidence_data

def data_load(args):
    def image_train(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
    target_train_loader_list = {}
    target_train_loader_list['train'] = []
    target_train_loader_list['test'] = []
    target_train_list = {}
    target_train_list['train'] = []
    target_train_list['test'] = []
    csv_file = f'{args.txt_folder}/{args.dset}/{args.csv_filename}'  # CSV 文件路径
    print("Loading data from: ", csv_file)

    for i in range(len(args.target_names)):
        target_train = ImageListWithLogits_SingleDomain(csv_file, i+1, transform=image_train()) # 实例化dataset

        target_train_list['train'].append(target_train)
        target_train_list['test'].append(target_train)

        if args.norepeat:#不repeat
            target_train_loader_list['train'].append(DataLoader(target_train_list['train'][i], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False))
        else:#repeat
            target_train_loader_list['train'].append(InfiniteDataLoader(DataLoader(target_train_list['train'][i], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)))
            target_train_loader_list['test'].append(DataLoader(target_train_list['test'][i], batch_size=args.batch_size, shuffle=False, num_workers=args.worker, drop_last=False))

    return target_train_loader_list, target_train_list

def soft_targets_scaling(soft_targets, temperature):
    # Apply softmax to the logits from the pre-trained model with temperature scaling
    return F.softmax(soft_targets / temperature, dim=1) * (1 / temperature**2)

def compute_kl_loss(logits, soft_targets, temperature):
    # Scale the soft targets and compute KL divergence with the logits
    scaled_soft_targets = soft_targets_scaling(soft_targets, temperature)
    log_probs = F.log_softmax(logits, dim=1)
    kl_loss = F.kl_div(log_probs, scaled_soft_targets, reduction='batchmean')
    return kl_loss


def train(args, all_loader, model, optimizer, temperature, epoch): 
    model.train()
    train_loss =0 
    correct = 0
    total = 0 
    pseudo_correct =0    

    num_iters=300
    for k in  range(num_iters):
        data_target_list=[]
        total_loss = 0
        #获取批次数据
        for i_target in range(len(args.target_names)):
            temp_data = next(all_loader[i_target])
            # print(temp_data[0].size(0))

            if temp_data[0].size(0) == 1:  # 假设 data[0] 是输入
                temp_data = next(all_loader[i_target])
            #每个域获取一个批次数据
            data_target_list.append(temp_data)#按tgt_dataset_list的顺序调用不同域的数据集loader
        # 对于每个域的批次数据
        for i in range(len(args.target_names)):
            # domain_id = i
            data_i, targets, pseudo_lbl, domains, _ = data_target_list[i]
            # if targets.size(0) != args.batch_size:
            #     continue
            if use_cuda:
                data_i, targets, pseudo_lbl = data_i.cuda(), targets.cuda(), pseudo_lbl.cuda()   

            model.backbone.update_id(i, i)
            model.backbone._enable_update()#更新mean std
            target_out_list = model(data_i, [i]) # 记录了这个域的batchsize的mean和std，获得
            # model.backbone.visualize_feature_map(save_dir='feature_maps', file_name=f'x_{i}_feature_map.png')
            target_out = target_out_list[0]
            try:
                probs = F.softmax(pseudo_lbl, dim=1)
                # 获取每个样本最大概率的索引作为标签
                pseudo_lbl_argmax = torch.argmax(probs, dim=1)
                ce_loss = criterion(target_out, pseudo_lbl_argmax)
                total_loss += ce_loss
            except:
                print(target_out.shape)
                print(pseudo_lbl.shape)
            
            # #style transfer
            # for ii in [0,1,2,3,4,5]:
            #     if ii == i:
            #         continue
            #     transfered_domain = ii

            #     model.backbone._enable_cross_norm()
            #     model.backbone.update_id(i, transfered_domain)

            #     target_out_transfer_list = model(data_i.clone(), [transfered_domain])

            #     model.backbone._disable_cross_norm()
            #     target_out_transfer = target_out_transfer_list[0]
            #     del target_out_transfer_list  # 删除未使用的变量
            #     torch.cuda.empty_cache()

            #     probs = F.softmax(pseudo_lbl, dim=1)
            #     # 获取每个样本最大概率的索引作为标签
            #     pseudo_lbl_argmax = torch.argmax(probs, dim=1)
            #     loss_CTS_transfered = criterion(target_out_transfer, pseudo_lbl_argmax)
                
            #     def get_adaptive_weight(epoch, loss, initial_weight=2, min_weight=0.4, total_epochs=20):
            #         """
            #         基于 epoch 动态调整权重，线性衰减。
                    
            #         参数:
            #         - epoch (int): 当前的训练轮数。
            #         - loss (torch.Tensor): 当前的损失值，用于计算动态权重时备用。
            #         - initial_weight (float): 初始权重值，默认为 0.8。
            #         - min_weight (float): 最低权重值，默认为 0.2。
            #         - total_epochs (int): 总的训练轮数。
                    
            #         返回:
            #         - weight (float): 自适应权重。
            #         """
            #         decay_rate = (initial_weight - min_weight) / total_epochs
            #         return max(initial_weight - epoch * decay_rate, min_weight)
            #     adaptive_weight = get_adaptive_weight(epoch, loss_CTS_transfered)
                # total_loss += adaptive_weight*loss_CTS_transfered #主要loss
   
            _, predicted = torch.max(target_out.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()
            pseudo_correct += predicted.eq(pseudo_lbl_argmax.data).cpu().sum().float()
        train_loss += total_loss.item()
         
        with torch.autograd.set_detect_anomaly(True):
            (total_loss).backward() 
        optimizer.step()
        optimizer.zero_grad()
        progress_bar(k, num_iters,
                    'Loss: %.3f | Acc: %.3f%% (%d/%d) | Pseudo Acc: %.3f%%'
                    % (total_loss/(k+1), 100.*correct/total, correct, total, 100.*pseudo_correct/total))

    return (total_loss/num_iters,  100.*correct/total) #平均loss        

def test(epoch, testloader, model):
    global best_acc
    global best_test_loss
    global early_stop
    global flag_bst
    model.eval()
    # netF.eval()
    # netB.eval()
    # netC.eval()
    test_loss = 0
    correct = 0
    total = 0
    domain_correct = [0] * 6
    domain_total = [0] * 6
    domain_pseudo_correct = [0] * 6
    pseudo_correct = 0
    with torch.no_grad():
        # data_target_list=[]
        # total_loss = 0
        #获取批次数据
        for i_target in range(len(args.target_names)):
            #每个域获取一个批次数据
            # data_target_list.append(next(all_loader[i_target]))#按tgt_dataset_list的顺序调用不同域的数据集loader

            for batch_idx, (inputs, targets, pseudo_lbl, domains, _) in enumerate(testloader[i_target]):
                # print(inputs)
                if use_cuda:
                    inputs, targets, pseudo_lbl, domains = inputs.cuda(), targets.cuda(), pseudo_lbl.cuda(), domains.cuda()
                # inputs, pseudo_lbl = Variable(inputs), Variable(pseudo_lbl)
                # outputs = netC(netB(netF(inputs)))
                logits_list=model(inputs, [i_target])

                for i, logits in enumerate(logits_list):
                # 选择与当前 head 对应的伪标签和输入
                    current_domain_mask = (domains-1 == i_target)  # 生成布尔掩码
                    current_pseudo_lbl = pseudo_lbl[current_domain_mask]  # 选择对应的伪标签
                
                # 如果 current_pseudo_lbl 为空，则跳过
                    if current_pseudo_lbl.numel() == 0:
                        continue

                # 同时选择对应的 logits，logits 和 inputs 的大小应该匹配
                    current_logits = logits[current_domain_mask]  # 选择对应的 logits
                    current_targets = targets[current_domain_mask]

                    current_pseudo_lbl = F.log_softmax(current_pseudo_lbl, dim=1)
                    current_pseudo_lbl = torch.argmax(current_pseudo_lbl, dim=1)
                    loss = criterion(current_logits, current_pseudo_lbl)

                    test_loss += loss.item()
                    _, predicted = torch.max(current_logits.data, 1)# 不用softmax
                    total += current_pseudo_lbl.size(0)
                    correct += predicted.eq(current_targets.data).cpu().sum()# 这里的test是使用真实的label
                    pseudo_correct += predicted.eq(current_pseudo_lbl.data).cpu().sum()
                    # 累计总样本数和正确预测数
                    domain_total[i_target] += current_pseudo_lbl.size(0)
                    domain_correct[i_target] += predicted.eq(current_targets.data).cpu().sum().item()
                    domain_pseudo_correct[i_target] += predicted.eq(current_pseudo_lbl.data).cpu().sum().item()
            # print(100.*pseudo_correct/total)
            progress_bar(i_target, len(args.target_names),'Loss: %.3f | Acc: %.3f%% Pseudo Acc: %.3f%% (%d/%d)'% (test_loss/(i_target+1), 100.*correct/total, 100.*pseudo_correct/total, correct, total))
        print( 100.*pseudo_correct/total)  
        print( 100.*correct/total)  
        for i in range(len(args.target_names)):
            if domain_total[i] > 0:
                domain_acc = 100. * domain_correct[i] / domain_total[i]
                domain_acc_pseudo = 100. * domain_pseudo_correct[i] / domain_total[i]
                print(f'Domain {i+1} Accuracy: {domain_acc:.2f}% ({domain_correct[i]}/{domain_total[i]})')
                print(f'Domain {i+1} Accuracy: {domain_acc_pseudo:.2f}% ({domain_pseudo_correct[i]}/{domain_total[i]})')
            else:
                print(f'Domain {i+1} has no samples')
        acc = 100.*correct/total

        avg_test_loss=test_loss/(i_target+1)
        if avg_test_loss < best_test_loss:
            early_stop=0
            best_test_loss = avg_test_loss
            flag_bst=True
            torch.save(model.state_dict(), f'./MTDA_weights/Stage2_step1_{args.dset}_{epoch}.pt')
            print(f'Saving best model with loss: {best_test_loss:.2f}')
        else:
            early_stop=early_stop+1
        if acc > best_acc:
            best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)

def pseudo_lbl_update(testloader, model, opt, epoch, device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')):
    model.eval()
    start_test=True
    with torch.no_grad():
        #获取批次数据
        for i_target in range(len(args.target_names)):
            for batch_idx, (inputs, targets, pseudo_lbl, domains, idx) in enumerate(testloader[i_target]):
                if use_cuda:
                    inputs, targets, pseudo_lbl, domains = inputs.cuda(), targets.cuda(), pseudo_lbl.cuda(), domains.cuda()
                logits_list = model(inputs, [i_target])
                logits_dom = logits_list[0]

                if start_test:
                    all_output = logits_dom.float().cpu()
                    all_idx = idx.int()
                    start_test = False 
                else:
                    all_output = torch.cat((all_output, logits_dom.float().cpu()), 0)
                    all_idx = torch.cat((all_idx, idx.int()), 0)
        logits_np = all_output.cpu().numpy()
        # print(logits_np.shape)
        # print(len(list(all_idx.numpy())))
        # print(len(logits_np))
        logits_str = [','.join(map(str, logit)) for logit in logits_np]

        

        csv_file = f'{args.txt_folder}/{args.dset}/{args.csv_filename}'  # CSV 文件路径
        csv_data = pd.read_csv(csv_file,header=None)
        for i in list(all_idx.numpy()):
            # try:
                if filter_single_logit_by_confidence(logits_np[i],epoch)==None:
                    continue
                else:           
                    csv_data.iloc[i, 3]=logits_str[i]
            # except:
            #     print(i)
            #     print(csv_data.index)
            #     print(len(csv_data))
            #     print(len(logits_str))

        args.csv_filename = f'temp_pl_{str(epoch)}.csv'
        csv_path=f'{args.txt_folder}/{args.dset}/{args.csv_filename}'
        csv_data.to_csv(csv_path, mode = 'w', header=False, index=False)
        all_loader, all_dset = data_load(args)

        
    return all_loader

def filter_single_logit_by_confidence(logit,epoch, confidence_threshold=0.35, initial_threshold=0.4):
    """
    判断单条 logit 是否高于置信度阈值，保留高于阈值的 logit 和对应标签。
    
    参数:
    - logit (torch.Tensor): 单条样本的 logit 输出，形状为 (num_classes,)。
    - confidence_threshold (float): 保留的置信度阈值，默认值为 0.4。
    
    返回:
    - high_confidence_logit (torch.Tensor or None): 如果置信度高于阈值，返回该 logit；否则返回 None。
    - high_confidence_label (int or None): 如果置信度高于阈值，返回该 logit 对应的标签；否则返回 None。
    """
    if isinstance(logit, np.ndarray):
        logit = torch.tensor(logit)
    # 计算该样本的预测概率

    # confidence_threshold = linear_decay_confidence_threshold(epoch, initial_threshold)
    prob = torch.softmax(logit, dim=0)
    
    # 获取该样本的最大置信度和对应的预测标签
    max_prob, predicted_label = torch.max(prob, dim=0)
    
    # 判断是否高于置信度阈值
    if max_prob >= confidence_threshold:
        return logit
    else:
        return None
    

def linear_decay_confidence_threshold(epoch, initial_threshold, min_threshold=0.1, total_epochs=10):
    """
    线性衰减置信度阈值。
    
    参数:
    - epoch (int): 当前训练的 epoch。
    - initial_threshold (float): 初始置信度阈值。
    - min_threshold (float): 最低置信度阈值。
    - total_epochs (int): 总训练 epoch 数。
    
    返回:
    - threshold (float): 动态调整后的置信度阈值。
    """
    decay_rate = (initial_threshold - min_threshold) / total_epochs
    return max(initial_threshold - epoch * decay_rate, min_threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage2Step1')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--net', default="resnet18", type=str, help='model type (default: ResNet18)')
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--seed', default=2025, type=int, help='random seed')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--epoch', default=4, type=int, help='total epochs to run')
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--csv_filename', default='guangzhou_source.csv', type=str, choices=['guangzhou_comp.csv', 'wuhan.csv', 'wuhan_comp.csv','changsha.csv','guangzhou_source.csv']) # class
    parser.add_argument('--txt_folder', default='csv_pseudo_labels', type=str)
    parser.add_argument('--dset', type=str, default='city_wise_png_jilin', choices=['city_wise_png', 'city_wise_png_jilin'])
    parser.add_argument("--ratio", default=1, type=float)
    parser.add_argument("--pseudo_update_interval", default=2, type=int)
    parser.add_argument('--isst', action='store_true', help='use standard augmentation (default: False)')

    args = parser.parse_args()

    early_stop = 0 
    args.classifier="bn"
    args.bottleneck=256
    args.layer="wn"
    args.class_num=15
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.seed != 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize the multi-head ResNet-50 model
   
    args.norepeat = False
    if args.dset == 'city_wise_png':
        args.target_names = ['hefei', 'hongkong', 'nanchang', 'nanjing', 'shanghai', 'wuhan']
        num_classes = 16
    elif args.dset == 'city_wise_png_jilin':
        args.target_names = ['guangzhou', 'changsha']
        num_classes = 15
    num_heads = len(args.target_names)
    temperature = 4.0 # Temperature scaling factor
    start_epoch=0
    use_cuda=True
    criterion = nn.CrossEntropyLoss()
    best_acc = 0 
    best_test_loss = 1000000
    flag_bst=False

    all_loader, all_dset = data_load(args)  
    
    model = MultiHeadResNet50(args,num_heads=num_heads)
    # checkpoint = torch.load('./MTDA_CN_weights_new/city_wise_png_mhmtda_class_guangzhou_5_0111_selftraining_800_01_st.pt')  # 加载 .pt wuhan class

    # model.load_state_dict(checkpoint)  # 将权重加载到模型中
    if use_cuda:
            model.cuda()
            cudnn.benchmark = True
            print('Using CUDA..')
    # 参数组，用于设置不同学习率
    param_group = []
    # 对每个部分的参数进行分类，设置不同的学习率
    for k, v in model.backbone.named_parameters():  # 假设 backbone 是 netF
        param_group += [{'params': v, 'lr': args.lr }]  # netF 的学习率设置为 args.lr * 0.1
    
    head_iter = 0
    for head in model.heads:
        # 提取 bottleneck 和 classifier
        bottleneck = head[0]  # head[0] 是 bottleneck
        classifier = head[1]  # head[1] 是 classifier

        # 分别为 bottleneck 和 classifier 设置学习率
        for name, param in bottleneck.named_parameters():
            # param.requires_grad = False
            param_group += [{'params': param, 'lr': args.lr}]  # 对 bottleneck 设置学习率
        for name, param in classifier.named_parameters():
            # param.requires_grad = False
            param_group += [{'params': param, 'lr': args.lr}]  # 对 classifier 设置学习率
        head_iter +=1

    optimizer = torch.optim.Adam(param_group, lr=args.lr) 
    optimizer = op_copy(optimizer)
    # Perform one training step
    print(args.isst)
    epoch_array=np.random.permutation(args.epoch)
    for epoch in range(start_epoch, args.epoch):
        decay_rate = 0.05  # 衰减速率

        if args.isst and epoch % args.pseudo_update_interval == 0 and epoch!=0 :
            all_loader = pseudo_lbl_update(all_loader['test'], model, args, epoch)
            print('Pseudo processing finished')
        print('\nEpoch: %d' % epoch)
        train_loss, train_acc = train(args, all_loader['train'],model, optimizer,temperature,epoch)
        optimizer = lr_scheduler(optimizer, iter_num=epoch, max_iter=args.epoch)
        flag_bst=False
        if epoch % args.interval == 0:
            print('\n Start Testing')
            test_loss, test_acc = test(epoch, all_loader['test'],model)
        # args.cn_loss_weight = args.cn_loss_weight * np.exp(-decay_rate * epoch)
        # print(f"第{epoch}epoch：cross_loss的weight：{args.cn_loss_weight}")



            
