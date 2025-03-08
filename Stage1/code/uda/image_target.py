import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import rotation
import matplotlib.pyplot as plt
import seaborn as sns

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

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
    dsets = {}
    dset_loaders = {}

    root_dir=args.data_folder

    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    full_paths_tr_txt = [os.path.join(root_dir, line.strip()) for line in txt_tar]
    full_paths_te_txt = [os.path.join(root_dir, line.strip()) for line in txt_test]

    dsets["target"] = ImageList_idx(full_paths_tr_txt, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dsets["target_te"] = ImageList_idx(full_paths_tr_txt, transform=image_test())
    dset_loaders["target_te"] = DataLoader(dsets["target_te"], batch_size=train_bs*3, shuffle=False, 
        num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList_idx(full_paths_te_txt, transform=image_test())
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
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

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
        return accuracy*100, mean_ent
    else:
        return accuracy*100, mean_ent

def cal_acc_rot(loader, netF, netB, netR):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            r_labels = np.random.randint(0, 4, len(inputs))
            r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)
            r_labels = torch.from_numpy(r_labels)
            r_inputs = r_inputs.cuda()
            
            f_outputs = netB(netF(inputs))
            f_r_outputs = netB(netF(r_inputs))

            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))
            if start_test:
                all_output = r_outputs.float().cpu()
                all_label = r_labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, r_labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    return accuracy*100

def train_target_rot(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netR = network.feat_classifier(type='linear', class_num=4, bottleneck_dim=2*args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    netF.eval()
    for k, v in netF.named_parameters():
        v.requires_grad = False
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    netB.eval()
    for k, v in netB.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netR.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*1}]
    netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    rot_acc = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:         
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)
        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        r_labels_target = np.random.randint(0, 4, len(inputs_test))
        r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
        r_labels_target = torch.from_numpy(r_labels_target).cuda()
        r_inputs_target = r_inputs_target.cuda()
        
        f_outputs = netB(netF(inputs_test))
        f_r_outputs = netB(netF(r_inputs_target))
        r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

        rotation_loss = nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)
        rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netR.eval()
            acc_rot = cal_acc_rot(dset_loaders['target'], netF, netB, netR)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_rot)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netR.train()

            if rot_acc < acc_rot:
                rot_acc = acc_rot
                best_netR = netR.state_dict()

    log_str = 'Best Accuracy = {:.2f}%'.format(rot_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return best_netR, rot_acc

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    
    if not args.ssl == 0:
        netR = network.feat_classifier(type='linear', class_num=4, bottleneck_dim=2*args.bottleneck).cuda()
        netR_dict, acc_rot = train_target_rot(args)
        netR.load_state_dict(netR_dict)
    
    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    if not args.ssl == 0:
        for k, v in netR.named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        netR.train()

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])#默认是15
    interval_iter = max_iter // args.interval #默认是15
    iter_num = 0

    #####记录true label###
    all_labels = []
    txt_tar_temp = open(args.t_dset_path).readlines()
    
    root_dir=args.data_folder
    full_paths_txt_tar_temp = [os.path.join(root_dir, line.strip()) for line in txt_tar_temp]

    dsets_temp= ImageList_idx(full_paths_txt_tar_temp)

    # 遍历 DataLoader 中的所有批次
    for i in range(len(dsets_temp)):
        _, label,_ = dsets_temp[i]  # 获取每个样本的标签
        all_labels.append(label)
    # iter_test = iter(dset_loaders["test"])
    # for i in range(len(dset_loaders["test"])):
    #     data = next(iter_test)
    #     inputs = data[0]
    #     labels = data[1]
    #     all_labels.append(labels)

    # for inputs, labels_te, _ in dset_loaders["target_te"]:
    #     # 将标签添加到列表中
    #     all_labels.append(labels_te)

    # 将所有批次的标签拼接到一起
    all_labels = torch.tensor(all_labels).cuda()

    ####################
    #####声明存储所有的误分tensor的list###
    list_epoch=[]
    num_classes_lcz=[i for i in range(16)]
    # print(len(num_classes_lcz))
    # pdb.set_trace()
    misclass_rate_per_class = np.zeros((len(num_classes_lcz), args.max_epoch+1))
    # ###################

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)
        # print('*****')
        # print(tar_idx)
        # print(ttt)
        # print(inputs_test)
        # print('*****')
        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['target_te'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        if args.cls_par > 0:
            pred = mem_label[tar_idx]

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)


        ###方法2：直接比较mem_label###
        if iter_num % interval_iter == 0:
            current_epoch = iter_num // len(dset_loaders["target"])
            label_bool = all_labels != mem_label
            for cls in range(len(num_classes_lcz)):
                cls_indices = np.where(all_labels.to('cpu') == cls)[0]
                misclass_rate_per_class[cls, current_epoch] = np.mean(label_bool[cls_indices].float().to('cpu').numpy())

        ###记录伪代码变化代码###
        # misclassified = np.zeros((len(dset_loaders['target_te'])*args.batch_size, args.max_epoch))
        # for i in range(len(outputs_test)):
            # if pseudo_labels[epoch][i] != true_labels[i]:
            #     misclassified[i, epoch] = 1
        ######################

        if args.cls_par > 0:
            # print(outputs_test)
            # print('$$$$$$')
            # print(pred)
            # pdb.set_trace()
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        classifier_loss.backward()

        if not args.ssl == 0:
            r_labels_target = np.random.randint(0, 4, len(inputs_test))
            r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
            r_labels_target = torch.from_numpy(r_labels_target).cuda()
            r_inputs_target = r_inputs_target.cuda()

            f_outputs = netB(netF(inputs_test))
            f_outputs = f_outputs.detach()
            f_r_outputs = netB(netF(r_inputs_target))
            r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

            rotation_loss = args.ssl * nn.CrossEntropyLoss()(r_outputs_target, r_labels_target)   
            rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                if iter_num == max_iter:
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
                else:
                    acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                    log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
            
                    

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()

    ####制图误分率####
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(10, 6)) 
    for cls in range(len(num_classes_lcz)):
        plt.plot(range(args.max_epoch+1), misclass_rate_per_class[cls], 
             label=f'Class {cls}', color=cmap(cls / len(num_classes_lcz)), 
             linestyle=linestyles[cls % len(linestyles)], 
             marker=markers[cls % len(markers)])  # 动态分配线型和标记
        # plt.plot(range(args.max_epoch+1), misclass_rate_per_class[cls], label=f'Class {cls}')#横轴的值，纵轴的值，以及曲线的图例

    plt.xlabel('Epoch')
    plt.ylabel('Misclassification Rate')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('output_plot_vanilla_comp_'+args.name+'.png', dpi=300, bbox_inches='tight')
    plt.close()
    # plot_confusion_matrix(mem_label.to('cpu'), all_labels.to('cpu'), len(num_classes_lcz),'output_plot_'+args.name+'_mtx_simi04.png')
    plot_confusion_matrix(mem_label.to('cpu'), all_labels.to('cpu'), len(num_classes_lcz),'output_plot_'+args.name+'_mtx_vanilla.png')

    
    
    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


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



def calculate_weighted_accuracy(predict, all_label, similarity_matrix):
    # 预测和真实标签的大小必须一致
    assert predict.shape == all_label.shape, "预测标签和真实标签的维度不匹配"
    
    num_classes = similarity_matrix.shape[0]
    print(f'num_classes:{num_classes}')

    # 将预测和真实标签转换为 numpy 数组
    predict_np = predict
    all_label_np = all_label

    # 初始化混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes))

    # 填充混淆矩阵
    for t, p in zip(all_label_np, predict_np):
        confusion_matrix[int(t)][int(p)] += 1

    # 计算加权的混淆矩阵元素的和
    total_samples = np.sum(confusion_matrix)
    elementwise_weighted_sum = np.sum(np.multiply(similarity_matrix, confusion_matrix))

    # 计算加权准确率
    weighted_accuracy = elementwise_weighted_sum / total_samples
    return weighted_accuracy


# 测试置信度
def calculate_accuracy_per_confidence_bin(all_output, all_label, num_bins=10):
    """
    Calculate pseudo-label accuracy per confidence bin.

    Parameters:
    - all_output: ndarray of shape (N, C), softmax probabilities for N samples and C classes.
    - all_label: ndarray of shape (N,), ground truth labels for N samples.
    - num_bins: int, number of bins to divide the confidence range [0, 1].

    Returns:
    - bin_accuracies: list of accuracies for each bin.
    - bin_ranges: list of bin range tuples [(start, end), ...].
    """

    if isinstance(all_output, torch.Tensor):
        all_output = all_output.cpu().numpy()  # Convert tensor to numpy array

    if isinstance(all_label, torch.Tensor):
        all_label = all_label.cpu().numpy()  # Convert tensor to numpy array
    # Step 1: Compute pseudo-labels and maximum confidence
    pseudo_labels = np.argmax(all_output, axis=1)
    max_confidences = np.max(all_output, axis=1)

    # Step 2: Define bins
    bins = np.linspace(0, 1, num_bins + 1)  # Define bin edges
    bin_indices = np.digitize(max_confidences, bins)  # Assign each confidence to a bin

    # Step 3: Calculate accuracy per bin
    bin_accuracies = []
    bin_ranges = []
    for i in range(1, len(bins)):
        indices_in_bin = np.where(bin_indices == i)[0]  # Samples in the current bin
        bin_range = (bins[i - 1], bins[i])
        bin_ranges.append(bin_range)

        if len(indices_in_bin) > 0:
            correct_count = np.sum(pseudo_labels[indices_in_bin] == all_label[indices_in_bin])
            accuracy = correct_count / len(indices_in_bin)
            bin_accuracies.append(accuracy)
        else:
            bin_accuracies.append(None)  # No samples in this bin

    return bin_accuracies, bin_ranges




def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)
    similarity_matrix=np.array([[1,0.92,0.83,0.67,0.58,0.5,0.58,0.75,0.42,0.75,0.33,0.25,0.08,0,0.25,0,0],
    [0.92,1,0.92,0.58,0.67,0.58,0.67,0.83,0.42,0.83,0.42,0.33,0.25,0.08,0.33,0.08,0.08],
    [0.83,0.92,1,0.5,0.58,0.67,0.75,0.92,0.58,0.92,0.5,0.42,0.33,0.17,0.42,0.17,0.17],
    [0.67,0.58,0.5,1,0.92,0.83,0.75,0.5,0.75,0.58,0.5,0.58,0.5,0.33,0.08,0.33,0.33],
    [0.58,0.67,0.58,0.92,1,0.92,0.83,0.67,0.83,0.67,0.58,0.67,0.5,0.42,0.17,0.42,0.42],
    [0.5,0.58,0.67,0.83,0.92,1,0.92,0.75,0.92,0.75,0.67,0.75,0.67,0.58,0.25,0.5,0.5],
    [0.58,0.67,0.75,0.75,0.83,0.92,1,0.67,0.83,0.67,0.75,0.67,0.58,0.42,0.17,0.42,0.42],
    [0.75,0.83,0.92,0.5,0.67,0.75,0.67,1,0.75,0.92,0.42,0.5,0.42,0.25,0.5,0.33,0.33],
    [0.42,0.42,0.58,0.75,0.83,0.92,0.83,0.75,1,0.67,0.58,0.67,0.75,0.58,0.33,0.58,0.58],
    [0.75,0.83,0.92,0.58,0.67,0.75,0.67,0.92,0.67,1,0.42,0.5,0.42,0.25,0.5,0.33,0.33],
    [0.33,0.42,0.5,0.5,0.58,0.67,0.75,0.2,0.58,0.42,1,0.92,0.83,0.67,0.42,0.67,0.67],
    [0.25,0.33,0.42,0.58,0.67,0.75,0.67,0.5,0.67,0.5,0.92,1,0.92,0.75,0.5,0.75,0.75],
    [0.08,0.25,0.33,0.5,0.5,0.67,0.58,0.42,0.75,0.42,0.83,0.92,1,0.83,0.58,0.83,0.83],
    [0,0.08,0.17,0.33,0.42,0.58,0.42,0.25,0.58,0.25,0.67,0.75,0.83,1,0.75,0.92,0.92],
    [0.25,0.33,0.42,0.08,0.17,0.25,0.17,0.5,0.33,0.5,0.42,0.5,0.58,0.75,1,0.75,0.75],
    [0,0.08,0.17,0.33,0.42,0.5,0.42,0.33,0.58,0.33,0.67,0.75,0.83,0.92,0.75,1,0.92],
    [0,0.08,0.17,0.33,0.42,0.5,0.42,0.33,0.58,0.33,0.67,0.75,0.83,0.92,0.75,0.92,1]])

    if args.dset == 'city_wise_png':
        similarity_matrix_16 = np.delete(np.delete(similarity_matrix, 6, axis=0), 6, axis=1)
        similarity_matrix = keep_diagonal_near_elements(similarity_matrix, 3)
        similarity_matrix = np.delete(np.delete(similarity_matrix, 6, axis=0), 6, axis=1)
    elif args.dset == 'city_wise_png_jilin':
        indices_to_remove = args.remove_indices
        similarity_matrix_16 = np.delete(similarity_matrix, indices_to_remove, axis=0)  # 删除行
        similarity_matrix_16 = np.delete(similarity_matrix_16, indices_to_remove, axis=1)  # 删除列
    
    
        similarity_matrix=keep_diagonal_near_elements(similarity_matrix, 3)    
        indices_to_remove = args.remove_indices
        similarity_matrix = np.delete(similarity_matrix, indices_to_remove, axis=0)  # 删除行
        similarity_matrix = np.delete(similarity_matrix, indices_to_remove, axis=1)  # 删除列


    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])#初始的准确率
    accuracy=calculate_weighted_accuracy(predict, all_label, similarity_matrix_16)
    
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    # flag=0
    for _ in range(2):#centroid 好怪论文是softmax但是这里似乎是hard label？
        initc = aff.transpose().dot(all_fea) # 加权
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None]) # 除以加权和
        cls_count = np.eye(K)[predict].sum(axis=0) # 类别个数
        labelset = np.where(cls_count > args.threshold) # 所有类别出现的数组
        labelset = labelset[0] # 取出具体的list

        dd = cdist(all_fea, initc[labelset], args.distance) # 计算feautre 和 类别出现的feature中心的距离
        pred_label = dd.argmin(axis=1) # 获得最近的类别中心的索引
        predict = labelset[pred_label]#获得新的predict pseudo label /将这些索引映射回 labelset 中实际的类别标签。
        # aff_cp=torch.from_numpy(aff)
        # if flag<2:        
        predict = refine_pseudo_label_with_relatedness(predict, all_output, similarity_matrix)
        #     flag+=1

        aff = np.eye(K)[predict] # 又是output了，这里的predict应该是softmax的？

    acc = calculate_weighted_accuracy(predict, all_label, similarity_matrix_16)

    # 计算置信区间的pseudo label准确率
    accuracies, ranges = calculate_accuracy_per_confidence_bin(all_output, all_label)

    # Display results
    for bin_range, acc_print in zip(ranges, accuracies):
        print(f"Bin {bin_range[0]:.1f} - {bin_range[1]:.1f}: Accuracy = {acc_print if acc_print is not None else 'No samples'}")


    # acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)#提升后的准确率
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)#这是等于原先准确率的概率

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    # predict=predict.cpu().numpy()
    return predict.astype('int')


def keep_diagonal_near_elements(matrix, k):
    # matrix 是输入矩阵，k 是要保留对角线及其周围 k 个元素
    n = matrix.shape[0]
    
    # 构建上三角矩阵，保留对角线及其上 k 行
    upper_mask = np.triu(np.ones_like(matrix), -k)
    
    # 构建下三角矩阵，保留对角线及其下 k 行
    lower_mask = np.tril(np.ones_like(matrix), k)
    
    # 结合上下三角，形成掩码
    mask = upper_mask * lower_mask
    
    # 应用掩码，非对角线周围元素设置为0
    return matrix * mask

def refine_pseudo_label_with_relatedness(predict, all_output, relatedness_matrix, max_threshold=0.4, similarity_reduction_factor=0.7):
    """
    使用相关性矩阵和自适应阈值来优化伪标签。
    :param predict: 原始的预测标签 (tensor, batch_size,)
    :param all_output: 模型输出的概率分布 (tensor, batch_size, num_classes)
    :param relatedness_matrix: 类别相关性矩阵 (numpy array, num_classes, num_classes)
    :param min_threshold: 最小的伪标签阈值
    :param max_threshold: 最大的伪标签阈值
    :param similarity_reduction_factor: 用于调整相似性对阈值的影响
    :return: 优化后的伪标签 (numpy array, batch_size,)
    """

    # 获取所有伪标签的最大置信度
    max_confidence, _ = torch.max(all_output, dim=1)

    # 初始化 refined_pseudo_labels 为原始预测标签
    # refined_pseudo_labels = predict.cpu().numpy().copy()
    refined_pseudo_labels = predict.copy()


    # 自适应阈值基于每个样本的置信度和相似性调整
    # adaptive_thresholds = adaptive_threshold_with_similarity(max_confidence, similarity_reduction_factor, min_threshold, max_threshold)

    # 选择低于自适应阈值的样本
    low_confidence_indices = max_confidence < max_threshold

    # 对于低置信度的样本，通过相关性矩阵调整伪标签
    if torch.nonzero(low_confidence_indices).squeeze().dim() == 0:
        print("低置信度样本")
    else:
        for i in torch.nonzero(low_confidence_indices).squeeze():
            # 获取该样本的输出概率分布
            output_probs = all_output[i].cpu().numpy()

            # 调整相关性矩阵，降低相似类别的影响，但保留对角线元素为1
            identity_matrix = np.eye(relatedness_matrix.shape[0])
            adjusted_relatedness_matrix = identity_matrix + (relatedness_matrix - identity_matrix) * similarity_reduction_factor

            # 使用相关性矩阵来平滑概率分布
            refined_probs = np.dot(output_probs, adjusted_relatedness_matrix)

            # 选择平滑后概率最大的类别作为新的伪标签
            refined_label = np.argmax(refined_probs)
            refined_pseudo_labels[i] = refined_label

    return refined_pseudo_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage1Step2')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=8, help="max iterations")
    parser.add_argument('--interval', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='city_wise_png_jilin', choices=['city_wise_png','city_wise_png_jilin'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet18, resnet50")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--ssl', type=float, default=0.0) 
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--list_folder', type=str, default='../data/')
    parser.add_argument('--data_folder', type=str, default='../../../../../dataset/city_wise_png_ext_jilin')

    parser.add_argument('--remove_indices', type=int, nargs='+', default=[6, 15], help="indices to remove from the lcz matrix")

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

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s].upper())#这里修改[0]
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s].upper()+names[args.t].upper())#这里修改[0]
        args.name = names[args.s].upper()+names[args.t].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.ssl > 0:
             args.savename += ('_ssl_' + str(args.ssl))
        
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()

        train_target(args)

        args.out_file.close()
