import torch
import torch.nn as nn
import argparse
from helper.mixup_utils import progress_bar
import torch.nn.functional as F
from test_model import MultiHeadResNet50
from torchvision import transforms
from MH_MTDA_self_training import ImageListWithLogits_SingleDomain
from torch.utils.data import DataLoader, Dataset
from utils import InfiniteDataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import umap
from matplotlib import cm  # 用于获取颜色映射
from matplotlib.lines import Line2D
import os
np.random.seed(4)

def test(testloader, model):
    # global best_acc
    # global best_test_loss
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
    num_classes =16
    confusion_matrices = {i: np.zeros((num_classes, num_classes), dtype=int) for i in range(len(args.target_names))}
    overall_confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)  # 总体混淆矩阵

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

                    # 更新每个域的混淆矩阵
                    confusion_matrices[i_target] += confusion_matrix(
                        current_targets.cpu().numpy(),
                        predicted.cpu().numpy(),
                        labels=list(range(num_classes))
                    )

                    # 更新总体混淆矩阵
                    overall_confusion_matrix += confusion_matrix(
                        current_targets.cpu().numpy(),
                        predicted.cpu().numpy(),
                        labels=list(range(num_classes))
                    )
            # print(100.*pseudo_correct/total)
            progress_bar(i_target, len(args.target_names),'Loss: %.3f | Acc: %.3f%% Pseudo Acc: %.3f%% (%d/%d)'% (test_loss/(i_target+1), 100.*correct/total, 100.*pseudo_correct/total, correct, total))
        
        for i_target, cm in confusion_matrices.items():
            print(f"Confusion Matrix for Domain {i_target}:")
            print(cm)
            # 计算OA (Overall Accuracy)
            diagonal_sum = cm.trace()  # 获取混淆矩阵对角线的和
            total_samples = cm.sum()  # 获取混淆矩阵中所有元素的总和
            oa = diagonal_sum / total_samples  # 计算OA
            
            # 打印OA
            print(f"Overall Accuracy (OA) for Domain {i_target}: {oa:.4f}")                                                                 

        # 打印所有样本的混淆矩阵
        print("Overall Confusion Matrix:")
        print(overall_confusion_matrix)

        
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
        # if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        #     checkpoint(args, netF, netB, netC)
        avg_test_loss=test_loss/(i_target+1)
        # if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        print(f'The best model with loss: {best_test_loss:.2f}%')

        # if acc > best_acc:
            # best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)

def data_load(args):
    # def image_train(resize_size=256, crop_size=224, alexnet=False):
    #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225])
    #     return transforms.Compose([

    #         transforms.Resize((crop_size, crop_size)),
    #         transforms.ToTensor(),
    #         normalize
    #     ])

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
    target_train_loader_list['test'] = []
    target_train_list= []
    csv_file = f'{args.txt_folder}/{args.dset}/{args.csv_filename}'  # CSV 文件路径
    print("Loading data from: ", csv_file)
    for i in range(len(args.target_names)):
        target_train = ImageListWithLogits_SingleDomain(csv_file, i+1, transform=image_train()) # 实例化dataset
        target_train_list.append(target_train)
        target_train_loader_list['test'].append(DataLoader(target_train_list[i], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False))
    return target_train_loader_list, target_train_list

def overall_accuracy(confusion_matrix):
    """
    计算整体准确率 (Overall Accuracy, OA)
    
    参数:
    confusion_matrix (np.ndarray): 混淆矩阵，大小为 [num_classes, num_classes]
    
    返回:
    float: 整体准确率 (OA)
    """
    # 计算对角线元素之和（正确分类的样本数）
    correct_samples = np.trace(confusion_matrix)  # 对角线元素的和
    
    # 计算混淆矩阵所有元素之和（总样本数）
    total_samples = np.sum(confusion_matrix)
    
    # 计算整体准确率
    oa = correct_samples / total_samples
    
    return oa

def weighted_accuracy(confusion_matrix, similarity_matrix):
    """
    计算加权准确率 (Weighted Accuracy, WA)
    
    参数:
    confusion_matrix (np.ndarray): 混淆矩阵，大小为 [num_classes, num_classes]
    similarity_matrix (np.ndarray): 相似性矩阵，大小为 [num_classes, num_classes]
    
    返回:
    float: 加权准确率 (WA)
    """
    # 逐元素相乘得到加权混淆矩阵
    weighted_cm = confusion_matrix * similarity_matrix
    
    # 计算加权混淆矩阵的所有元素之和
    weighted_cm_sum = np.sum(weighted_cm)
    
    # 计算原始混淆矩阵的所有元素之和
    confusion_matrix_sum = np.sum(confusion_matrix)
    
    # 计算加权准确率
    wa = weighted_cm_sum / confusion_matrix_sum
    
    return wa

def extract_and_save_features(model, testloader):
    """
    Extract features from the backbone and save them for dimensionality reduction.
    :param model: MultiHeadResNet50 model.
    :param dataloader: Dataloader for the test dataset.
    :param device: Device (CPU or GPU).
    :return: Feature tensor and corresponding labels.
    """
    model.eval()
    all_features = []
    all_labels = []
    domain_labels=[]
    with torch.no_grad():
        for i_target in range(len(args.target_names)):
            #每个域获取一个批次数据
            # data_target_list.append(next(all_loader[i_target]))#按tgt_dataset_list的顺序调用不同域的数据集loader

            for batch_idx, (inputs, targets, pseudo_lbl, domains, _) in enumerate(testloader[i_target]):
                # print(inputs)
                # print(f"domain{domains}")
                # print(f"targets{targets}")
                if use_cuda:
                    inputs, targets, pseudo_lbl, domains = inputs.cuda(), targets.cuda(), pseudo_lbl.cuda(), domains.cuda()
                # inputs, pseudo_lbl = Variable(inputs), Variable(pseudo_lbl)
                # outputs = netC(netB(netF(inputs)))
        #         logits_list=model(inputs, [i_target])
        # for inputs, labels in dataloader:
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
            
        #     # Extract features
                    features = model.extract_features(inputs)
                    all_features.append(features.cpu())
                    all_labels.append(targets.cpu())
                    domain_labels.append(domains.cpu())

    # Concatenate all features and labels
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    domain_labels = torch.cat(domain_labels,dim=0)
    return all_features, all_labels, domain_labels



def plot_umap(all_embeddings, all_labels, domain_labels, sample_names, output_file='umap_projection.png',output_file2='umap_projection2.png'):
    """
    Function to perform UMAP on embeddings and plot the results.

    Parameters:
    - all_embeddings (numpy.ndarray): The concatenated feature embeddings.
    - all_labels (numpy.ndarray): The true labels corresponding to the embeddings.
    - sample_names (list): List of sample/domain names.
    - output_file (str): The file path to save the UMAP plot.
    """
    # Apply UMAP to reduce the dimensionality of the feature embeddings
    reducer = umap.UMAP(n_components=2, random_state=42, metric='cosine')
    umap_embeddings = reducer.fit_transform(all_embeddings)
    print(umap_embeddings.shape)

    # Plot all samples on the same figure
   
    
    # Plot each domain's points with different colors
    domain_labels = np.array(domain_labels)
    all_labels = np.array(all_labels)
    unique_values = np.unique(all_labels)

    plt.figure(figsize=(8, 6))
    colormap = plt.cm.tab20
    num_domains = len(sample_names)
    colors = [colormap(i / num_domains) for i in range(num_domains)]

    for i, sample in enumerate(sample_names):
        # Select the indices corresponding to the current domain's embeddings
        domain_indices = domain_labels == (i+1)  # Assuming `i` corresponds to the domain index
        
        # Plot the points for the current domain with a specific color
        plt.scatter(umap_embeddings[domain_indices, 0], umap_embeddings[domain_indices, 1], label=sample, s=10)

    # plt.title('UMAP Projection of Feature Space')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    # Add a legend to differentiate between samples (domains)
    plt.legend(loc='lower left')

    # Save the UMAP visualization
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print("fig domain saved")

    plt.figure(figsize=(8, 6))
    colormap = plt.cm.tab20
    num_categories = len(unique_values)  # 类别数量
    colors = [colormap(i / num_categories) for i in range(num_categories)] 
    class_lcz = ['LCZ1', 'LCZ2', 'LCZ3', 'LCZ4', 'LCZ5', 'LCZ6', 'LCZ8', 'LCZ9', 'LCZ10', 'LCZA', 'LCZB', 'LCZC', 'LCZD', 'LCZE', 'LCZF', 'LCZG']

    for i, sample in enumerate(unique_values):
        label_indices = all_labels == sample
        plt.scatter(umap_embeddings[label_indices, 0], umap_embeddings[label_indices, 1], label=class_lcz[i], s=10,c=colors[i])

    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')

    # Add a legend to differentiate between samples (domains)
    plt.legend(loc='lower left', ncol=4)

    # Save the UMAP visualization
    plt.savefig(output_file2, bbox_inches='tight')
    plt.close()
    print("fig class saved")
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--outpath', default='confusion_matrix', type=str)
    parser.add_argument('--txt_folder', default='csv_pseudo_labels', type=str)
    parser.add_argument('--dset', type=str, default='city_wise_png_jilin', choices=['city_wise_png', 'city_wise_png_jilin'])
    parser.add_argument('--csv_filename', default='guangzhou_source.csv', type=str)# # class
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--premodel', default='./MTDA_weights/Stage2_step2_city_wise_png_jilin_1.pt', type=str)# wuhan class
    parser.add_argument('--net', default="resnet18", type=str, help='model type (default: ResNet18)')
    
    args = parser.parse_args()
    # args.target_names = ['Guangzhou','Hefei', 'Hong Kong', 'Nanchang', 'Nanjing', 'Shanghai']
    args.target_names =  ['guangzhou', 'changsha']

    pretrain_model=args.premodel
    use_cuda = True
    criterion = nn.CrossEntropyLoss()
    num_heads = 2
    args.classifier="bn"
    args.bottleneck=256
    args.layer="wn"
    args.class_num=15

    model = MultiHeadResNet50(args,num_heads=num_heads)


    checkpoint = torch.load(f'{pretrain_model}')  # 加载 .pt wuhan 这里保存的应该是不包含transfer模块的参数
    model.load_state_dict(checkpoint)  # 将权重加载到模型中

    all_loader, _ = data_load(args)  
   
    # features, labels, domain_labels = extract_and_save_features(model, all_loader['test'])
    # plot_umap(features, labels, domain_labels, args.target_names, output_file='umap_projection_wuhan_ours_1226_domain.png',output_file2='umap_projection_wuhan_ours_1226_classes.png')

    test_loss, test_acc = test(all_loader['test'], model)
    

