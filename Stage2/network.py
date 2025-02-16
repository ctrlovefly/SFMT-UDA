import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from non_local_embedded_gaussian import NONLocalBlock2D
import matplotlib.pyplot as plt
import os
import random
# from test_model import CrossNorm_list
 
def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std#函数最终返回每个样本的特征图在各通道内的均值和标准差（形状为 (N, C, 1, 1)）。这种统计信息在特征图归一化、风格迁移等操作中常用

def remove_and_select(array, index1, index2):
    # 确保索引在数组范围内
    if max(index1, index2) >= len(array) or min(index1, index2) < 0:
        raise IndexError("索引超出数组范围")
    
    # 移除指定索引的元素
    filtered_array = [array[i] for i in range(len(array)) if i not in (index1, index2)]
    
    # 随机选择一个数
    if not filtered_array:
        raise ValueError("数组中没有可选的元素")
    
    return random.choice(filtered_array)


class CrossNorm_list(nn.Module):
    """CrossNorm module"""
    def __init__(self, nb_target):
        super(CrossNorm_list, self).__init__()
        self.current_id = 0
        self.transfer_id = 0
        self.mean_list = [torch.tensor(0.0).to(0) for i in range(nb_target)]
        self.std_list = [torch.tensor(1.0).to(0) for i in range(nb_target)]

    def forward(self, x ):
        # print( self.mean_list[self.transfer_id])
        # print("*************")
        size = x.size()
        content_mean, content_std = calc_ins_mean_std(x)

        normalized_feat = (x - content_mean.expand(
            size)) / content_std.expand(size)#先normalized
        # 再进行transfered
        # print(self.std_list[self.transfer_id].shape[0])

        try:
            size = list(x.size())  # 将 size 转换为可修改的列表

            if self.std_list[self.transfer_id].shape and self.std_list[self.transfer_id].shape[0] != normalized_feat.shape[0]:
                # if self.std_list[self.transfer_id].shape[0] < normalized_feat.shape[0]:
                if self.std_list[self.transfer_id].shape[0] < normalized_feat.shape[0]: 
                    size[0] = self.std_list[self.transfer_id].shape[0]
                    repeat_count = normalized_feat.shape[0] - size[0] #4是batchsize的大小 默认为4
                    size = torch.Size(size)
                    expanded_std = torch.cat([self.std_list[self.transfer_id].expand(size)] * (repeat_count + 1), dim=0)[:normalized_feat.shape[0]]
                    # torch.cat([self.std_list[self.transfer_id].expand(size), self.std_list[self.transfer_id].expand(size)[size[0]-1:normalized_feat.shape[0]-1]], dim=0)
                    expanded_mean = torch.cat([self.mean_list[self.transfer_id].expand(size)] * (repeat_count + 1), dim=0)[:normalized_feat.shape[0]]
                else:
                    # cut_count = size[0] - normalized_feat.shape[0]
                    expanded_std = self.std_list[self.transfer_id][:normalized_feat.shape[0]].expand(size)
                    expanded_mean = self.mean_list[self.transfer_id][:normalized_feat.shape[0]].expand(size)

                    # torch.cat([self.mean_list[self.transfer_id].expand(size), self.mean_list[self.transfer_id].expand(size)[size[0]-1:normalized_feat.shape[0]-1]], dim=0)
            # else:
                #     size[0] = normalized_feat.shape[0]
                #     size = torch.Size(size)
                #     expanded_std = torch.cat([self.std_list[self.transfer_id].expand(size), self.std_list[self.transfer_id].expand(size)[size[0]-1:normalized_feat.shape[0]-1]], dim=0)
                #     expanded_mean = torch.cat([self.mean_list[self.transfer_id].expand(size), self.mean_list[self.transfer_id].expand(size)[size[0]-1:normalized_feat.shape[0]-1]], dim=0)
           
            else:
                size = torch.Size(size)
                expanded_std = self.std_list[self.transfer_id].expand(size)
                expanded_mean = self.mean_list[self.transfer_id].expand(size)

            x = normalized_feat * expanded_std + expanded_mean
        except:
            

            print(normalized_feat.shape)
            print(f"{self.std_list[self.transfer_id].shape}")
            print(self.mean_list[self.transfer_id].expand(size).shape)
            print(size)
        return x
    
    def update_ins_mean_std(self, x):# 一个batch的数据进来
        with torch.no_grad():
            mean, std = calc_ins_mean_std(x)
            self.mean_list[self.current_id], self.std_list[self.current_id] = mean.detach(), std.detach()
            # print("******update_ins_mean_std******")
            # print(self.current_id)
            # print(self.mean_list[self.current_id])
            # print("******update_ins_mean_std******")
    




def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152}

class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 100
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        self.feature_extractor = ViT_seg(config_vit, img_size=[224, 224], num_classes=config_vit.n_classes)
        self.feature_extractor.load_from(weights=np.load(config_vit.pretrained_path))
        self.in_features = 2048

    def forward(self, x):
        _, feat = self.feature_extractor(x)
        return feat

class ResBase(nn.Module):
    def __init__(self, res_name,se=False, nl=False, num_target=2):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.se=se
        self.nl=nl
        if self.se:
            self.SELayer=SELayer(self.in_features)
        if self.nl:
            self.nlLayer=NONLocalBlock2D(self.in_features)
        self.cross=False
        self.update=False
        self.real_cn = CrossNorm_list(num_target)
        self.origin_cn = CrossNorm_list(num_target)
        for i in self.real_cn.parameters():
            i.requires_grad = False
        
        self.feature_maps = None
           
        def forward_hook(module, input, output):
            self.feature_maps = output  # 保存特征图输出
        
        self.real_cn.register_forward_hook(forward_hook)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.update:
            self.real_cn.update_ins_mean_std(x)
            # print("******if self.update******")
            
        if self.cross:
            # print(f"******{x.shape}")
            x = self.real_cn(x)
            # print("******if self.cross******")

        x = self.layer1(x)
 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.se:
            x=self.SELayer(x)
        if self.nl:
            x=self.nlLayer(x)
        x = self.avgpool(x)
      
        x = x.view(x.size(0), -1)# 将 x 重新调整为二维张量，便于后续的全连接层处理

        return x
    
    def _enable_update(self):
        self.update = True
    
    def _disable_update(self):
        self.update = False

    def _enable_cross_norm(self):
        self.cross = True

    def _disable_cross_norm(self):
        self.cross = False

    def update_id(self, current_id, transfer_id):
        self.real_cn.current_id = current_id
        self.real_cn.transfer_id = transfer_id
        self.origin_cn.current_id = current_id
        self.origin_cn.transfer_id = transfer_id
        # print("******update_id******")
        # print(self.real_cn.current_id)
        # print(self.real_cn.transfer_id)
        # print("******update_id******")
    
    def visualize_feature_map(self,save_dir='feature_maps', file_name='feature_map.png'):
        if self.feature_maps is not None:
            # 将特征图移动到CPU并转换为numpy
            feature_maps = self.feature_maps.detach().cpu().numpy()
            
            # 只显示部分通道的特征图（比如前8个通道）
            num_channels_to_show = min(8, feature_maps.shape[1])
            fig, axs = plt.subplots(1, num_channels_to_show, figsize=(20, 5))

            for i in range(num_channels_to_show):
                axs[i].imshow(feature_maps[0, i, :, :], cmap='viridis')  # 可调整 colormap
                axs[i].axis('off')
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file_name)
            
            # 保存特征图
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)  # 关闭图以释放内存
            print(f"Feature maps saved at: {save_path}")



class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)