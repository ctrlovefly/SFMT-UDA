import torch
import torch.nn as nn
import network  # Assuming the definition of ResBase, feat_bootleneck, feat_classifier is in network.py
import argparse
import os.path as osp


class MultiHeadResNet50(nn.Module):
    def __init__(self, args, num_heads=2):
        super(MultiHeadResNet50, self).__init__()

        # Initialize the backbone (netF)
        self.backbone = network.ResBase(res_name=args.net, num_target=len(args.target_names)).cuda()

        # Initialize multiple heads (bottleneck + classifier)
        self.heads = nn.ModuleList()
        self.backbone_cache = None
        for _ in range(num_heads):
            bottleneck = network.feat_bootleneck(
                type=args.classifier, 
                feature_dim=self.backbone.in_features, 
                bottleneck_dim=args.bottleneck
            ).cuda()

            classifier = network.feat_classifier(
                type=args.layer, 
                class_num=args.class_num, 
                bottleneck_dim=args.bottleneck
            ).cuda()

            # Combine bottleneck and classifier into a single head
            self.heads.append(nn.Sequential(bottleneck, classifier))
        # self.img_cn_list = CrossNorm_list(len(args.target_names))
        # self.default_gpu = 0
        # self.img_cn_list = self.init_device(self.img_cn_list, gpu_id=self.default_gpu, whether_DP=False)
    def extract_features(self, x):
        """
        Extract features from the backbone.
        :param x: Input image tensor.
        :return: Feature tensor from the backbone.
        """
        return self.backbone(x)

    def head(self, features, head_id):
        """Pass features through the specified head by head_id"""
        return self.heads[head_id](features)

    def forward(self, x, domain_list):
        # Extract features from the backbone (netF)
        features = self.backbone(x)
        # print(f"After self.backbone: {torch.cuda.memory_allocated() / 1024 ** 2} MB")

        # if self.backbone_cache is None:
        #     self.backbone_cache = self.backbone(x).detach()
        # Pass features through each head (bottleneck + classifier)
        logits = []
        for head_id in domain_list:
            logits.append(self.head(features, head_id))
        # for i in domain_list:
        #     x1=self.heads[i](features)
        #     logits.append(x1)
        return logits
    
    def clear_cache(self):
        self.backbone_cache = None

    
    def load_pretrained_weights(self, backbone_path, bottleneck_paths, classifier_paths):
        # 加载 backbone 的权重
        backbone_weights = torch.load(backbone_path)
        self.backbone.load_state_dict(backbone_weights)

        # 加载每个 head 的 bottleneck 和 classifier 权重
        for i, head in enumerate(self.heads):
            # 加载 bottleneck
            bottleneck_weights = torch.load(bottleneck_paths)  # 如果是多个文件，按需调整路径
            head[0].load_state_dict(bottleneck_weights)

            # 加载 classifier
            classifier_weights = torch.load(classifier_paths)  # 如果是多个文件，按需调整路径
            head[1].load_state_dict(classifier_weights)
 
class MultiHeadResNet50_init(nn.Module):
    def __init__(self, args, num_heads=2):
        super(MultiHeadResNet50_init, self).__init__()

        # Initialize the backbone (netF)
        self.backbone = network.ResBase(res_name=args.net).cuda()

        # Initialize multiple heads (bottleneck + classifier)
        self.heads = nn.ModuleList()
        self.backbone_cache = None
        for _ in range(num_heads):
            bottleneck = network.feat_bootleneck(
                type=args.classifier, 
                feature_dim=self.backbone.in_features, 
                bottleneck_dim=args.bottleneck
            ).cuda()

            classifier = network.feat_classifier(
                type=args.layer, 
                class_num=args.class_num, 
                bottleneck_dim=args.bottleneck
            ).cuda()

            # Combine bottleneck and classifier into a single head
            self.heads.append(nn.Sequential(bottleneck, classifier))
        # self.img_cn_list = CrossNorm_list(len(args.target_names))
        # self.default_gpu = 0
        # self.img_cn_list = self.init_device(self.img_cn_list, gpu_id=self.default_gpu, whether_DP=False)


    def head(self, features, head_id):
        """Pass features through the specified head by head_id"""
        return self.heads[head_id](features)

    def forward(self, x, domain_list):
        # Extract features from the backbone (netF)
        features = self.backbone(x)
        # print(f"After self.backbone: {torch.cuda.memory_allocated() / 1024 ** 2} MB")

        # if self.backbone_cache is None:
        #     self.backbone_cache = self.backbone(x).detach()
        # Pass features through each head (bottleneck + classifier)
        logits = []
        for head_id in domain_list:
            logits.append(self.head(features, head_id))
        # for i in domain_list:
        #     x1=self.heads[i](features)
        #     logits.append(x1)
        return logits
    
    def clear_cache(self):
        self.backbone_cache = None
  

    # def init_device(self, net, gpu_id=None, whether_DP=False):
    #     gpu_id = gpu_id or self.default_gpu
    #     device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
    #     net = net.to(device)
    #     # if torch.cuda.is_available():
    #     # if whether_DP:
    #         #net = DataParallelWithCallback(net, device_ids=[0])
    #         # net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
    #     return net
# parser = argparse.ArgumentParser(description='SHOT')
# args = parser.parse_args()
# args.net='resnet50'
# args.classifier="bn"
# args.bottleneck=256
# args.layer="wn"
# args.class_num=16
# args.output='MTDA_weights'
# args.output_dir_src=osp.join(args.output, 'city_wise_png', 'guangzhou')
# backbone_path = args.output_dir_src + '/target_F.pt'
# model = MultiHeadResNet50(args, num_heads=2)

# model.backbone.load_state_dict(torch.load(backbone_path))

# # Load each head's bottleneck (netB) and classifier (netC) weights
# for i, head in enumerate(model.heads):
#     bottleneck_path = args.output_dir_src + f'/target_B.pt'
#     classifier_path = args.output_dir_src + f'/target_C.pt'

#     # Load bottleneck and classifier weights
#     head[0].load_state_dict(torch.load(bottleneck_path))  # bottleneck part
#     head[1].load_state_dict(torch.load(classifier_path))  # classifier part

