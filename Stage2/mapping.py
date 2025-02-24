import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from test_model import MultiHeadResNet50
import argparse
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import network

# 1. 读取遥感影像
def read_tif(tif_path):
    with rasterio.open(tif_path) as src:
        image = src.read([1, 2, 3])  # 读取RGB波段
        transform = src.transform
        width, height = src.width, src.height
    return image, transform, width, height

# 2. 滑动窗口切割影像
# def sliding_window_cut(image, window_size=(640, 640), stride=200):
#     patches = []
#     locations = []
#     for i in range(0, image.shape[1] - window_size[1], stride):
#         for j in range(0, image.shape[2] - window_size[0], stride):
#             patch = image[:, i:i+window_size[1], j:j+window_size[0]]
#             patches.append(patch)
#             # print((i, j))
#             locations.append((i, j))
#     return patches, locations
def sliding_window_cut(image, window_size=(640, 640), stride=200):
    patches = []
    locations = []
    
    height, width = window_size[0], window_size[1]
    
    start_i = height // 2 - stride // 2
    start_j =  width // 2 - stride // 2
    end_i = image.shape[1] - (height // 2 + stride // 2)
    end_j = image.shape[2] - (width // 2 + stride // 2)

    for i in range(start_i, end_i, stride):
        for j in range(start_j, end_j, stride):
            patch = image[:, i-start_i:i+start_i+stride, j-start_j:j+start_j+stride]
            # print( i-start_i)
            # print(i+start_i+stride)
            # print(j-start_j)
            # print(j+start_j+stride)
            # print("******")
            patches.append(patch)
            # print(patch.shape)
            # print((i,j))
            locations.append((i, j))

    return patches, locations

# 3. 加载预训练模型
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def image_train(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
class PatchDataset(Dataset):
    def __init__(self, patches, transform=None):
        self.patches = patches
        self.transform = transform
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        # 将patch从 (3, 640, 640) 转换为 (640, 640, 3)
        patch = np.transpose(patch, (1, 2, 0))  # 转换维度顺序
        patch = Image.fromarray(patch.astype(np.uint8))  # 转换为PIL.Image
        if self.transform:
            patch = self.transform(patch)
        return patch

       
# def data_load(args):
#     # def image_train(resize_size=256, crop_size=224, alexnet=False):
#     #     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                                      std=[0.229, 0.224, 0.225])
#     #     return transforms.Compose([

#     #         transforms.Resize((crop_size, crop_size)),
#     #         transforms.ToTensor(),
#     #         normalize
#     #     ])

#     def image_train(resize_size=256, crop_size=224, alexnet=False):
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                         std=[0.229, 0.224, 0.225])
#         return transforms.Compose([
#             transforms.Resize((resize_size, resize_size)),
#             transforms.CenterCrop(crop_size),
#             transforms.ToTensor(),
#             normalize
#         ])
#     target_train_loader_list = {}
#     target_train_loader_list['test'] = []
#     target_train_list= []
#     csv_file = f'{args.txt_folder}/{args.dset}/{args.csv_filename}'  # CSV 文件路径
#     print("Loading data from: ", csv_file)
#     for i in range(len(args.target_names)):
#         target_train = ImageListWithLogits_SingleDomain(csv_file, i+1, transform=image_train()) # 实例化dataset
#         target_train_list.append(target_train)
#         target_train_loader_list['test'].append(DataLoader(target_train_list[i], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False))
#     return target_train_loader_list, target_train_list
# 4. 分类每个patch
def classify_patches(patches, model, device):
    dataset = PatchDataset(patches, transform=image_train())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():  # 禁用梯度计算
        for batch in tqdm(dataloader, desc="Classifying patches", unit="batch"):
            batch = batch.to(device)  # 将数据移动到指定设备
            outputs = model(batch, [0])  # 调用模型，传递 domain_list=[0]
            
            # # 检查 outputs 的形状
            # for logits in outputs:
            #     print(f"Logits shape: {logits.shape}")  # 打印 logits 的形状
            
            # 拼接所有 logits 并计算预测类别
            logits = outputs[0]  # 拼接 logits
            predicted_class = logits.argmax(dim=1).cpu().numpy()  # 选择得分最大的类别
            predictions.extend(predicted_class)  # 存储预测结果
    
    # for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
    #     # print(type(batch))
    #     batch = batch.to(device)
    #     with torch.no_grad():
    #         outputs = model(batch,[0])
    #     logits = torch.cat(outputs, dim=0)  # 拼接所有 logits
    #     predicted_class = logits.argmax(dim=1).cpu().numpy()  # 选择得分最大的类别
    #     predictions.extend(predicted_class)
    return predictions

def classify_patches_CoNMix(patches, model_list):
    # print('Started testing on ', len(dsets), ' images')
    dataset = PatchDataset(patches, transform=image_train())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    netF = model_list[0]
    netB = model_list[1]
    netC = model_list[2]

    netF.eval()
    netB.eval()
    netC.eval()

    all_predictions = []
    
    with torch.no_grad():
        print('Started Testing')
        iter_test = iter(dataloader)
        for _ in tqdm(range(len(dataloader))):
            data = next(iter_test)
            inputs = data.to('cuda')
            
            outputs = netC(netB(netF(inputs)))
            
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.append(predicted.cpu())

    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    
    return all_predictions

# 5. 拼接分类结果
def reassemble_classification(predictions, locations, width, height, window_size=(640, 640), stride=200):
    classified_image = np.zeros((height, width), dtype=np.uint8)
    count_matrix = np.zeros((height, width), dtype=int)
    # print(len(predictions))
    # print(len(locations))
    height, width = window_size[0], window_size[1]
    start_i = height // 2 - stride // 2
    start_j =  width // 2 - stride // 2
    # end_i = image.shape[1] - (height // 2 + stride // 2)
    # end_j = image.shape[2] - (width // 2 + stride // 2)
    for pred, (i, j) in zip(predictions, locations):
        # print((i, j))
        classified_image[i:i+stride, j:j+stride] = pred

    classified_image = classified_image.astype(np.uint8)
    return classified_image

# 6. 保存输出为tif
def save_classification_output(classified_image, output_path, transform):
    # 创建新的tif文件
    with rasterio.open(output_path, 'w', driver='GTiff', count=1, dtype=classified_image.dtype, width=classified_image.shape[1], height=classified_image.shape[0], crs='+proj=latlong', transform=transform) as dst:
        dst.write(classified_image, 1)

# 主程序
def main():
    parser = argparse.ArgumentParser(description='LCZ mapping')
    parser.add_argument('--premodel', default='./MTDA_weights/Stage2_step2_city_wise_png_jilin_1.pt', type=str)
    parser.add_argument('--net', default="resnet18", type=str, help='model type (default: ResNet18)')
    parser.add_argument('--method', default="SFMT-UDA", type=str, help='classification method (default: SFMT-UDA)')
    parser.add_argument('--outpath', default='classified_output_SFMT-UDA_cs.tif', type=str)





    args = parser.parse_args()
    # tif_path = '/media/lscsc/nas/qianqian/kexin_label/image/Extract_tif11.tif'
    # tif_path = '/media/lscsc/nas/qianqian/kexin_label/image/clear_rgb.tif'
    tif_path = '/media/lscsc/nas/qianqian/kexin_label/image/clear_1_1_p.tif'


    # model_path = './MTDA_weights/Stage2_step2_city_wise_png_jilin_1.pt'
    output_path = args.outpath
    
    # 1. 读取影像
    image, transform, width, height = read_tif(tif_path)
    
    # 2. 滑动窗口切割影像
    patches, locations = sliding_window_cut(image)
    # print(locations)
    # print(len(patches)) #80940
    # print(patches[0].shape)#(3, 640, 640)
    
    # 3. 加载预训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_heads = 2
    args.classifier="bn"
    args.bottleneck=256
    args.layer="wn"
    args.class_num=15
    pretrain_model=args.premodel
    cls_method=args.method

    if cls_method == 'SFMT-UDA':
        model = MultiHeadResNet50(args, num_heads = num_heads)
        checkpoint = torch.load(f'{pretrain_model}')
        model.load_state_dict(checkpoint)  # 将权重加载到模型中
        # 4. 对每个patch进行分类
        predictions = classify_patches(patches, model, device)
    elif cls_method == 'CoNMix':
        modelpathF = '/media/lscsc/nas/qianqian/UDA/multitarget/CoNMix/MTDA_weights/city_wise_png_jilin/guangzhou_source/target_F.pt'
        modelpathB = '/media/lscsc/nas/qianqian/UDA/multitarget/CoNMix/MTDA_weights/city_wise_png_jilin/guangzhou_source/target_B.pt'
        modelpathC = '/media/lscsc/nas/qianqian/UDA/multitarget/CoNMix/MTDA_weights/city_wise_png_jilin/guangzhou_source/target_C.pt'

        netF = network.ResBase(res_name='resnet50').cuda()
        netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
        netC = network.feat_classifier(type='wn', class_num=args.class_num, bottleneck_dim=256).cuda()
        netF.load_state_dict(torch.load(modelpathF))
        netB.load_state_dict(torch.load(modelpathB))
        netC.load_state_dict(torch.load(modelpathC))
        model_list = [netF, netB, netC]
        predictions = classify_patches_CoNMix(patches, model_list)
    elif cls_method == 'aggregation':
        modelpathF = '/media/lscsc/nas/qianqian/UDA/semisupervised/SHOT-plus/code/uda/ckps/target/uda/city_wise_png_jilin_combined_gz_source/GUANGZHOU_SOURCECHANGSHA/target_F_par_0.3_ssl_0.6.pt'
        modelpathB = '/media/lscsc/nas/qianqian/UDA/semisupervised/SHOT-plus/code/uda/ckps/target/uda/city_wise_png_jilin_combined_gz_source/GUANGZHOU_SOURCECHANGSHA/target_B_par_0.3_ssl_0.6.pt'
        modelpathC = '/media/lscsc/nas/qianqian/UDA/semisupervised/SHOT-plus/code/uda/ckps/target/uda/city_wise_png_jilin_combined_gz_source/GUANGZHOU_SOURCECHANGSHA/target_C_par_0.3_ssl_0.6.pt'

        netF = network.ResBase(res_name='resnet50').cuda()
        netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
        netC = network.feat_classifier(type='wn', class_num=args.class_num, bottleneck_dim=256).cuda()
        netF.load_state_dict(torch.load(modelpathF))
        netB.load_state_dict(torch.load(modelpathB))
        netC.load_state_dict(torch.load(modelpathC))
        model_list = [netF, netB, netC]
        predictions = classify_patches_CoNMix(patches, model_list)
    elif cls_method == 'source-only':
        modelpathF = '/media/lscsc/nas/qianqian/UDA/SFMTDA/Stage1/code/uda/ckps/source/uda/city_wise_png_jilin/GUANGZHOU_SOURCE/source_F.pt'
        modelpathB = '/media/lscsc/nas/qianqian/UDA/SFMTDA/Stage1/code/uda/ckps/source/uda/city_wise_png_jilin/GUANGZHOU_SOURCE/source_B.pt'
        modelpathC = '/media/lscsc/nas/qianqian/UDA/SFMTDA/Stage1/code/uda/ckps/source/uda/city_wise_png_jilin/GUANGZHOU_SOURCE/source_C.pt'

        netF = network.ResBase(res_name='resnet50').cuda()
        netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
        netC = network.feat_classifier(type='wn', class_num=args.class_num, bottleneck_dim=256).cuda()
        netF.load_state_dict(torch.load(modelpathF))
        netB.load_state_dict(torch.load(modelpathB))
        netC.load_state_dict(torch.load(modelpathC))
        model_list = [netF, netB, netC]
        predictions = classify_patches_CoNMix(patches, model_list)

    

    
    # 5. 拼接分类结果
    classified_image = reassemble_classification(predictions, locations, width, height)
    
    
    # 6. 保存输出
    save_classification_output(classified_image, output_path, transform)
    
    print(f"Classification complete, saved to {output_path}")

if __name__ == '__main__':
    main()
