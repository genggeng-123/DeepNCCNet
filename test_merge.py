#test.py
#!/usr/bin/env python3
from matplotlib import pyplot as plt
import warnings
import torch
from utils.util import *
from torch.utils.data import DataLoader
import pandas as pd
warnings.filterwarnings("ignore") #忽略警告
import argparse
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from utils.score_ import *
import time
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# weight_path = '/home/lll/Alice/code_classify/my_code/checkpoint/resnet50/2023-05-30-20:14/best_weight_tumor_TCGA_train.pth'
# data = 'TCGA_test'
# threshold = 'tumor'
# model_name = 'resnet50'
# image_path = '/home/lll/Alice/code_classify/my_code/data/'
# csv_path = '/home/lll/Alice/code_classify/my_code/csv/'
# image_path = os.path.join(image_path, data,threshold)
# csv_path = os.path.join(csv_path, data,f'{threshold}.csv')

weight_1 = '/home/lll/Alice/Dr.yu/code_classify/my_code/checkpoint/weight_fine_tune/mobilenetv2/tumor_nor/best_weight_tumor_nor_TCGA_train.pth'
weight_2 = '/home/lll/Alice/Dr.yu/code_classify/my_code/checkpoint/weight_fine_tune/mobilenetv2/tumor_/best_weight_tumor_TCGA_train.pth'
weight_3 = '/home/lll/Alice/Dr.yu/code_classify/my_code/checkpoint/weight_fine_tune/mobilenetv2/0.3+_0.5+/best_weight_0.3+_TCGA_train.pth'

model_name = 'mobilenetv2'


# image_path = os.path.join(image_path, data,threshold)
image_path = '/home/lll/Alice/Dr.yu/code_classify/my_code/data/TCGA_test/TCGA_all/'
csv_path ='/home/lll/Alice/Dr.yu/code_classify/my_code/csv/TCGA_test/TCGA_all.csv'
def read_info():
    df = pd.read_csv(csv_path)
    return df

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-batch_size', type=int, default = 16, help='batch size for dataloader')
    parser.add_argument('-net', default=model_name, type=str, help='net type')
    parser.add_argument('--img_path', default=image_path,
                        help='the number of training epoch')
    args = parser.parse_args()
    df = read_info()
    images = read_images(df, args.img_path)
    print("Total images number: ", len(images))
    # 用于图像转换
    data_transforms = {
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 裁剪中间的224x224区域
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])}

    val_x,  val_y = data_split_test(images, df)
    test_dataset = data_convert_test(val_x, val_y,data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers = 4,  shuffle=True)
    model_1 = get_network(args)
    model_2 = get_network(args)
    model_3 = get_network(args)
    model_1.load_state_dict(torch.load(weight_1))
    model_2.load_state_dict(torch.load(weight_2))
    model_3.load_state_dict(torch.load(weight_3))
    real_labels, pred_labels = [], []
    predicted_list = []
    labels_list = []

    if args.gpu:
        model_1 = model_1.cuda()
        model_2 = model_2.cuda()
        model_3 = model_3.cuda()

    model_1.eval()
    model_2.eval()
    model_3.eval()

    # 开始测试
    start_time = time.time()
    with torch.no_grad():
        for image, label in test_loader:
            real_labels += label.detach().tolist()
            label = label.cuda()
            image = image.cuda()
            pred_1 = model_1(image)
            pred_2 = model_2(image)
            pred_3 = model_2(image)
            out_1 = F.softmax(pred_1, dim=1).cpu().detach().numpy()
            out_2 = F.softmax(pred_2, dim=1).cpu().detach().numpy()
            out_3 = F.softmax(pred_2, dim=1).cpu().detach().numpy()

            out =(0.6 * out_1 +  0.4 * out_2 ) / 2

            # out =(out_1 +  out_2 + out_3) / 3

            pred_label = np.argmax(out, 1)
            labels_list += label.tolist()
            predicted_list += pred_label.tolist()
            pred_labels += list(pred_label)
        val_acc = metrics.accuracy_score(real_labels, pred_labels)
        # 计算fpr和tpr
        fpr, tpr, thresholds = roc_curve(real_labels, pred_labels)

        # 计算AUC曲线值
        auc = roc_auc_score(real_labels, pred_labels)
        # 输出混淆矩阵
        cm = confusion_matrix(labels_list, predicted_list)
        # 计算fpr和tpr
        fpr, tpr, thresholds = roc_curve(real_labels, pred_labels)
        # 计算AUC曲线值
        auc = roc_auc_score(real_labels, pred_labels)
        # 输出混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels_list, predicted_list).ravel()
        val_acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        npv = tn / (tn + fn + 1e-8)
        print(
            "accuracy:{:.4f} precision:{:.4f} specificity:{:.4f} sensitivity:{:.4f}  NPV:{:.4f} PPV:{:.4f}".
            format(val_acc,precision, specificity, sensitivity, npv, ppv))
        # 结束测试
print(cm)
end_time = time.time()
 # 输出测试用的时间
 # 计算模型的计算复杂度
num_params = sum(p.numel() for p in model_1.parameters())
# 计算内存消耗
memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # 单位是MB
print("Memory used:", memory_used, "MB", f"Number of parameters: {num_params}",f"Testing time: {end_time - start_time:.4f}s")