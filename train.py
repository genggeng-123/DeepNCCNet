# 导入数据分析的工具
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from utils.EarlyStopping import EarlyStopping
# 导入读取和展示图片的工具
import matplotlib.pyplot as plt
import argparse
# 导入模型构建的工具

from sklearn import metrics
from utils.util import *
from utils import settings
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available(), torch.cuda.device_count()) # 查看CUDA是否可用
import os
# model_name =  efficientNet, densenet121 , resnetv2 , googlenet,
# vgg16 , alexnet  , alexnet , inception_v3  , mobilenetv2   shufflenet , Xception
os.environ['CUDA_VISIBLE_DEVICES'] ='1,2,3,4'
path = '/home/lll/Alice/code_classify/my_code'
weight_path = '/home/lll/Alice/code_classify/my_code/checkpoint/resnet50/2023-05-31-14:44/best_weight_tumor_all.pth'
model_name = 'resnet50'
dataa = 'TCGA_train'
threshold = 'tumor'
image = '/home/lll/Alice/code_classify/my_code/data/'
csv = '/home/lll/Alice/code_classify/my_code/csv/'

image_path = os.path.join(image,dataa,threshold)
csv_path = os.path.join(csv,dataa, f'{threshold}.csv')
def read_info():
    df = pd.read_csv(csv_path)
    return df

def train(model, epochs, batch_size):
    log_txt = os.path.join(checkpoint_path, '{net}.txt')
    val_best_acc = 0.0
    count = 0
    train_loss_history = []
    val_loss_history = []

    # obtain dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # defining the optimizer
    # optimizer = Adam(model.parameters(), lr=0.0001)
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay=  5e-3)  # 实现L2正则化
    # defining the loss function
    criterion = nn.CrossEntropyLoss()

    # checking if GPU is availabel
    if torch.cuda.is_available():
        model = model.to("cuda")
        criterion = criterion.to("cuda")

    # training
    for epoch in range(1, epochs + 1):
        training_loss = []
        real_labels, pred_labels = [], []
        for data, label in train_loader:
            real_labels += label.detach().tolist()
            data = data.to('cuda')
            label = label.to('cuda')
            pred = model(data)
            # Add dropout to the output layer
            pred = nn.Dropout(p=0.5)(pred)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record batch loss
            training_loss.append(loss.item())
            # for evaluation
            out = F.softmax(pred, dim=1).cpu().detach().numpy()
            pred_label = np.argmax(out, 1)
            pred_labels += list(pred_label)

        # evaluation for train
        train_acc = metrics.accuracy_score(real_labels, pred_labels)
        # train_acc = accuracy(real_labels, pred_labels)
        train_loss = np.mean(training_loss)
        train_loss_history.append(train_loss.item())
        # evaluation for val
        val_loss = []
        real_labels, pred_labels = [], []
        predicted_list = []
        labels_list = []
        model.eval()
        with torch.no_grad():
            for data, label in val_loader:
                real_labels += label.detach().tolist()
                data = data.to('cuda')
                label = label.to('cuda')
                pred = model(data)
                loss = criterion(pred, label)
                val_loss.append(loss.item())
                out = F.softmax(pred, dim=1).cpu().detach().numpy()
                pred_label = np.argmax(out, 1)
                labels_list += label.tolist()
                predicted_list += pred_label.tolist()
                pred_labels += list(pred_label)
        val_acc = metrics.accuracy_score(real_labels, pred_labels)
        val_loss = np.mean(val_loss)
        val_loss_history.append(val_loss)
        if val_acc >= val_best_acc:
            val_best_acc = val_acc
            weights_path = f'{checkpoint_path}/best_weight_{threshold}_{dataa}.pth'
            # print('saving weights file to {}'.format(weights_path))
            torch.save(model.state_dict(), weights_path)

        else:
            count += 1
        early_stopping(val_loss,model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        print('val_best_acc', val_best_acc)
        # 计算fpr和tpr
        fpr, tpr, thresholds = roc_curve(real_labels, pred_labels)
        # 计算AUC曲线值
        auc = roc_auc_score(real_labels, pred_labels)
        # 输出混淆矩阵
        tn, fp, fn, tp = confusion_matrix(labels_list, predicted_list).ravel()
        val_acc = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp + 1e-8)
        sensitivity = tp / (tp + fn + 1e-8)
        precision = tp / (tp +fp + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        npv = tn / (tn + fn + 1e-8)
        a_1 = log_txt.format(net=args.net)
        with open(os.path.join(a_1), 'a') as f:
            f.writelines('[epoch %d] train_loss: %.4f  train_acc: %.4f val_loss: %.4f val_accuracy: %.4f  precision: %.4f specificity: %.4f sensitivity: %.4f npv: %.4f ppv: %.4f' % (
                epoch + 1, train_loss, train_acc, val_loss, val_acc, precision,specificity, sensitivity, npv, ppv) + '\n')
        print("Epoch {:3d} | training loss {:5f} | training acc {:5f} | val loss {:5f} | val acc {:5f}".format(
            epoch, train_loss, train_acc, val_loss, val_acc))
    return model, train_loss_history,val_loss_history

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', default=model_name, type=str, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-batch_size', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0015, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--epochs', type=int, default=5000, help='the number of training epoch')
    parser.add_argument('--img_path',  default= image_path,help='the number of training epoch')
    args = parser.parse_args()
    dir_path = os.path.realpath(__file__)
    path = os.path.dirname(dir_path)
    model = get_network(args)
    model.load_state_dict(torch.load(weight_path))
    df = read_info()
    patience = 10  # 在模型20论训练没有降低，停止训练
    early_stopping = EarlyStopping(patience, verbose=True)
    # 加载训练图像
    images = read_images(df,args.img_path)
    print("Total images number: ", len(images))

    train_x, val_x, train_y, val_y = data_split(images, df)
    # 用于图像转换
    data_transforms = {'train': transforms.Compose([
        transforms.ToTensor(),  # 因为我们用skimage读入图片，图片数据为np.array格式。因此，需要首先转换为tensor。
        transforms.Resize(256),  # 裁剪224 x 224 的尺寸
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # 对每个通道做normalize，条件与VGG预训练模型一致。
    ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),  # 裁剪中间的224x224区域
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])}

    # 定义数据集和数据加载器
    train_dataset, val_dataset = data_convert(train_x, train_y, val_x, val_y, data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,num_workers = 4)
    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    log_txt = os.path.join(checkpoint_path, '{net}.txt')
    if not os.path.exists(checkpoint_path):
        folder_path1 = os.getcwd()
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(path,checkpoint_path)
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()


    model,train_loss_history,val_loss_history = train(model = model, epochs = args.epochs,batch_size= args.batch_size )

    plt.plot(train_loss_history,color = 'red', label ='train_loss')
    plt.plot(val_loss_history,color = 'blue', label ='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #保存图像并制定dpi
    plt.savefig(f'{checkpoint_path}/loss.png',dpi=300,format= 'png')
    plt.show()








