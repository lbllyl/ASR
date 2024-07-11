import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CrohnsDataset
from torchvision import transforms
from preprocess import preprocess_data
from model import ASR, aap, feature_max
import os
from sklearn.metrics import accuracy_score, roc_auc_score

def loss_function(output, target, loss_fuc=None):
    m = output.shape[0]  # 样本数量
    num_classes = output.shape[1]  # 类别数量

    beta = num_classes / m
    
    # 计算 p(k|xi),即模型预测的概率
    p_k_xi = torch.softmax(output, dim=1)
    
    # 创建 I(ti = k) 的掩码矩阵
    mask = torch.zeros_like(output)
    mask[torch.arange(m), target] = 1
    
    # 计算 L_OC
    if loss_fuc == 'oc':
        L = -1/m * torch.sum(mask * beta * torch.log(p_k_xi) + (1 - mask) * torch.log(1 - p_k_xi))
    else:
        L = -1/m * torch.sum(mask * torch.log(p_k_xi) + (1 - mask) * torch.log(1 - p_k_xi))
    
    return L

def parse_args():
    parser = argparse.ArgumentParser(description='Crohn\'s Disease Classification')
    parser.add_argument('--data_dir', type=str, default='data_processed', help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training (should be 1 for patient-wise input)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads for data loading')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the trained model')
    parser.add_argument('--image_per_model', type=int, default=3, help='image per model')
    return parser.parse_args()

def main(args):
    # 定义数据转换
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集实例
    dataset = CrohnsDataset(data_dir=args.data_dir, transform=data_transforms)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # 创建模型实例
    model = ASR()

    # 定义损失函数和优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    alpha = 0.25
    gamma = 0.25

    # 训练模型
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_outputs = []
        for images, labels in dataloader:
            # 确保输入的图像维度为 (batch_size, 12, channels, height, width)
            assert images.shape[:2] == (args.batch_size, 12)
            
            labels = labels.view(-1)  # 调整标签的形状
            
            # 将图像分为4组,每组3张图像
            image_groups = images.view(args.batch_size, 4, 3, 3, 224, 224)

            # 对每个图像组进行处理
            output_1, output_2, output_3, output_4 = model(image_groups)
            loss_1 = loss_function(output_1, labels)
            loss_2 = loss_function(output_2, labels)
            loss_3 = loss_function(output_3, labels)
            loss_4 = loss_function(output_4, labels, 'oc')

            loss = alpha * loss_4 + gamma * (loss_1 + loss_2 + loss_3)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 记录标签和输出用于计算准确率和AUC
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(output_4.detach().cpu().numpy())
        
        # 计算准确率和AUC
        all_outputs = torch.tensor(all_outputs)
        _, predicted = torch.max(all_outputs, 1)
        accuracy = accuracy_score(all_labels, predicted.cpu().numpy())
        auc = roc_auc_score(all_labels, all_outputs[:, 1].cpu().numpy())

        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss/(len(dataloader)*4):.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}')

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists('data_processed'):
        print("数据预处理中...")
        preprocess_data('data', 'data_processed')

    main(args)