import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.classes = os.listdir(data_path)
        self.num_classes = len(self.classes)
        self.image_paths = []
        for i, cls in enumerate(self.classes):
            class_path = os.path.join(data_path, cls)
            for image_name in os.listdir(class_path):
                self.image_paths.append((os.path.join(class_path, image_name), i))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path, label = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据路径
TrainImage = r'C:\Users\lemon\Desktop\FL-GAN_COVID-main\CNN_Classifier\KvasirV2_Z-Line\Training'
TestImage = r'C:\Users\lemon\Desktop\FL-GAN_COVID-main\CNN_Classifier\KvasirV2_Z-Line\Testing'

# 图像尺寸和批量大小
image_size = 64  # 数据特征图大小为64x64
BATCH_SIZE = 32
epochs = 10
num_classes = 8

# 数据转换和增强
train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = CustomDataset(TrainImage, transform=train_transforms)
test_dataset = CustomDataset(TestImage, transform=test_transforms)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 自定义适合小尺寸输入的ResNet模型
class ResNet64(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNet64, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.resnet(x)

# 初始化自定义ResNet模型
model = ResNet64(num_classes=num_classes)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练过程中保存的精度和损失
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 训练模型
for epoch in range(epochs):
    model.train()  # 将模型设置为训练模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 清除之前的梯度

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # 计算训练集上的精确度
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_preds / total_preds
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}')

    # 在测试集上评估模型
    model.eval()
    test_running_loss = 0.0
    test_correct_preds = 0
    test_total_preds = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)

        test_running_loss += test_loss.item() * inputs.size(0)

        # 计算测试集上的精确度
        _, preds = torch.max(outputs, 1)
        test_correct_preds += torch.sum(preds == labels).item()
        test_total_preds += labels.size(0)

    test_epoch_loss = test_running_loss / len(test_dataset)
    test_epoch_acc = test_correct_preds / test_total_preds
    test_losses.append(test_epoch_loss)
    test_accuracies.append(test_epoch_acc)
    print(f'Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_acc:.4f}')

# 保存训练好的模型
torch.save(model.state_dict(), r'resnet64_model_resnet64_2000_2000.pth')

# 绘制并保存精度和损失的折线图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(r'training_metrics_resnet64_2000_2000.png')
plt.show()

# 打印最终的分类报告和混淆矩阵
model.eval()
predictions = []
true_labels = []

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    predictions.extend(preds.cpu().numpy())
    true_labels.extend(labels.cpu().numpy())

print(classification_report(true_labels, predictions, target_names=train_dataset.classes, zero_division=1))
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predictions))
