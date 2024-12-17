import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)

        # Use adaptive pooling to make sure the output size is consistent
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size will always be (1, 1)

        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Apply adaptive pooling
        out = self.adaptive_pool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


# Use the ResNet18 on Cifar-10
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # check gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} GPUs".format(torch.cuda.device_count()))

    # set hyperparameter
    EPOCH = 40
    pre_epoch = 0
    BATCH_SIZE = 64
    LR = 0.01
    image_size = 64
    Num_workers = 16

    # 数据路径
    TrainImage = r'C:\Users\lemon\Desktop\FL-GAN_COVID-main\CNN_Classifier\KvasirV2_Z-Line\Training'
    TestImage = r'C:\Users\lemon\Desktop\FL-GAN_COVID-main\CNN_Classifier\KvasirV2_Z-Line\Testing'


    # prepare dataset and preprocessing
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图片大小
        transforms.RandomCrop(image_size, image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图片大小
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = ImageFolder(root=TrainImage, transform=transform_train)
    testset = ImageFolder(root=TestImage, transform=transform_test)

    # 创建 DataLoader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=Num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=Num_workers)

    # 类别名称
    classes = trainset.classes
    print(f'Classes: {classes}')

    # define ResNet18
    net = ResNet18().to(device)

    # define loss funtion & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    train_losses = []  # 用于记录每个epoch的训练损失
    test_losses = []  # 用于记录每个epoch的测试损失

    train_accuracies = []  # 用于记录训练准确率
    test_accuracies = []  # 用于记录测试准确率

    # train
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        total_train_loss = 0
        correct_train = 0.0
        total_train = 0.0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            print(f'[Epoch: {epoch + 1}, Iter: {i + 1}] Loss: {total_train_loss / (i + 1):.3f}, '
                  f'Acc: {100 * correct_train / total_train:.3f}%')
        # 记录训练损失和准确率
        train_losses.append(total_train_loss / len(trainloader))
        train_accuracies.append(100 * correct_train / total_train)

        # 测试过程
        print('Waiting Test...')
        net.eval()
        true_labels = []
        pred_labels = []
        total_test_loss = 0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

            # 记录测试损失和准确率
            test_losses.append(total_test_loss / len(testloader))
            test_accuracies.append(100 * correct_test / total_test)


            # Print test accuracy as well
            print(f"Test Accuracy for Epoch {epoch + 1}: {100 * correct_test / total_test:.3f}%")



    # 在测试过程结束后，添加以下内容来打印评估报告
    print("\nClassification Report:")
    report = classification_report(true_labels, pred_labels, target_names=classes)
    print(report)


    # Generate confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    # Plot confusion matrix with labels on both axes
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(cmap=plt.cm.Blues, ax=ax)

    # Customizing the display to add labels on both the top and bottom
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45, fontsize=10)
    plt.yticks(ticks=range(len(classes)), labels=classes, fontsize=10)

    plt.title(f"Confusion Matrix for Epoch {epoch + 1}")
    plt.tight_layout()
    plt.show()

    print(f"Test Accuracy for Epoch {epoch + 1}: {100 * correct_test / total_test:.3f}%")

    # 生成损失和准确率的趋势图
    plt.figure(figsize=(12, 5))

    # Training and Testing Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCH), train_losses, label='Train Loss')
    plt.plot(range(EPOCH), test_losses, label='Test Loss')
    plt.title('Training & Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Training and Testing Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCH), train_accuracies, label='Train Accuracy')
    plt.plot(range(EPOCH), test_accuracies, label='Test Accuracy')
    plt.title('Training & Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print('Training has finished, total epoch is %d' % EPOCH)