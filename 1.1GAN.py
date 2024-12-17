# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:45 2020
This is the standalone GAN code for COVID-19 data augmentation, as part of the paper publication: 
    "Federated Learning for COVID-19 Detection with Generative Adversarial Networks in Edge Cloud Computing", 
    IEEE Internet of Things Journal, Nov. 2021, Accepted (https://ieeexplore.ieee.org/abstract/document/9580478)
@author: Dinh C. Nguyen 
"""
import argparse
import datetime
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.display import clear_output
from tqdm import tqdm
import numpy as np
import pandas as pd
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler

now = datetime.datetime.now()

from model_GAN3 import netG, netD

'''
定义用于数据增强的GAN架构
netG定义生成器网络, 生成假的图像
    使用ConvTranspose2d图层将早点上传到图像中
netD定义鉴别器网络,区分真假
    使用conv2d层下采样图像并对其进行分类
两个网络都使用ReLu,tanh和leakyrelu等激活函数, 以及dropout和批处理归一化来实现正则化和稳定化

weights_init初始化网络层的权重
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# data_dir = '4/3AHDbc50wauQLH_Uxb1lOAm9fH5_nZ8o_gvOQGzGEebKG_8SK7yvan4div2k' #Just to check
data_dir = 'DIS+GEN/KvasirV2'  # 'covid_alone_synthetic' #'Pneumonia' #'covid_alone'#数据集的目录

batch_size = 16# 每个批次的样本数

num_epochs = 500# 训练的轮数

imageSize = 64# 图像的大小

nc = 1# 图像通道数

nz = 100# 生成器输入的噪声向量大小

ngf = 64# 生成器中特征图的大小

ndf = 64# 鉴别器中特征图的大小

nb_label = 3 # 类别数

lr = 0.001# 优化器学习率

beta1 = 0.5 # Adam优化器的beta1参数

beta2 = 0.999# Adam优化器的beta2参数

real_label = 1.
fake_label = 0.

fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # 固定的噪声输入，用于生成示例图像

# 定义损失函数
s_criterion = nn.BCELoss().to(device)  # For synthesizing
c_criterion = nn.NLLLoss().to(device)  # For classification

input = torch.FloatTensor(batch_size, 3, imageSize, imageSize).to(device)
# Batch size, 3为通道数, imagesize是图像的高度和宽度
noise = torch.FloatTensor(batch_size, nz, 1, 1).to(device)
# 用于存储生成器的输入噪声。nz 是噪声向量的大小，通常用于生成器网络的输入。
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
# 其中的元素服从均值为 0，标准差为 1 的正态分布。这个噪声用于在训练过程中生成固定的假图像样本。
s_label = torch.FloatTensor(batch_size).to(device)
c_label = torch.LongTensor(batch_size).to(device)
# 创建了两个一维张量 s_label 和 c_label，分别用于存储生成器的损失函数中的二元分类标签和多类分类标签。

# input = Variable(input)
# s_label = Variable(s_label)
# c_label = Variable(c_label)
# noise = Variable(noise)
# fixed_noise = Variable(fixed_noise)
# 用变量来代替张量,但是新版本的pytorch可以直接用张量进行. 因此可以去掉....


fixed_noise_ = np.random.normal(0, 1, (batch_size, nz))
# 使用NumPy生成随机的均值为 0，标准差为 1 的正态分布噪声向量
random_label = np.random.randint(0, nb_label, batch_size)
# 生成随机的类别标签
print('fixed label:{}'.format(random_label))
random_onehot = np.zeros((batch_size, nb_label))
random_onehot[np.arange(batch_size), random_label] = 1
fixed_noise_[np.arange(batch_size), :nb_label] = random_onehot[np.arange(batch_size)]
# random_onehot 是一个大小为 (batch_size, nb_label) 的零矩阵，用于将随机生成的类别标签转换为独热编码。将独热编码的标签放入 fixed_noise_ 中的前 nb_label 列。

fixed_noise_ = (torch.from_numpy(fixed_noise_))
# 将NumPy数组 fixed_noise_ 转换为PyTorch张量
fixed_noise_ = fixed_noise_.resize_(batch_size, nz, 1, 1)
# 调整其大小为 (batch_size, nz, 1, 1)
fixed_noise.data.copy_(fixed_noise_).to(device)
# 将其内容复制到 fixed_noise 变量中

if nc == 1:
    mu = (0.5)
    sigma = (0.5)
    transform = transforms.Compose([  # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        # transforms.Scale(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(mu, sigma)])
elif nc == 3:
    mu = (0.5, 0.5, 0.5)
    sigma = (0.5, 0.5, 0.5)
    # Originally authors used just scaling
    transform = transforms.Compose([  # transforms.RandomHorizontalFlip(),
        transforms.Resize((64, 64)),
        # transforms.Scale(imageSize),
        transforms.ToTensor(),
        transforms.Normalize(mu, sigma)])
else:
    print("Tranformation not defined for this option")
train_set = datasets.ImageFolder(data_dir, transform=transform)
#这部分代码定义了数据预处理的转换方式，包括图像通道数、图像大小、归一化等，并创建了一个数据加载器来加载训练数据集。

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True)

generator = netG(nz, ngf, nc).to(device)
discriminator = netD(ndf, nc, nb_label).to(device)
# 这里创建了生成器（netG）和鉴别器（netD）
# setup optimizer
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
# 分别为生成器和鉴别器定义了Adam优化器。


def test(predict, labels):
    # test函数接受两个参数：predict和labels，分别表示模型预测的结果和真实的标签。
    correct = 0
    # correct变量用于记录预测正确的样本数量。
    pred = predict.data.max(1)[1]
    # predict是模型的预测输出。在这里，.max(1)用于获取每个样本预测的最大值及其索引，[1]表示取索引。因此，pred是一个张量，包含了每个样本预测结果的索引值。
    correct = pred.eq(labels.data).sum().to(device)
    # labels是真实的标签。.eq(labels.data)用于比较预测值和真实标签是否相等，返回一个布尔类型的张量。.sum()计算所有预测正确的样本数量。
    return correct, len(labels.data)
    # correct：预测正确的样本数量。
    # len(labels.data)：总样本数量，即标签张量的长度。


Loss_D = []
Loss_G = []
Accu_D = []
# 记录每个训练步骤或周期中的损失和准确率，以便后续分析和可视化模型的训练过程和性能。

def training():
    for epoch in range(num_epochs + 1):
        # 外层循环迭代次数为num_epochs加1次，用于控制训练的总轮数。
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)
            # 内层循环通过train_loader迭代加载训练数据集中的每个批次数据，img是图像数据，label是对应的标签。
            ###########################
            # (1) Update D network
            ###########################
            # train with real
            discriminator.zero_grad()
            # 清除鉴别器的梯度缓存。
            batch_size = img.size(0)
            input1 = Variable(torch.FloatTensor(batch_size, 3, imageSize, imageSize).to(device))
            # input1用于存储图像数据
            with torch.no_grad():
                input1.resize_(img.size()).copy_(img)
                s_label.resize_(batch_size, 1).fill_(real_label)
                c_label.resize_(batch_size).copy_(label)
            #     s_label和c_label用于存储对应的真实标签。s_label是用于鉴别器的二进制分类标签，c_label是用于鉴别器的多类别分类标签。
            s_output, c_output = discriminator(img)
            s_output = s_output.to(device)
            c_output = c_output.to(device)
            # s_output和c_output是鉴别器对真实图像的输出。
            s_errD_real = nn.BCELoss()(s_output, s_label)
            # s_errD_real计算鉴别器在二进制分类任务上的损失，使用二元交叉熵损失函数。
            c_errD_real = nn.NLLLoss()(c_output, c_label)
            # c_errD_real计算鉴别器在多类别分类任务上的损失，使用负对数似然损失函数。
            errD_real = s_errD_real + c_errD_real
            # errD_real是总的鉴别器损失，是二者之和。
            errD_real.backward()
            # errD_real.backward()用于反向传播，计算梯度。
            D_x = s_output.data.mean()
            # correct, length = test(c_output, c_label)

            # train with fake
            with torch.no_grad():
                noise.resize_(batch_size, nz, 1, 1)
                noise.normal_(0, 1)


            label = np.random.randint(0, nb_label, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            label_onehot = np.zeros((batch_size, nb_label))
            label_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]

            noise_ = (torch.from_numpy(noise_))
            noise_ = noise_.resize_(batch_size, nz, 1, 1)
            noise.data.copy_(noise_)

            c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
            #     生成器的输入noise被随机生成，用于生成假图像。label是用于假图像的随机标签。noise_是生成器输入的扩展，包括随机标签的编码。
            fake = generator(noise)
            s_label.data.fill_(fake_label)
            s_output, c_output = discriminator(fake.detach())
            s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake = c_criterion(c_output, c_label)
            errD_fake = s_errD_fake + c_errD_fake

            errD_fake.backward()
            D_G_z1 = s_output.data.mean()
            errD = s_errD_real + s_errD_fake
            optimizerD.step()
            # generator(noise)生成假图像。
            # s_label设置为假图像的标签。
            # s_output和c_output是鉴别器对假图像的输出。
            # s_errD_fake计算鉴别器在二进制分类任务上的假图像损失。
            # c_errD_fake计算鉴别器在多类别分类任务上的假图像损失。
            # errD_fake是总的鉴别器损失，是二者之和。
            # errD_fake.backward()用于反向传播，计算梯度。
            # optimizerD.step()用于更新鉴别器的参数。

            ###########################
            # (2) Update G network
            ###########################
            generator.zero_grad()
            s_label.data.fill_(real_label)  # fake labels are real for generator cost
            s_output, c_output = discriminator(fake)
            s_errG = s_criterion(s_output, s_label)
            c_errG = c_criterion(c_output, c_label)

            errG = s_errG + c_errG
            errG.backward()
            D_G_z2 = s_output.data.mean()
            optimizerG.step()
            # 清除生成器的梯度缓存。
            # 设置生成器输出的标签为真实标签。
            # s_output和c_output是鉴别器对生成的假图像的输出。
            # s_errG计算生成器在二进制分类任务上的损失。
            # c_errG计算生成器在多类别分类任务上的损失。
            # errG是总的生成器损失，是二者之和。
            # errG.backward()用于反向传播，计算梯度。
            # optimizerG.step()用于更新生成器的参数。

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(train_loader),
                     errD, errG, D_x, D_G_z1, D_G_z2))
            # 在每个训练步骤结束后，打印当前轮次、总轮次、批次编号、总批次数、鉴别器损失、生成器损失以及鉴别器对真实图像和假图像的输出均值。
            if (epoch) % 50 == 0:
                vutils.save_image(img,
                                  '%s/real_samples.png' % './0_output_images', normalize=True)
                # fake = netG(fixed_cat)
                fake = generator(fixed_noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ('./0_output_images', epoch),
                                  normalize=True)
            # 每50个轮次保存一次真实图像和生成的假图像。vutils.save_image用于保存图像到指定路径。

            # do checkpointing        
        # torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join('.', '0_saved_model'), epoch))
        Loss_D.append(errD.item())
        Loss_G.append(errG.item())
        Accu_D.append(D_x.item())
    # 将每个训练步骤结束后的鉴别器损失、生成器损失和鉴别器对真实图像的输出均值记录到Loss_D、Loss_G和Accu_D列表中。
    torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join('COVID', '0_saved_model'), epoch))
    saved_training(Loss_D, Loss_G)
# 每个训练轮次结束后，保存生成器的模型参数到指定路径。
# 调用saved_training函数，传递损失和精度列表，可能用于进一步分析或可视化。

def test_image(model):
    PATH = 'orginal result/0_saved_model/netG_epoch_19.pth'
    # PATH指定了已经保存的生成器模型的路径。torch.load(PATH)加载了模型参数到内存中。
    model.load_state_dict(torch.load(PATH))
    # model.load_state_dict()将加载的模型参数加载到指定的model中。这里假设model是一个预先定义好的生成器模型。
    loop = 10

    fake = model(fixed_noise)
    # 使用加载的生成器模型model对预先定义的fixed_noise输入进行生成。
    # fixed_noise通常是预先定义好的噪声张量，用于生成假图像。
    vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ('./1_output_images', loop), normalize=True)
    #vutils.save_image()用于将生成的假图像fake.data保存为图像文件。
    # '%s/fake_samples_epoch_%03d.png' % ('./1_output_images', loop)指定了保存路径和文件名格式。
    # ./1_output_images是保存假图像的目录路径。
    # fake_samples_epoch_%03d.png是文件名格式，%03d表示使用3位数字来表示轮次。
    # loop是轮次变量，用于指定当前保存的假图像的轮次编号。

def test2(generator, discriminator, num_epochs, loader):
    print('Testing Block.........')
    now = datetime.datetime.now()
    # g_losses = metrics['train.G_losses'][-1]
    # d_losses = metrics['train.D_losses'][-1]
    path = 'COVID/0_output_images'
    try:
        os.mkdir(os.path.join(path))
    except Exception as error:
        print(error)

    # path是用于保存测试结果图像的目录路径。
    # os.mkdir(os.path.join(path))尝试创建目录，如果目录已存在则会捕获异常并打印错误信息。

    real_batch = next(iter(loader))
    # 使用iter(loader)创建一个迭代器，并通过next()函数获取下一个批次的数据。
    # loader是数据加载器，用于加载测试数据集。这里假设它能够按批次提供真实的图像和标签。

    test_img_list = []
    test_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    test_fake = generator(test_noise).detach()
    test_img_list.append(vutils.make_grid(test_fake, padding=2, normalize=True))
    # torch.randn(batch_size, nz, 1, 1, device=device)生成一个随机噪声张量，用于生成假图像。
    # generator(test_noise)使用给定的生成器模型generator生成假图像。
    # vutils.make_grid(test_fake, padding=2, normalize=True)将生成的假图像以网格形式组合，并进行归一化和填充。


    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.subplot(1, 2, 1)
    ax1 = plt.axis("off")
    ax1 = plt.title("Real Images")
    ax1 = plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).to(device), (1, 2, 0)))

    ax2 = plt.subplot(1, 2, 2)
    ax2 = plt.axis("off")
    ax2 = plt.title("Fake Images")
    ax2 = plt.imshow(np.transpose(test_img_list[-1], (1, 2, 0)))
    # ax2 = plt.show()
    # fig.savefig('%s/image_%.3f_%.3f_%d_%s.png' %
    #                (path, g_losses, d_losses, num_epochs, now.strftime("%Y-%m-%d_%H:%M:%S")))


def saved_training(Loss_D1, Loss_G1):
    dict = {'Loss_D': Loss_D1, 'Loss_G': Loss_G1}
    df = pd.DataFrame(dict)
    # saving the dataframe 
    file = 'file5007.csv'
    df.to_csv(file, index=False)
    # df.to_csv(file)
    df = pd.read_csv(file)
    z1 = df['Loss_D']
    z2 = df['Loss_G']
    # plt.plot(z1)
    # plt.plot(z2)
    plt.plot(z1, label='Loss_D')
    plt.plot(z2, label='Loss_G')

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Training Accuracy')
    plt.legend(loc='upper left')
    # bbox_to_anchor=(0.75, 0.95),
    plt.savefig("0_plot/graph1_%s.png" % now.strftime("%H_%M_%S"))
    plt.show()


y = training()

# test_image(generator)
