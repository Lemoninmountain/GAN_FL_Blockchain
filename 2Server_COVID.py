# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:59:45 2020

This is the FL-GAN code for COVID-19 data augmentation, as part of the paper publication: 
    "Federated Learning for COVID-19 Detection with Generative Adversarial Networks in Edge Cloud Computing", 
    IEEE Internet of Things Journal, Nov. 2021, Accepted (https://ieeexplore.ieee.org/abstract/document/9580478)
@author: Dinh C. Nguyen 
"""
import pandas as pd 
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
os.getcwd()
from Client_COVID import Client

class Server:
    def __init__(self):
        # print(self.model)
        # self.clients = None
        self.clients= [Client('CovidDataset'),Client('CovidDataset')]
        self.client_index = []
        self.target_round = -1
        self.global_round = 10
        # self.clients_total = 50
        self.clients_total = len(self.clients)
        self.frac = 0.1
        self.selected = 0
        self.loss_d_aver = 0
        self.loss_g_aver = 0
        self.client = Client('CovidDataset')
        self.client = Client('CovidDataset')
        # self.client = Client('X-Ray Image DataSet')
    #     初始化了Server类的各种属性，
    #     例如clients（客户端列表）、
    #     client_index（客户端索引列表）、
    #     global_round（全局训练轮数）、
    #     clients_total（总客户端数量）、
    #     frac（选择客户端的分数）、
    #     loss_d_aver和loss_g_aver（损失的平均值，用于全局汇总）、
    #     client（客户端对象，实例化了Client类）。

    def run(self):
        print('GLobal Federated Learning start: ')
        for epoch in (range(1, self.global_round)):
            # 使用for循环迭代全局轮次 (global_round)，每次迭代代表一个全局训练轮次。
            para_collector_g = []
            para_collector_d = []
            self.selected = self.clients_selection()
            # 通过self.clients_selection()方法选择一组客户端。
            print(self.selected)
            selected_user_length = len(self.selected)

            # weight_d, weight_g, loss_d, loss_g=[], [], [],[]
            # for client in self.selected:
            #     w_d, w_g, l_d, l_g = self.clients[client].client_training([client])
            #     weight_d.extend(w_d)
            #     weight_g.extend(w_g)
            #     loss_d.extend(l_d)
            #     loss_g.extend(l_g)
            # self.loss_d_aver = sum(loss_d) / selected_user_length
            # self.loss_g_aver = sum(loss_g) / selected_user_length
            # print(f'Global epoch: {epoch} \tloss_D: {self.loss_d_aver:.3f} \tloss_G: {self.loss_g_aver:.3f}')
            selected1 = [10, 20]
            weight_d, weight_g, loss_d, loss_g = self.client.client_training(selected1)

            self.loss_d_aver = sum(loss_d) / selected_user_length
            self.loss_g_aver = sum(loss_g) / selected_user_length

            print('Global epoch: {} \tloss_D: {:.3f} \tloss_G: {:.3f}'.format(
                epoch, self.loss_d_aver, self.loss_g_aver))


            para_global_d = self.FedAvg(weight_d)
            para_global_g = self.FedAvg(weight_g)
            for client in self.selected:
                self.clients[client].client_update(para_global_d, para_global_g)
            # 调用self.client.client_training(selected1)
            # 方法，向选定的客户端发送训练请求，获取每个客户端的模型权重(weight_d和weight_g)，
            # 以及损失值(loss_d和loss_g)。


            self.loss_d_aver = sum(loss_d) / selected_user_length
            self.loss_g_aver = sum(loss_g) / selected_user_length
            # 计算选定客户端的平均损失 (loss_d_aver和loss_g_aver)。

            print('Global epoch: {} \tloss_D: {:.3f} \tloss_G: {:.3f}'.format(
               epoch, self.loss_d_aver, self.loss_g_aver))
            # 打印当前全局轮次的损失信息。

            para_global_d = self.FedAvg(weight_d)
            para_global_g = self.FedAvg(weight_g)
            # 使用FedAvg方法对客户端的权重进行全局平均，得到全局的判别器和生成器的权重参数 (para_global_d和para_global_g)。

            self.client.client_update(para_global_d, para_global_g)
            #调用self.client.client_update(para_global_d, para_global_g)方法，将全局平均后的权重发送给客户端，进行模型更新。

    def connect_clients(self):
        client_id = [i for i in range(0, self.clients_total)]
        self.client_index = client_id
        return self.client_index
    # 创建并返回一个包含所有客户端ID的列表 (client_id)
    def clients_selection(self):
        n_clients = max(1, int(self.clients_total * self.frac))
        self.client_index = self.connect_clients()
        training_clients = np.random.choice(self.client_index, n_clients, replace=False)
        return training_clients 
    # 从所有客户端中随机选择一定比例 (frac) 的客户端进行训练，并返回选择的客户端ID列表。.

    def FedAvg(self,weight):
        w_avg = weight[0]
        for key in w_avg:
            for i in range(len(weight)): 
                w_avg[key] = w_avg[key] + weight[i][key]
            w_avg[key] = w_avg[key] / float(len(weight))
        return w_avg
    # 对传入的权重列表 (weight) 进行全局平均，返回平均后的权重。
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    running = Server().run()
    
    
    
    

