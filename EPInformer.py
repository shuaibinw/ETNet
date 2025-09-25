
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:00:50 2025

@author: 123
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import copy
import warnings
warnings.filterwarnings('ignore')

# 设置GPU

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class seq_256bp_encoder(nn.Module):
    def __init__(self, base_size=4, out_dim=128, conv_dim=256):
        super(seq_256bp_encoder, self).__init__()
        self.conv_dim = conv_dim
        self.out_dim = out_dim
        self.base_size = base_size
        # cropped_len = 46
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels = base_size, out_channels = self.conv_dim, kernel_size = (1, 8), stride = 1, padding='same'),
            nn.ELU(),
        )
        self.conv_tower = nn.ModuleList([])
        conv_dim = [self.conv_dim, 128, 64, 64, 128]
        for i in range(4):
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels = conv_dim[i], out_channels=conv_dim[i+1], kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(conv_dim[i+1]),
                nn.ELU(),                   
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            ))
            self.conv_tower.append(nn.Sequential(
                nn.Conv2d(in_channels = conv_dim[i+1], out_channels=conv_dim[i+1], kernel_size=(1, 1)),
                nn.ELU(),
            ))
        
    def forward(self, enhancers_input):
        if enhancers_input.shape[2] == 1:
            x_enhancer = enhancers_input
        else:
            x_enhancer = enhancers_input.permute(0, 3, 1, 2).contiguous()  
        x_enhancer = self.stem_conv(x_enhancer)
        for i in range(0, len(self.conv_tower), 2):
            x_enhancer = self.conv_tower[i](x_enhancer)
            x_enhancer = self.conv_tower[i+1](x_enhancer) + x_enhancer
        return x_enhancer

class enhancer_predictor_256bp(nn.Module):
    def __init__(self):
        super(enhancer_predictor_256bp, self).__init__()
        self.encoder = seq_256bp_encoder()
        self.embedToAct = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128*16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )  
    def forward(self, enhancer_seq):
        if len(enhancer_seq.shape) < 4:
            enhancer_seq = enhancer_seq.unsqueeze(2)
        seq_embed = self.encoder(enhancer_seq)
        epi_out = self.embedToAct(seq_embed)
        return epi_out.squeeze(-1)

class MHAttention_encoderLayer(nn.Module):
    def __init__(self, d_model=128, nhead=8, dropout=0.):
        super(MHAttention_encoderLayer, self).__init__()
        # self.activation = activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, 4*d_model) might cause loading problem, this parameter is not neccessary
        # self.linear2 = nn.Linear(4*d_model, d_model) might cause loading problem, this parameter is not neccessary
        # self.dropout = nn.Dropout(dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
    # self-attention block
    def _sa_block(self, x, key_padding_mask, attn_mask):
        x, w = self.self_attn(x, x, x,
                           key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w
        
    def forward(self, x, enhancers_padding_mask=None, attn_mask=None):
        x2 = self.norm1(x)
        x2, attention_w = self._sa_block(x2, key_padding_mask=enhancers_padding_mask, attn_mask=attn_mask)
        x = x2 + x
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x, attention_w

class MHAttention_encoderLayer_noLN(nn.Module):
    def __init__(self, d_model=2048, nhead=8, dim_feedforward=256, dropout=0.1, activation=F.relu):
        super(MHAttention_encoderLayer_noLN, self).__init__()
        self.activation = activation
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
    
    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)
    
    # self-attention block
    def _sa_block(self, x, key_padding_mask, attn_mask):
        x, w = self.self_attn(x, x, x,
                           key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return x, w
    
    def forward(self, x_pe, enhancers_padding_mask=None, attn_mask=None):
        xt, attention_w = self._sa_block(x_pe, enhancers_padding_mask, attn_mask=attn_mask)
        x_pe = x_pe + xt
        x_pe = x_pe + self._ff_block(x_pe)
        return x_pe, attention_w


class EPInformer_v2(nn.Module):
    def __init__(self, base_size = 4, n_encoder=3, out_dim=128, head = 4, pre_trained_encoder= None, n_enhancer=50, device='cuda', useBN=True, usePromoterSignal=True, useFeat=True, n_extraFeat=0, useLN=True):
        super(EPInformer_v2, self).__init__()
        self.n_enhancer = n_enhancer
        self.out_dim = out_dim
        self.useFeat = useFeat
        self.usePromoterSignal = usePromoterSignal
        self.n_extraFeat = n_extraFeat
        self.useBN = useBN
        self.base_size = base_size
        self.useLN = useLN
        if pre_trained_encoder is not None:
            self.seq_encoder = pre_trained_encoder
            self.name = 'EPInformerV2.preTrainedConv.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN, useLN, useFeat, n_extraFeat, n_enhancer) 
        else:
            self.seq_encoder = seq_256bp_encoder(base_size=base_size)
            self.name = 'EPInformerV2.{}base.{}dim.{}Trans.{}head.{}BN.{}LN.{}Feat.{}extraFeat.{}enh'.format(base_size, out_dim, n_encoder, head, useBN,useLN, useFeat, n_extraFeat, n_enhancer)
        self.n_encoder = n_encoder
        self.device = device
        if useLN:
            self.attn_encoder = get_clones(MHAttention_encoderLayer(d_model=out_dim, nhead=head), self.n_encoder)
        else:
            self.attn_encoder = get_clones(MHAttention_encoderLayer_noLN(d_model=out_dim, nhead=head), self.n_encoder)
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.attn_mask = attn_mask
        if self.useBN:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)), 
                 # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        else:
            self.conv_out = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels=64, kernel_size=(1, 3), dilation=(1, 2)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 4)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=(1, 3), dilation=(1, 6)),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=(1, 1)),
                nn.ELU(),
                nn.Linear(101, int(self.out_dim/32)),
                # nn.Linear(38, 8), # 2kb nn.Linear(101, 8)
                nn.ELU(),
            )
        if self.useFeat:
            if self.usePromoterSignal:
                feat_n = 9
            else:
                feat_n = 8
            self.pToExpr = nn.Sequential(
                        nn.Linear(self.out_dim+feat_n, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 1),
                    )
        else:
            self.pToExpr = nn.Sequential(
                    nn.Linear(self.out_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                )
        self.add_pos_conv = nn.Sequential(
                nn.Conv1d(in_channels = self.out_dim+n_extraFeat, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels = self.out_dim, out_channels=self.out_dim, kernel_size=1),
                nn.ReLU(),
        )

    def forward(self, pe_seq, rna_feat=None, extraFeat=None):
        # if enhancers_padding_mask is None:
        enhancers_padding_mask = ~(pe_seq.sum(-1).sum(-1) > 0).bool()
        pe_embed = self.seq_encoder(pe_seq)
        pe_embed = self.conv_out(pe_embed)
        pe_flatten_embed = torch.flatten(pe_embed.permute(0, 2, 1, 3), start_dim=2)
        if extraFeat is not None:
            pe_flatten_embed = self.add_pos_conv(torch.concat([pe_flatten_embed, extraFeat], axis=-1).permute(0,2,1)).permute(0,2,1)
        attn_list = []
        for i in range(self.n_encoder):
            pe_flatten_embed, attn = self.attn_encoder[i](pe_flatten_embed, enhancers_padding_mask=enhancers_padding_mask, attn_mask=self.attn_mask.to(self.device))
            attn_list.append(attn.unsqueeze(0))
        p_embed = torch.flatten(pe_flatten_embed[:,0,:], start_dim=1)
        if self.useFeat:
            p_embed = torch.cat([p_embed, rna_feat], dim=-1)
        p_expr = self.pToExpr(p_embed)
        return p_expr, torch.cat(attn_list)


# 自定义数据集加载
class EnhancerDataset(Dataset):
    def __init__(self, seq1, seq2, labels):
        self.seq1 = torch.FloatTensor(seq1)
        self.seq2 = torch.FloatTensor(seq2)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.seq1[idx], self.seq2[idx], self.labels[idx]


# 修改后的模型 - 使用EPInformer_v2
class EPInformer(nn.Module):
    def __init__(self):
        super(EPInformer, self).__init__()
        # 使用EPInformer_v2作为基础模型
        self.epinformer1 = EPInformer_v2(
            base_size=4,
            n_encoder=3,
            out_dim=128,
            head=4,
            n_enhancer=1,  # 只有一个增强子
            device=device,
            useBN=True,
            usePromoterSignal=False,
            useFeat=False,  # 不使用额外特征
            n_extraFeat=0,
            useLN=True
        )
        
        self.epinformer2 = EPInformer_v2(
            base_size=4,
            n_encoder=3,
            out_dim=128,
            head=4,
            n_enhancer=1,  
            device=device,
            useBN=True,
            usePromoterSignal=False,
            useFeat=False,  
            n_extraFeat=0,
            useLN=True
        )
        
        # 合并两个EPInformer的输出
        self.merge_layer = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, seq1, seq2):
       
        
        batch_size = seq1.shape[0]
        
        seq1_reshaped = seq1.view(batch_size, 1, 2000, 4)  # [batch, 1, 2000, 4]
        promoter1 = torch.zeros(batch_size, 1, 2000, 4).to(device)
        seq1_input = torch.cat([promoter1, seq1_reshaped], dim=1)  # [batch, 2, 2000, 4]
        
      
        seq2_reshaped = seq2.view(batch_size, 1, 2000, 4)  # [batch, 1, 2000, 4]
        promoter2 = torch.zeros(batch_size, 1, 2000, 4).to(device)
        seq2_input = torch.cat([promoter2, seq2_reshaped], dim=1)  # [batch, 2, 2000, 4]
        
        # 通过EPInformer处理
        out1, _ = self.epinformer1(seq1_input)  
        out2, _ = self.epinformer2(seq2_input)  
        
        # 合并两个输出
        combined = torch.cat([out1, out2], dim=1)
        output = self.merge_layer(combined)
        
        return output


# 训练函数
def training(model, epochs=50, batch_size=100, learning_rate=0.0001):
    print('Loading data...')
    SEQ_LEN = 2000
    enhancer_shape = (-1, 2000, 4)

    # 加载数据
    seq1 = np.load('enhancer1_B.npz')
    seq2 = np.load('enhancer2_B.npz')

    # 准备数据
    label = seq1['label'].shape[0]
    np.random.seed(label)
    rand_index = np.arange(label)
    np.random.shuffle(rand_index)
    labels = seq1['label'][rand_index]
    seq1_data = seq1['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]
    seq2_data = seq2['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]
    
    # 分割训练集和验证集
    val_split = 0.1
    val_size = int(len(labels) * val_split)
    
    train_seq1 = seq1_data[:-val_size]
    train_seq2 = seq2_data[:-val_size]
    train_labels = labels[:-val_size]
    
    val_seq1 = seq1_data[-val_size:]
    val_seq2 = seq2_data[-val_size:]
    val_labels = labels[-val_size:]
    
    # 创建数据加载器
    train_dataset = EnhancerDataset(train_seq1, train_seq2, train_labels)
    val_dataset = EnhancerDataset(val_seq1, val_seq2, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    best_val_acc = 0
    model_path = 'EPInformer.pth'
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        
        for seq1_batch, seq2_batch, labels_batch in train_loader:
            seq1_batch, seq2_batch, labels_batch = seq1_batch.to(device), seq2_batch.to(device), labels_batch.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(seq1_batch, seq2_batch)
            loss = criterion(outputs, labels_batch.unsqueeze(1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels_batch.unsqueeze(1)).sum().item()
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for seq1_batch, seq2_batch, labels_batch in val_loader:
                seq1_batch, seq2_batch, labels_batch = seq1_batch.to(device), seq2_batch.to(device), labels_batch.to(device)
                
                outputs = model(seq1_batch, seq2_batch)
                loss = criterion(outputs, labels_batch.unsqueeze(1))
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels_batch.unsqueeze(1)).sum().item()
                
                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(labels_batch.cpu().numpy())
        
        train_acc = train_correct / len(train_dataset)
        val_acc = val_correct / len(val_dataset)
        
        # 计算F1分数
        val_pred_labels = (np.array(val_preds) > 0.5).astype(int).flatten()
        val_f1 = f1_score(val_true, val_pred_labels)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'Model saved with validation accuracy: {val_acc:.4f}')


def load_test_data():
    print('Loading test data...')
    SEQ_LEN = 2000
    enhancer_shape = (-1, 2000, 4)

    # 加载测试数据
    seq1_test = np.load('enhancer1_C.npz')  # 测试数据文件
    seq2_test = np.load('enhancer2_C.npz')  # 测试数据文件

    # 提取数据和标签
    test_labels = seq1_test['label']
    seq1_test = seq1_test['sequence'].astype('float32').reshape(enhancer_shape)
    seq2_test = seq2_test['sequence'].astype('float32').reshape(enhancer_shape)

    return seq1_test, seq2_test, test_labels

# 修改测试模型函数，增加precision和recall计算，并保存ROC和PRC曲线数据
def test_model(model, model_path):
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载测试数据
    seq1_test, seq2_test, test_labels = load_test_data()
    
    # 转换为PyTorch张量
    seq1_test = torch.FloatTensor(seq1_test).to(device)
    seq2_test = torch.FloatTensor(seq2_test).to(device)
    
    # 批量预测以避免内存问题
    batch_size = 100
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(seq1_test), batch_size):
            batch_seq1 = seq1_test[i:i+batch_size]
            batch_seq2 = seq2_test[i:i+batch_size]
            
            batch_preds = model(batch_seq1, batch_seq2)
            predictions.extend(batch_preds.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    predicted_labels = (predictions > 0.5).astype(int)  # 将概率转换为二分类标签

    # 计算评估指标
    from sklearn.metrics import precision_score, recall_score, precision_recall_curve
    
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1_value = f1_score(test_labels, predicted_labels)
    precision_value = precision_score(test_labels, predicted_labels)
    recall_value = recall_score(test_labels, predicted_labels)
    auc_value = roc_auc_score(test_labels, predictions)
    aupr = average_precision_score(test_labels, predictions)

    # 打印结果
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1_value:.4f}")
    print(f"Test Precision: {precision_value:.4f}")
    print(f"Test Recall: {recall_value:.4f}")
    print(f"Test AUC: {auc_value:.4f}")
    print(f"Test AUPR: {aupr:.4f}")
    
    # 创建输出目录
    os.makedirs('EPInformer_Results', exist_ok=True)
    
    # 计算 ROC 曲线数据
    fpr, tpr, roc_thresholds = roc_curve(test_labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    # 保存 ROC 曲线数据到文件
    with open('Feature2/EPInformer_ROC.txt', 'w') as f:
        f.write(f"ROC Curve Data\n")
        f.write(f"=============\n\n")
        f.write(f"ROC AUC Score: {roc_auc:.6f}\n\n")
        f.write(f"X-axis: False Positive Rate (FPR)\n")
        f.write(f"Y-axis: True Positive Rate (TPR)\n\n")
        f.write(f"FPR\tTPR\tThreshold\n")
        f.write(f"--------------------\n")
        for i in range(len(fpr)):
            if i < len(roc_thresholds):
                f.write(f"{fpr[i]:.6f}\t{tpr[i]:.6f}\t{roc_thresholds[i]:.6f}\n")
            else:
                f.write(f"{fpr[i]:.6f}\t{tpr[i]:.6f}\tN/A\n")
    
    # 计算 PR 曲线数据
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test_labels, predictions)
    
    # 保存 PR 曲线数据到文件
    with open('Feature2/EPInformer_PRC.txt', 'w') as f:
        f.write(f"Precision-Recall Curve Data\n")
        f.write(f"==========================\n\n")
        f.write(f"PR AUC Score (AUPR): {aupr:.6f}\n\n")
        f.write(f"X-axis: Recall\n")
        f.write(f"Y-axis: Precision\n\n")
        f.write(f"Recall\tPrecision\tThreshold\n")
        f.write(f"--------------------\n")
        for i in range(len(recall_curve)):
            if i < len(pr_thresholds):
                f.write(f"{recall_curve[i]:.6f}\t{precision_curve[i]:.6f}\t{pr_thresholds[i]:.6f}\n")
            else:
                f.write(f"{recall_curve[i]:.6f}\t{precision_curve[i]:.6f}\tN/A\n")
    
    # 将评估指标保存到文件
    with open('Feature2/EPInformer_metrics.txt', 'w') as f:
        f.write(f"Model Evaluation Metrics\n")
        f.write(f"======================\n\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"F1 Score: {f1_value:.6f}\n")
        f.write(f"Precision: {precision_value:.6f}\n")
        f.write(f"Recall: {recall_value:.6f}\n")
        f.write(f"AUC: {auc_value:.6f}\n")
        f.write(f"AUPR: {aupr:.6f}\n")
    
    # 保存 ROC 和 PR 曲线数据为 CSV 格式，方便后续处理
    # ROC 曲线数据 - 只保存 FPR 和 TPR，移除阈值列避免长度不一致问题
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    })
    roc_df.to_csv('Feature2/EPInformer_ROC.csv', index=False)
    
    # PR 曲线数据 - 只保存 Recall 和 Precision，移除阈值列避免长度不一致问题
    pr_df = pd.DataFrame({
        'Recall': recall_curve,
        'Precision': precision_curve
    })
    pr_df.to_csv('Feature2/EPInformer_PRC.csv', index=False)
    
   



# 主程序
if __name__ == "__main__":
    model = EnhancerMDLF().to(device)
    print(model)  # 打印模型结构
    training(model)
    model_path = 'EPInformer.pth'  # 训练脚本保存的模型路径
    test_model(model, model_path)

