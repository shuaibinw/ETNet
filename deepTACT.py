#!/usr/bin/env python
import os
import sys
import keras
import datetime
import numpy as np
# import hickle as hkl
from sklearn import metrics
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.layers import LSTM, Bidirectional, Concatenate
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Reshape, Permute
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.models import load_model
from keras.layers import Layer, InputSpec
from keras import initializers
from sklearn.metrics import precision_score, recall_score, roc_curve, precision_recall_curve, auc as sklearn_auc
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Attention GRU network
class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),  
                                 initializer=self.init,
                                 trainable=True)
        super(AttLayer, self).build(input_shape)

    def call(self, x, mask=None):
        M = K.tanh(x)  
        # 计算注意力分数
        alpha = K.dot(M, self.W)  
        alpha = K.squeeze(alpha, axis=-1) 

        # 计算注意力权重
        ai = K.exp(alpha)
        weights = ai / K.sum(ai, axis=1, keepdims=True)  
        weighted_input = x * K.expand_dims(weights, axis=-1)  
        return K.tanh(K.sum(weighted_input, axis=1))  

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1]) 


def model(SEQ_LEN):
    # 输入层
    enhancer_input = Input(shape=(1, 4, SEQ_LEN))
    promoter_input = Input(shape=(1, 4, SEQ_LEN))
    
    # CNN层处理增强子序列
    enhancer_conv = Conv2D(1024, (4, 10), activation='relu', padding='valid', data_format='channels_first')(enhancer_input)
    enhancer_pool = MaxPooling2D(pool_size=(1, 20), data_format='channels_first')(enhancer_conv)
    enhancer_flat = Reshape((1024, -1))(enhancer_pool)
    
    # CNN层处理启动子序列
    promoter_conv = Conv2D(1024, (4, 10), activation='relu', padding='valid', data_format='channels_first')(promoter_input)
    promoter_pool = MaxPooling2D(pool_size=(1, 20), data_format='channels_first')(promoter_conv)
    promoter_flat = Reshape((1024, -1))(promoter_pool)
    
    merged = Concatenate(axis=2)([enhancer_flat, promoter_flat])
    
    permuted = Permute((2, 1))(merged)
    norm1 = BatchNormalization()(permuted)
    drop1 = Dropout(0.5)(norm1)
    
    # 双向LSTM层
    bilstm = Bidirectional(LSTM(100, return_sequences=True))(drop1)
    
    # 注意力层
    att_layer = AttLayer()(bilstm)
    
    # 批归一化和Dropout
    norm2 = BatchNormalization()(att_layer)
    drop2 = Dropout(0.5)(norm2)
    
    # 全连接层
    dense1 = Dense(925, activation='relu')(drop2)
    norm3 = BatchNormalization()(dense1)
    drop3 = Dropout(0.5)(norm3)
    
    output = Dense(1, activation='sigmoid')(drop3)
    
    model = Model(inputs=[enhancer_input, promoter_input], outputs=output)
    
    return model


def f1(y_true, y_pred):
    TP = K.sum(K.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), 'float32'))
    FP = K.sum(K.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), 'float32'))
    FN = K.sum(K.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), 'float32'))
    TN = K.sum(K.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0), 'float32'))

    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())
    return F1

def training(model):
    print('Loading data...')
    SEQ_LEN = 2000
    enhancer_shape = (-1, 1, 4, SEQ_LEN)

    # Load data
    seq1 = np.load('enhancer1_B.npz')
    seq2 = np.load('enhancer2_B.npz')

    # Prepare data
    label = seq1['label'].shape[0]
    np.random.seed(label)
    rand_index = np.arange(label)
    np.random.shuffle(rand_index)
    label = seq1['label'][rand_index]
    seq1 = seq1['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]
    seq2 = seq2['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]

    # Train model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy', f1])
    filename = 'deepTACT.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor='val_accuracy', save_best_only=True, mode='max')
    model.fit([seq1, seq2], label, epochs=50, batch_size=100,
              validation_split=0.1, callbacks=[modelCheckpoint])

def load_test_data():
    print('Loading test data...')
    SEQ_LEN = 2000
    enhancer_shape = (-1, 1, 4, SEQ_LEN)

    # 加载测试数据
    seq1_test = np.load('enhancer1_C.npz')  # 测试数据文件
    seq2_test = np.load('enhancer2_C.npz')  # 测试数据文件

    # 提取数据和标签
    test_labels = seq1_test['label']
    seq1_test = seq1_test['sequence'].astype('float32').reshape(enhancer_shape)
    seq2_test = seq2_test['sequence'].astype('float32').reshape(enhancer_shape)

    return [seq1_test, seq2_test], test_labels


def test_model(model_path):
    # 加载模型
    model.load_weights(model_path)
    # model = load_model(model_path, custom_objects={'f1': f1, 'precision': precision, 'recall': recall})

    # 加载测试数据
    test_data, test_labels = load_test_data()

    # 进行预测
    predictions = model.predict(test_data)
    predicted_labels = (predictions > 0.5).astype(int).flatten()  # 将概率转换为二分类标签

    # 计算评估指标
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1_value = f1_score(test_labels, predicted_labels)
    precision_value = precision_score(test_labels, predicted_labels)  # 计算精确率
    recall_value = recall_score(test_labels, predicted_labels)  # 计算召回率
    roc_auc_value = roc_auc_score(test_labels, predictions)
    aupr = average_precision_score(test_labels, predictions)

    # 打印结果
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1_value:.4f}")
    print(f"Test Precision: {precision_value:.4f}")  # 打印精确率
    print(f"Test Recall: {recall_value:.4f}")  # 打印召回率
    print(f"Test AUC: {roc_auc_value:.4f}")
    print(f"Test AUPR: {aupr:.4f}")
    
    # 计算 ROC 曲线数据
    fpr, tpr, roc_thresholds = roc_curve(test_labels, predictions)
    roc_auc = sklearn_auc(fpr, tpr)
    
    # 保存 ROC 曲线数据到文件
    with open('Feature2/deepTACT_ROC.txt', 'w') as f:
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
    with open('Feature2/deepTACT_PRC.txt', 'w') as f:
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
    with open('Feature2/deepTACT_metrics.txt', 'w') as f:
        f.write(f"Model Evaluation Metrics\n")
        f.write(f"======================\n\n")
        f.write(f"Accuracy: {accuracy:.6f}\n")
        f.write(f"F1 Score: {f1_value:.6f}\n")
        f.write(f"Precision: {precision_value:.6f}\n")
        f.write(f"Recall: {recall_value:.6f}\n")
        f.write(f"AUC: {roc_auc_value:.6f}\n")
        f.write(f"AUPR: {aupr:.6f}\n")
    
 
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr
    })
    roc_df.to_csv('Feature2/deepTACT_ROC.csv', index=False)
    
   
    pr_df = pd.DataFrame({
        'Recall': recall_curve,
        'Precision': precision_curve
    })
    pr_df.to_csv('Feature2/deepTACT_PRC.csv', index=False)
   


""" MAIN """
if __name__ == "__main__":
    SEQ_LEN = 2000
    model = model(SEQ_LEN)
    
    model_path = 'deepTACT.h5'  # 训练脚本保存的模型路径
    test_model(model_path)
