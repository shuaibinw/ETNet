
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:00:50 2025

@author: 123
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, Concatenate, BatchNormalization, Bidirectional, GRU, Flatten, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import pyBigWig
import pandas as pd
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization, Bidirectional, Reshape,GRU, Input, Conv1D, MaxPooling1D,Add, LSTM, Flatten,Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
# 设置GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from keras.layers import Bidirectional, LSTM, Attention



from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout, Dense

def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    # Feed-forward network
    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(attention_output + ff_output)

    return ff_output


def Enhancer_MDLF():
    sequence_input1 = Input(shape=(2000, 4))
    sequence_input2 = Input(shape=(2000, 4))

    # 使用L2正则化的卷积层
    x_dna2vec1 = Conv1D(128, kernel_size=9, strides=2, activation='relu', padding='same')(sequence_input1)
    x_dna2vec1 = MaxPooling1D(pool_size=2)(x_dna2vec1)
    x_dna2vec1 = Dropout(0.5)(x_dna2vec1)
    x_dna2vec1 = Flatten()(x_dna2vec1)
    x_dna2vec1 = Dense(1000, activation='relu')(x_dna2vec1)
    x_dna2vec1 = Dropout(0.5)(x_dna2vec1)
  
    
    # 使用L2正则化的卷积层
    x_dna2vec2 = Conv1D(128, kernel_size=9, strides=2, activation='relu', padding='same')(sequence_input2)
    x_dna2vec2 = MaxPooling1D(pool_size=2)(x_dna2vec2)
    x_dna2vec2 = Dropout(0.5)(x_dna2vec2)
    x_dna2vec2 = Flatten()(x_dna2vec2)
    x_dna2vec2 = Dense(1000, activation='relu')(x_dna2vec2)
    x_dna2vec2 = Dropout(0.5)(x_dna2vec2)
    
    merge1 = Concatenate(axis=1)([x_dna2vec1, x_dna2vec2])
    
    # Reshape for Transformer
    merge3 = Reshape((-1, 1000))(merge1)
    
    # Add Transformer block
    merge3 = transformer_block(merge3, num_heads=8, ff_dim=1024)
    
    # Flatten the output of the Transformer
    merge3 = Flatten()(merge3)
    
    merge3 = Dropout(0.5)(merge3)
    output = Dense(1, activation='sigmoid')(merge3)
    
    model = Model([sequence_input1, sequence_input2], output)
    print(model.summary())
    return model

# from tensorflow.keras.layers import Multiply, Add

# def cross_conv1d(input1, input2, filters, kernel_size, strides=1, padding='same'):
#     """
#     交叉卷积操作：对两个输入进行卷积并交互特征。
#     """
#     # 对 input1 和 input2 分别进行卷积
#     conv1 = Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(input1)
#     conv2 = Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(input2)
    
#     # 特征交互：逐元素相乘
#     cross_feature = Multiply()([conv1, conv2])
    
#     # 特征融合：将交互后的特征与原始特征相加
#     output = Add()([conv1, conv2, cross_feature])
    
#     return output

# def Enhancer_MDLF():
#     sequence_input1 = Input(shape=(2000, 4))
#     sequence_input2 = Input(shape=(2000, 4))

#     # 第一层交叉卷积
#     x_cross1 = cross_conv1d(sequence_input1, sequence_input2, filters=128, kernel_size=9, strides=2)
#     x_cross1 = MaxPooling1D(pool_size=2)(x_cross1)
#     x_cross1 = Dropout(0.5)(x_cross1)

#     # 第二层交叉卷积
#     x_cross2 = cross_conv1d(x_cross1, x_cross1, filters=128, kernel_size=9, strides=2)
#     x_cross2 = MaxPooling1D(pool_size=2)(x_cross2)
#     x_cross2 = Dropout(0.5)(x_cross2)

#     # 展平特征
#     x_flatten = Flatten()(x_cross2)

#     # 全连接层
#     x_dense = Dense(1000, activation='relu')(x_flatten)
#     x_dense = Dense(1000, activation='relu')(x_dense)
#     x_dense = Dense(1000, activation='relu')(x_dense)
    
#     x_dense = Reshape((-1, 1000))(x_dense)
    
#     x_dense = transformer_block(x_dense, num_heads=8, ff_dim=1024)
    
#     x_dense = Flatten()(x_dense)  # 添加展平操作
#     # 输出层
#     output = Dense(1, activation='sigmoid')(x_dense)
    

#     # 构建模型
#     model = Model([sequence_input1, sequence_input2], output)
#     print(model.summary())
#     return model


# F1分数计算函数
def f1(y_true, y_pred):
    TP = K.sum(K.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 1), 'float32'))
    FP = K.sum(K.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 1), 'float32'))
    FN = K.sum(K.cast(K.equal(y_true, 1) & K.equal(K.round(y_pred), 0), 'float32'))
    TN = K.sum(K.cast(K.equal(y_true, 0) & K.equal(K.round(y_pred), 0), 'float32'))

    P = TP / (TP + FP + K.epsilon())
    R = TP / (TP + FN + K.epsilon())
    F1 = 2 * P * R / (P + R + K.epsilon())
    return F1

# 训练函数
def training(model):
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
    label = seq1['label'][rand_index]
    seq1 = seq1['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]
    seq2 = seq2['sequence'].astype('float32').reshape(enhancer_shape)[rand_index]
   
    # 训练模型
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy', f1])
    filename = 'k562.h5'
    modelCheckpoint = ModelCheckpoint(filename, monitor='val_accuracy', save_best_only=True, mode='max')
    model.fit([seq1, seq2], label, epochs=50, batch_size=100,
              validation_split=0.1, callbacks=[modelCheckpoint])

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

    return [seq1_test, seq2_test], test_labels


# 测试模型
def test_model(model_path):
    # 加载模型
   
    model.load_weights(model_path)
    # model = load_model(model_path, custom_objects={'f1': f1})

    # 加载测试数据
    test_data, test_labels = load_test_data()

    # 进行预测
    predictions = model.predict(test_data)
    predicted_labels = (predictions > 0.5).astype(int).flatten()  # 将概率转换为二分类标签

    # 计算评估指标
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1_value = f1_score(test_labels, predicted_labels)
    auc = roc_auc_score(test_labels, predictions)
    aupr = average_precision_score(test_labels, predictions)

    # 打印结果
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1_value:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Test AUPR: {aupr:.4f}")


# 主程序
if __name__ == "__main__":
    model = Enhancer_MDLF()
    training(model)
    model_path = 'k562.h5'  # 训练脚本保存的模型路径
    test_model(model_path)

