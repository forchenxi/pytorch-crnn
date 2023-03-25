# -*- coding: utf-8 -*-
import torch
from PIL import Image
from torch.autograd import Variable

import dataset
import utils
from models import crnn


model_path = './data/crnn.pth'  # 预训练模型权重（需自行下载）
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

# classes为什么是len(alphabet)+1,因为后面又添加了字符'-'，可能是用来表示空格
model = crnn.CRNN(32, 1, len(alphabet)+1, 256)
# GPU加速相关
if torch.cuda.is_available():
    model = model.cuda()

print(f'loading pretrained model from {model_path}')

model.load_state_dict(torch.load(model_path))  # 加载模型
converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))

image = Image.open(img_path).convert('L')  # 转化为灰度图

image = transformer(image)  # resize和归一化-->(1, 32, 100)
if torch.cuda.is_available():
    image = image.cuda()

# 相当于是在原来的张量基础上又增加了一维(大概是批次的意思？)
image = image.view(1, *image.size())   # (1, 1, 32, 100)
image = Variable(image)
model.eval()  # 表示网络接下来要开始预测

# 传入网络预测
preds = model(image)  # (26, 1, 37)

# 获取类别概率最大值的索引
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))  # 计算预测得到的长度（通俗理解就是有多少个字符，包含空格）
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print(f'{raw_pred}  {sim_pred}')
