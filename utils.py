# -*- coding: utf-8 -*-

import torch
import os
import random
import torch.nn as nn
from torch.autograd import Variable


class strLabelConverter:
    def __init__(self, alphabet):
        # 添加字符'-',然后转化为字典
        self.alphabet = alphabet + u'-'
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    # encode就是把字符串，转换成数字编码
    def encode(self, text, depth=0):
        length = []
        result = []
        for _str in text:
            # _str = unicode(_str, 'utf-8')  # python3中不存在Unicode
            # _str形如 "b'jdvfl0k'",要去掉b和'
            _str = _str.replace("b'", "").replace("'", '')
            length.append((len(_str)))
            for char in _str:
                index = self.dict[char]
                result.append(index)
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    # 解码操作就是把数组转换成字符输出（如果raw=False，会把连在一起的相同字符去重，仅保留一个）
    def decode(self, t, length, raw=False):
        if length.numel() == 1:  # 通过numel()函数，查看一个张量到底有多少元素
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


# a = '0123456789abcdefghijklmnopqrstuvwxyz'
# converter = strLabelConverter(a)
# print(converter.dict)
#
# s = 'helloworld'  # 字符串中不能有空格，只能有37个类别中的其中之一
# b, c = converter.encode(s)
# print(b)
# print(c)
#
# x = [18, 15, 22, 22, 25, 33, 25, 28, 22, 14]
# # y = torch.IntTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# y1 = torch.IntTensor([10])
# print(converter.decode(x, y1))  # heloworld 把连续两个ll去掉了一个
# print(converter.decode(x, y1, raw=True))  # helloworld


class averager:
    """
    compute average for torch.Variable and torch.Tensor
    """
    def __init__(self):
        self.n_count = 0
        self.sum = 0

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

    def reset(self):
        self.n_count = 0
        self.sum = 0


def loadData(v, data):
    # v.data.resize_(data.size()).copy_(data)
    v.resize_(data.size()).copy_(data)


def prettyPrint(v):
    print(f'Size {str(v.size())}, Type: {v.data.type()}')
    print(f'|Max: {v.max().data[0]} | Min: {v.min().data[0]} | Mean: {v.mean().data[0]}')


def group_data():
    """
    将数据集划分为训练集、验证集和测试集
    :return:
    """
    # 一开始数据全部在train文件夹下，分别转移一部分到val和test中
    images_list = os.listdir('data/train/')
    for i in range(200):
        t = int(random.random() * len(images_list))
        i_path = images_list[t]
        with open('data/train/' + i_path, 'rb') as f1:
            with open('data/val/'+i_path, 'wb') as f2:
                f2.write(f1.read())
                print('copy {} success'.format(i_path))
        os.remove('data/train/' + i_path)
        del images_list[t]


# group_data()
