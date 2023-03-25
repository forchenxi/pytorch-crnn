# -*- coding: utf-8 -*-

import lmdb
import sys
import six
import torch
import random
import numpy as np

from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import sampler


# resize和标准化
class resizeNormalize:

    """
    把一个取值范围是[0,255]的PIL.Image 转换成 Tensor，
    shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的Tensor
    """

    def __init__(self, size, interpolation=Image.BILINEAR):  # bilinear
        self.size = size
        self.interpolation = interpolation   # 插值法
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # 先resize
        img = img.resize(self.size, self.interpolation)
        # 转换为tensor，且形状会变为 (c, h, w)
        img = self.toTensor(img)
        # 减去0.5，除以0.5，具体含义未知？
        img.sub_(0.5).div_(0.5)
        return img


# lmdb数据由键值对组成，可以实现将所有键取出来，根据键去读取对应的值
class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        """
        创建lmdb环境,root 指定存放生成的lmdb数据库的文件夹路径，如果没有该文件夹则自动创建。
        会在指定路径下创建 data.mdb 和 lock.mdb 两个文件，一是个数据文件，一个是锁文件
        """

        if not self.env:
            print(f'cannot create lmdb fron {root}')
            sys.exit(0)
        """
        先创建一个事务(transaction) 对象 txn，所有的操作都必须经过这个事务对象。
        """
        with self.env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode()))  # txn.get(key)：进行查询
            self.n_sample = n_samples
        self.transform = transform
        self.target_transform = target_transform

    # 重写len方法
    def __len__(self):
        return self.n_sample

    # 重写get方法
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = f'image-{index}'  # 关键字
            img_buffer = txn.get(img_key.encode())  # get函数通过键值查询数据

            buffer = six.BytesIO()  # 在内存缓冲区中读写数据

            """BytesIO实现了在内存中读写bytes，我们创建一个BytesIO，然后写入一些bytes."""

            buffer.write(img_buffer)
            buffer.seek(0)  # 通过seek先设置"文件指针"的位置

            try:
                # 打开图像，并转换为灰度图
                img = Image.open(buffer).convert('L')
            except IOError:
                print(f'Corrupted image for {index}')
                return self[index+1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = f'label-{index}'
            label = str(txn.get(label_key.encode()))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


# 随机序列样例
class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        # super().__init__(data_source)
        self.num_samples = len(data_source)  # 样本数量
        self.batch_size = batch_size  # 批量尺寸

    def __iter__(self):
        n_batch = len(self) // self.batch_size  # 样本总量/批量尺寸=批量数
        tail = len(self) % self.batch_size  # 尾部（余数）
        # torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
        # fill_ 以特定数值填充tensor
        index = torch.LongTensor(len(self)).fill_(0)
        # 每个批次返回 随机的样本
        for i in range(n_batch):
            random_start = random.randint(0, len(self)-self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i+1)*self.batch_size] = batch_index
        # 最后的剩于批次
        if tail:
            random_start = random.randint(0, len(self)-self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[n_batch*self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


# 整齐校准
class alignCollate:

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w/float(h))
            ratios.sort()  # 宽高比集合
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio*imgH))
            imgW = max(imgH*self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        # unsqueeze()函数 : 在指定位置增加维度
        """torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起
           torch.cat((A,B),0) 按维数0（行）拼接 """
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


env = lmdb.open('data/train')
env.close()
