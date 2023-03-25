# -*- coding: utf-8 -*-4
from torch import nn
from PIL import Image
from torch.autograd import Variable

import dataset


# (双向)LSTM 长短期记忆网络，RNN网络的一种
class BidirectionalLSTM(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.embedding = nn.Linear(n_hidden*2, n_out)

    def forward(self, n_input):
        recurrent, _ = self.rnn(n_input)
        t, b, h = recurrent.size()
        t_rec = recurrent.view(t*b, h)

        output = self.embedding(t_rec)
        output = output.view(t, b, -1)

        return output


# 构建CRNN模型
class CRNN(nn.Module):
    def __init__(self, img_h, nc, n_class, nh, n_rnn=2, leakyRelu=False):
        """
        :param img_h: 输入图片的图片高度
        :param nc: 初始输入图片的channels
        :param n_class: 多少个分类（共有多少不同的字符）
        :param nh: n_hidden RNN中的隐藏层的过渡size
        :param n_rnn:　实际没有用到，推测是要设置的rnn层数
        :param leakyRelu: 控制激活函数使用relu还是leakyrelu
        """
        super(CRNN, self).__init__()
        assert img_h % 16 == 0, 'img_h has to be a multiple of 16'

        # 模型共7层卷积，4层池化
        ks = [3, 3, 3, 3, 3, 3, 2]  # kernel
        ps = [1, 1, 1, 1, 1, 1, 0]  # padding
        ss = [1, 1, 1, 1, 1, 1, 1]  # stride
        nm = [64, 128, 256, 256, 512, 512, 512]  # channel

        cnn = nn.Sequential()

        def conv_relu(i, batch_normalization=False):
            # batchNormalization用于确定是否需要BN（批规范化）
            n_in = nc if i == 0 else nm[i-1]
            n_out = nm[i]
            # 每调用一次，添加一层卷积
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            # 根据参数确定是否添加批规范化层
            if batch_normalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(n_out))
            # 根据参数确定激活函数
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # conv+relu+max_pooling
        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(kernel_size=2, stride=2))  # 64x16x80(假设输入高度为16，宽度为160)

        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(kernel_size=2, stride=2))  # 128x8x40

        conv_relu(2, True)

        conv_relu(3)  # 256x8x40
        # 最大池化 window：1x2, s:2
        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)))  # 256x4x40
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

        # conv+bn+relu
        conv_relu(4, True)

        conv_relu(5)  # 512x4x40

        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)))  # 512x2x40
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))

        conv_relu(6, True)  # 512x1x39
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, n_class)
        )

    def forward(self, n_input):
        # conv features
        conv = self.cnn(n_input)
        # b批量大小，c channel，h高度，w 宽度
        b, c, h, w = conv.size()
        # 这里限制了输入h必须为32
        assert h == 1, "the height of conv must be 1"
        # squeeze()函数删除单维度条目
        conv = conv.squeeze(2)
        # 将tensor的维度换位
        conv = conv.permute(2, 0, 1)

        # run features
        output = self.rnn(conv)
        return output


# image = Image.open('../data/demo.png').convert('L')
#
# transformer = dataset.resizeNormalize((160, 32))
#
# image = transformer(image)
# image = image.view(1, *image.size())
# image = Variable(image)
# model = CRNN(32, 1, 37, 256)
#
# preds = model(image)
# print(preds)
# print(preds.size())
