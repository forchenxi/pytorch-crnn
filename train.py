import argparse
import random
import torch
import numpy as np
import os

from torch.backends import cudnn
from torch import optim
from torch.utils import data
from torch.autograd import Variable
from torch.nn import CTCLoss

# from warpctc_pytorch import CTCLoss  # 应该是pytorch已经内置了？

# 自定义模块
import utils
import dataset
import models.crnn as crnn

# 定义一些程序运行时的参数
parser = argparse.ArgumentParser()
# required=True 表示该参数不可省略 这里通过pycharm运行直接设置好参数值，去掉required参数
parser.add_argument('--trainroot', help='path to dataset', default='data/train/lmdb_data')
parser.add_argument('--valroot', help='path to dataset', default='data/val/lmdb_data')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')

# action·- 命令行遇到参数时的动作 只要运行时该变量有传参就将该变量设为True
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='data/crnn.pth', help='path to pretrained model (to continue training)')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')

parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
# interval 间隔
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=35, help='Interval to be val')
parser.add_argument('--saveInterval', type=int, default=5, help='Interval to be save')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, not used by adadelta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# adam是一种优化器
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
# adadelta 是另一种优化器（不依赖于全局学习率）
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
# ratio 比例
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
# 人工种子
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiment')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# CUDA是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题
# cuDNN是用于深度神经网络的GPU加速,基于CUDA
# 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
print(train_dataset is True)
assert train_dataset, 'please set train root'
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=False,  # sampler不为None,shuffle就需要为True
    sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio)
)

test_dataset = dataset.lmdbDataset(
    root=opt.valroot,
    transform=dataset.resizeNormalize((100, 32))
)

n_class = len(opt.alphabet)+1
nc = 1  # channels

converter = utils.strLabelConverter(opt.alphabet)
# criterion = CTCLoss()
criterion = CTCLoss(blank=0, reduction='mean')


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, n_class, opt.nh)
crnn.apply(weights_init)  # apply是nn.Module自带的方法
if opt.pretrained != '':
    print(f'loading pretrained model from {opt.pretrained}')
    crnn.load_state_dict(torch.load(opt.pretrained))
print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgW)
text = torch.IntTensor(opt.batchSize*5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(
        crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
    )
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)]*batch_size))
    cost = criterion(preds.log_softmax(2), text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers)
    )
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds.log_softmax(2), text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # preds = preds.squeeze(2)  # preds此时的shape为(26, 64),应该无需squeeze
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            target = target.replace("b'", "").replace("'", "")
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        gt = gt.replace("b'", "").replace("'", "")
        print(f'{raw_pred} => {pred}, gt:{gt}')

    # accuacy = n_correct / float(max_iter*opt.batchSize)
    accuacy = n_correct / 200  # 验证集共200张图片
    print(f'Test loss: {loss_avg.val()}, accuracy: {accuacy}')


for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print(f'[{epoch}/{opt.nepoch}][{i}/{len(train_loader)}] Loss: {loss_avg.val()}')
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        # if i % opt.saveInterval == 0:
        #     torch.save(crnn.state_dict(), f'{opt.expr_dir}/netCRNN_{epoch}_{i}.pth')

    if epoch % opt.saveInterval == 0:
        torch.save(crnn.state_dict(), f'{opt.expr_dir}/netCRNN_{epoch}.pth')
