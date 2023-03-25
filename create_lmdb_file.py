# -*- coding: utf-8 -*-
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import glob
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):   # 在python3环境下运行
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(v) is str:
                txn.put(k.encode(), v.encode())
                continue
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList)), 'len(x) != len(y)'
    nSamples = len(imagePathList)
    print('...................')
    # env = lmdb.open(outputPath, map_size=104857600)  # 最大100MB
    env = lmdb.open(outputPath, map_size=10485760)

    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        # .mdb数据库文件保存了两种数据，一种是图片数据，一种是标签数据，它们各有其key
        imageKey = f'image-{cnt}'
        labelKey = f'label-{cnt}'
        cache[imageKey] = imageBin
        cache[labelKey] = label

        if lexiconList:
            lexiconKey = f'lexicon-{cnt}'
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_text(path):
    with open(path) as f:
        text = f.read()
    text = text.strip()

    return text


if __name__ == '__main__':

    # lmdb 输出目录
    # outputPath = 'data/train/lmdb_data'  # 训练数据
    outputPath = 'data/val/lmdb_data'    # 验证数据

    # 训练图片路径，标签是txt格式，名字跟图片名字要一致，如123.jpg对应标签需要是123.txt
    # input_path = 'data/train/origin_data/*.jpg'
    input_path = 'data/val/origin_data/*.jpg'

    imagePathList = glob.glob(input_path)
    print('------------', len(imagePathList), '------------')
    imgLabelLists = []
    for p in imagePathList:
        try:
            # imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
            imgLabelLists.append((p, p.split('_')[2].replace('.jpg', '')))
        except Exception as _e:
            print(_e)
            continue

    # sort by labelList
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    createDataset(outputPath, imgPaths, txtLists, lexiconList=None, checkValid=True)
