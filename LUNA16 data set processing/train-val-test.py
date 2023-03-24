#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 2018/07/11 by DQ

import os
import random  # 随机数包

MainFolder = '/home/dq/CodeProject'
TrainValTestFiles = {'train': 'train.txt', 'val': 'val.txt', 'trainval': 'trainval.txt',
                     'test': 'test.txt'}  # 图片集划分文件集合
TrainR = 0.7  # 用于训练的数据量占比
ValR = 0.2  # 用于验证的数据量占比
PreImNum = 1186  # 数据总量
fileIdLen = 4  # 图片名字字符数量，不够补0占位


def CreateImIdTxt(ImIdS, FilePath):
    if os.path.exists(FilePath):
        os.remove(FilePath)  # 保存的文件夹下有同名的文件先删除
    with open(FilePath, 'w') as FId:
        for ImId in ImIdS:
            ImIdStr = str(ImId).zfill(fileIdLen) + '\n'  # 占位换行
            FId.writelines(ImIdStr)


ImIdSet = range(1, PreImNum + 1)  # 图片名标记从1开始
random.shuffle(ImIdSet)  # 随机打乱这个集合
ImNum = len(ImIdSet)
TrainNum = int(TrainR * ImNum)  # 用于训练的图片数量
ValNum = int(ValR * ImNum)  # 用于验证的图片数量

TrainImId = ImIdSet[:TrainNum - 1]  # 从打乱的集合中抽取前TrainNum个数据
TrainImId.sort()  # 从小到大排序，主要是为了好看
ValImId = ImIdSet[TrainNum:TrainNum + ValNum - 1]  # 从打乱的集合中抽取ValNum个数据
ValImId.sort()
TrainValImId = list(set(TrainImId).union(set(ValImId)))  # train和val集合组合成trainval
TrainValImId.sort()
TestImId = (list(set(ImIdSet).difference(set(TrainValImId))))  # 从总集合中除去trainval就是test
TestImId.sort()
TrainValTestIds = {}  # 把上述集合按字典方式组合在一起
TrainValTestIds['train'] = TrainImId
TrainValTestIds['val'] = ValImId
TrainValTestIds['trainval'] = TrainValImId
TrainValTestIds['test'] = TestImId

for Key, KeyVal in TrainValTestFiles.iteritems():  # 遍历字典产生文件
    ImIdS = TrainValTestIds[Key]
    FileName = TrainValTestFiles[Key]
    FilePath = os.path.join(MainFolder, FileName)
    CreateImIdTxt(ImIdS, FilePath)