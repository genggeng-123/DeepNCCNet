""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.8749971, 0.7552563, 0.8888364)
CIFAR100_TRAIN_STD = (0.14148019, 0.18288018, 0.104028426)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = './checkpoint'

#total training epoches
EPOCH = 200
#MILESTONES = [60, 120, 160, 240]
#MILESTONES = [60, 120, 160, 200, 230,260]
MILESTONES = [60, 120, 160,200,240,280,320]

#initial learning rate
#INIT_LR = 0.1


DATE_FORMAT = '%Y-%m-%d-%H:%M'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








