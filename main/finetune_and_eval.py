import sys
import os
import pickle

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config_init
from frame import Learner

# # 临时添加，为保证散点图的可复现性，设置全局随机种子
# import numpy as np
# import random
# import torch
# def set_seed(seed):
#     print('manual seed:', seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def SL_fintune(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    roc_datas, prc_datas = [], []

    # ToDo 两种的kmers的更改
    if config.model == 'FusionDNAbert':
        config.kmers = [3, 6] # dual-scale
        # config.kmers = [3, 4, 5, 6] # four-scale

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    if config.do_eval:
        learner.test_model() # 评估模型性能
    else:
        learner.train_model()

# Use the dataset for the specified downstream task to fine-tune the model
def select_dataset(dataset):
    if "4mC" in dataset:
        path_train_data = os.path.join('../data/DNA_MS/tsv/4mC', dataset, 'train.tsv')
        path_test_data = os.path.join('../data/DNA_MS/tsv/4mC', dataset, 'test.tsv')
        print("train" + path_train_data, "test" + path_test_data)
    elif "5hmC" in dataset:
        path_train_data = os.path.join('../data/DNA_MS/tsv/5hmC', dataset, 'train.tsv')
        path_test_data = os.path.join('../data/DNA_MS/tsv/5hmC', dataset, 'test.tsv')
        print("train" + path_train_data, "test" + path_test_data)
    elif "5hmC" in dataset:
        path_train_data = os.path.join('../data/DNA_MS/tsv/6mA', dataset, 'train.tsv')
        path_test_data = os.path.join('../data/DNA_MS/tsv/6mA', dataset, 'test.tsv')
        print("train" + path_train_data, "test" + path_test_data)
    else:
        print("Please input the correct dataset.")

    return path_train_data, path_test_data


if __name__ == '__main__':
    # set_seed(10)  # 临时添加，为保证散点图的可复现性，设置全局随机种子
    config = config_init.get_config()
    config.path_train_data, config.path_test_data = select_dataset(config.dataset)
    SL_fintune(config)
