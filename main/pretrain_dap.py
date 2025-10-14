import time
import sys
import os
import pickle

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from configuration import config_init
from frame import Learner


def SL_pretrain(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)

    if config.model == 'FusionDNAbert':
        config.kmers = [3, 6] # dual-scale
        # config.kmers = [3, 4, 5, 6] # four-scale

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()


def select_dataset():
    train_dict = {
        "4mCF": '../data/DNA_MS/tsv/4mC/4mC_F.vesca/train.tsv',
        "4mCS": '../data/DNA_MS/tsv/4mC/4mC_S.cerevisiae/train.tsv',
        "4mCC": '../data/DNA_MS/tsv/4mC/4mC_C.equisetifolia/train.tsv',
        "4mCT": '../data/DNA_MS/tsv/4mC/4mC_Tolypocladium/train.tsv',
        "5hmCH": '../data/DNA_MS/tsv/5hmC/5hmC_H.sapiens/train.tsv',
        "5hmCM": '../data/DNA_MS/tsv/5hmC/5hmC_M.musculus/train.tsv',
        "6mAA": '../data/DNA_MS/tsv/6mA/6mA_A.thaliana/train.tsv',
        "6mACEL": '../data/DNA_MS/tsv/6mA/6mA_C.elegans/train.tsv',
        "6mACEQ": '../data/DNA_MS/tsv/6mA/6mA_C.equisetifolia/train.tsv',
        "6mAD": '../data/DNA_MS/tsv/6mA/6mA_D.melanogaster/train.tsv',
        "6mAF": '../data/DNA_MS/tsv/6mA/6mA_F.vesca/train.tsv',
        "6mAH": '../data/DNA_MS/tsv/6mA/6mA_H.sapiens/train.tsv',
        "6mAR": '../data/DNA_MS/tsv/6mA/6mA_R.chinensis/train.tsv',
        "6mAS": '../data/DNA_MS/tsv/6mA/6mA_S.cerevisiae/train.tsv',
        "6mATT": '../data/DNA_MS/tsv/6mA/6mA_T.thermophile/train.tsv',
        "6mATO": '../data/DNA_MS/tsv/6mA/6mA_Tolypocladium/train.tsv',
        "6mAX": '../data/DNA_MS/tsv/6mA/6mA_Xoc BLS256/train.tsv',
    }

    # print("train" + path_train_data, "test" + path_test_data)
    # return path_train_data, path_test_data
    return train_dict


if __name__ == '__main__':
    # start_time = time.time()
    config = config_init.get_config()
    config.path_train_data = select_dataset()
    SL_pretrain(config)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"Total pretraining time: {total_time/3600:.2f} h")
