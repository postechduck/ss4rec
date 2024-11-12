import argparse
import os
import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from SS4Rec import SS4Rec
# from TiSASRec import TiSASRec
# from mamba4rec import Mamba4Rec
from recbole.model.sequential_recommender import SASRec, GRU4Rec, NARM, BERT4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

import torch
from tqdm import tqdm
from recbole.data.interaction import Interaction
# from utils import data_preparation

def run_baseline(model_class, config_file):

    print('Model class:', model_class)
    print('Config file:', config_file)
    config = Config(model=model_class, config_file_list=[str(config_file)])
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info(train_data.dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = SS4Rec(config, train_data.dataset).to(config['device'])
    logger.info(model)
    

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, show_progress=config["show_progress"]
    )
    
    # trainer.resume_checkpoint('saved/SS4Rec-Jul-18-2024_16-57-16.pth')
    
    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    
if __name__ == '__main__':
    config_list = ['config_sport.yaml', 'config_video.yaml', 'config_ml.yaml', 'config_kuairec_big.yaml', 'config_kuairec_small.yaml']
    # model_list = [SASRec, BERT4Rec, GRU4Rec, NARM, SS4Rec, TiSASRec]
    
    paser = argparse.ArgumentParser(description='Run SS4Rec.')
    paser.add_argument('--config', type=int, required=True, help='Config file to use')

    args = paser.parse_args()
    
    model_class = SS4Rec
    config = config_list[args.config]
    
    config_file = 'config/' + model_class.__name__ + '/' + config
    if not os.path.exists(config_file):
        print('Config file not found:', config_file)
        config_file = 'config/SASRec/' + config
    run_baseline(model_class, config_file)
            