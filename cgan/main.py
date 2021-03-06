from parameter import get_parameters
from utils import *
from data_loader import Data_Loader
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    config = get_parameters()
    print(config)

    data_loader = Data_Loader(config.image_path, config.image_size, config.batch_size, True)

    config.save_dir = get_save_dir(config.save_dir, config.name)
    log = get_logger(config.save_dir, config.name)

    # Set random seed
    log.info(f'Using random seed {config.seed}...')
    set_all_seeds(config.seed)

    make_folder(config.save_dir, 'log')
    make_folder(config.save_dir, 'model')
    make_folder(config.save_dir, 'samples')
    # make_folder(config.save_dir, 'attn')

    trainer = Trainer(data_loader.loader(), config)
    trainer.train()

    # if config.train:
    #     if config.model == 'sagan':
    #         trainer = Trainer(data_loader.loader(), config)
    #     elif config.model == 'qgan':
    #         trainer = qgan_trainer(data_loader.loader(), config)
    #     trainer.train()
    # else:
    #     tester = Tester(data_loader.loader(), config)
    #     tester.test()

