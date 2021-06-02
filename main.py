import click
import torch
import logging
import random
import numpy as np

from config import Config
from model import Model
from datasets.main import load_dataset, implemented_datasets
from networks.main import implemented_networks

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('dataset_name', type=click.Choice(list(implemented_datasets())))
@click.argument('net_name', type=click.Choice(list(implemented_networks())))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))

@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'multi-class']), default='multi-class',
              help='Specify Deep SVDD objective ("one-class" or "multi-class").')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=0, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for network training.')
@click.option('--lr', type=float, default=0.0001,
              help='Initial learning rate for network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
@click.option('--lr_milestone', type=int, default=[10,20,30,40], multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6, help='Weight decay (L2 penalty).') #1e-6, 
@click.option('--l1', type=float, default=1e-6, help='L1 penalty to loss.')
@click.option('--l2', type=float, default=1e-6, help='L2 penalty to loss.')
@click.option('--n_jobs_dataloader', type=int, default=5,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--visdom', flag_value=True, default=True, help='Enable visdom output (Needs visdom server)')
@click.option('--valid_epoch', type=int, default=-1, help='Epoch for validation.')
@click.option('--restore_best', flag_value=True, default=False, help='Restore best model based on validation.')

def main(dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, l1, l2,
         n_jobs_dataloader, visdom, valid_epoch, restore_best):
    """
    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    
    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)
    
    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        # for torch 1.4.0
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        cfg.settings['device'] = 'cpu'
    logger.info('Computation device: %s' % cfg.settings['device'])
    logger.info('Number of dataloader workers: %d' % cfg.settings['n_jobs_dataloader'])

    # Load data
    dataset = load_dataset(dataset_name, data_path)
    
    logger.info('Dataset type: %s' % dataset)
    logger.info('Dataset class count: %s' % dataset.n_classes)
    logger.info('Dataset dir: %s' % dataset.root)
    logger.info('Dataset train set len: %s' % len(dataset.train_set))
    if hasattr(dataset, "validation_set"):
        logger.info('Dataset validation set len: %s' % len(dataset.validation_set))
    logger.info('Dataset test set len: %s' % len(dataset.test_set))

    # Initialize model and set neural network \phi
    model = Model(cfg.settings['objective'])
    model.set_network(net_name)
    # If specified, load model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        Model.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    model.train(dataset,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    lbd1=cfg.settings['l1'],
                    lbd2=cfg.settings['l2'],
                    device=cfg.settings['device'],
                    n_jobs_dataloader=cfg.settings['n_jobs_dataloader'],
                    enable_vis=cfg.settings['visdom'],
                    valid_epoch=cfg.settings['valid_epoch'],
                    restore_best=cfg.settings['restore_best'])

    # Test model
    model.test(dataset, device=cfg.settings['device'], n_jobs_dataloader=cfg.settings['n_jobs_dataloader'])

    # Save results, model, and configuration
    model.save_results(export_json=xp_path + '/results.json')
    model.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')
    if cfg.settings['visdom']:
        model.trainer.create_log_at(xp_path + '/visdom.log', 'example')
    logging.shutdown()
    logger.handlers.clear()


if __name__ == '__main__':
    main()
