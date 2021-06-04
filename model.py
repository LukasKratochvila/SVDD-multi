import json
import torch

from base.base_dataset import BaseADDataset
from networks.main import build_network
from model_trainer import ModelTrainer


class Model(object):
    """A class for the Model method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str = 'one-class'):
        """Inits Model with one of the two objectives."""

        assert objective in ('one-class', 'multi-class'), "Objective must be either 'one-class' or 'multi-class'."
        self.objective = objective
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

    def set_network(self, net_name):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, lbd1: float = 0,
              lbd2: float = 0, device: str = 'cuda', n_jobs_dataloader: int = 0, enable_vis: bool = False,
              valid_epoch: int = -1, restore_best: bool = False):
        """Trains the Model on the training data."""
        
        vis_title = 'Model ' + self.net_name + ' on ' + dataset.name + ' dataset'
        
        self.optimizer_name = optimizer_name
        self.trainer = ModelTrainer(self.objective, self.c, optimizer_name, lr=lr, 
                                    n_epochs=n_epochs, lr_milestones=lr_milestones, 
                                    batch_size=batch_size, weight_decay=weight_decay, lbd1=lbd1, lbd2=lbd2,
                                    device=device, n_jobs_dataloader=n_jobs_dataloader, 
                                    enable_vis=enable_vis, vis_title=vis_title,
                                    valid_epoch=valid_epoch, restore_best=restore_best)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0, batch_size: int = 128):
        """Tests the Model on the test data."""

        if self.trainer is None:
            self.trainer = ModelTrainer(self.objective, self.c, 
                                        device=device, n_jobs_dataloader=n_jobs_dataloader, batch_size=batch_size)

        self.trainer.test(dataset, self.net)
        # Get results
        self.results['test_acc'] = self.trainer.test_acc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def save_model(self, export_model, save_ae=True):
        """Save Model to export_model."""

        net_dict = self.net.state_dict()

        torch.save({'c': self.c,
                    'net_dict': net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)
        
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp) 
