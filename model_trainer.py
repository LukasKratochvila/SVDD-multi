from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet

from torch.utils.data.dataloader import DataLoader

#from sklearn.metrics import roc_auc_score
from metric import top_k_accuracy_score
#from sklearn.metrics import accuracy_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

import copy

import visdom
import json

class ModelTrainer(BaseTrainer):

    def __init__(self, objective, c, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, 
                 lbd1: float = 0, lbd2: float = 0, device: str = 'cuda', n_jobs_dataloader: int = 0,
                 enable_vis: bool = True, vis_title: str = 'Model on dataset',
                 valid_epoch: int = -1, restore_best: bool = False):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'multi-class'), "Objective must be either 'one-class' or 'multi-class'."
        self.objective = objective
        
        self.c = torch.tensor(c, device=self.device) if c is not None else None

        # Regularization
        self.lbd1 = lbd1
        self.lbd2 = lbd2
        
        # Validation
        self.valid_epoch = valid_epoch
        self.restore_best = restore_best 
        
        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        
        # Visdom
        self.enable_vis = enable_vis
        self.vis_title = vis_title
        if self.enable_vis:
            self.viz = visdom.Visdom(env='example')

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        if hasattr(dataset, "validation_set"):
            validation_loader = dataset.val_loader(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
            best_acc = 0
            
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net, dataset.n_classes, eps=0.00001)
            
            if self.objective == 'multi-class':
                for j in range(dataset.n_classes):
                    l=list()
                    for i in range(dataset.n_classes):
                        if i != j:
                            s=sum((self.c[...,j]-self.c[...,i]) ** 2)
                            l.append(s) 
                    l=np.array(l)
                    i=np.argmin(l)
                    if i >= j:
                        i += 1
                    logger.info('Minimal dist for {:d} class to {:d} class: {:.5f}'.format(j, i, np.min(l)))
            
            logger.info('Center c initialized.')
            
       
            if self.enable_vis:
                #self.vis_legend = ['Total Loss']
                #self.iter_plot = self.create_vis_plot('Epoch[-]', 'Loss[-]', 'Model training loss', self.vis_legend)
                #self.val_plot = self.create_vis_plot('Epoch[-]', 'Loss[-]', 'Model validation loss', self.vis_legend)
                self.epoch_plot = self.create_vis_plot('Epoch[-]', 'Loss[-]',
                                                       self.vis_title, list(('Training','Validation')))
    
                #self.iter_acc_plot = self.create_vis_plot('Epoch[-]', 'Average Top1 accuracy[-]', 'Model Top1 accuracy', self.vis_legend)
                #self.val_acc_plot = self.create_vis_plot('Epoch[-]', 'Average Top1 accuracy[-]', 'Model Top1 accuracy', self.vis_legend)
                self.epoch_acc_plot = self.create_vis_plot('Epoch[-]', 'Average Top1 accuracy[-]',
                                                           self.vis_title, list(('Training','Validation')))    

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        
        for epoch in range(self.n_epochs):
            net.train()
            loss_epoch = 0.0
            n_batches = 0
            idx_label_score = []
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, labels, _ = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()
                
                outputs = net(inputs)
                # dist = (outputs - labels) ** 2 + self.lbd1 * abs(s) + self.lbd2 * pow(s,2) # simple classification
                if self.objective == 'one-class':
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                    loss = torch.mean(dist)
                else:
                    dist = torch.sum((outputs.unsqueeze(-1).repeat(1,1,dataset.n_classes) - self.c) ** 2, dim=1)
                    loss = torch.sum(dist.gather(1, labels.view(-1,1)))
                    #loss = torch.sum(torch.mean(dist, dim=0))
                    #loss = torch.sum(torch.mean(dist[labels != torch.argmin(dist,dim=1)], dim=0))
                
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1
                
                idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                            dist.cpu().data.numpy().tolist()))
            
            labels, scores = zip(*idx_label_score)
            #labels = np.array(np.argmax(labels,1))
            labels = np.array(labels)
            scores = np.array(scores)
            
            #train_acc = accuracy_score(labels, np.argmin(scores, axis=1))
            train_acc=top_k_accuracy_score(labels, scores, k=1)
            #self.update_vis_plot(epoch*len(train_loader), [loss.item()], self.iter_plot, 'append')
            #self.update_vis_plot(epoch*len(train_loader), [train_acc], self.iter_acc_plot, 'append')
            
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))
            
            # validation
            if hasattr(dataset, "validation_set") and epoch % self.valid_epoch == 0:
                val_start_time = time.time()
                idx_label_score = []
                loss_val = 0.0
                net.eval()
                with torch.no_grad():
                    for data in validation_loader:
                        inputs, labels, idx = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
            
                        outputs = net(inputs)
                        if self.objective == 'one-class':
                            dist = torch.sum((outputs - self.c) ** 2, dim=1)
                        else:
                            dist = torch.sum((outputs.unsqueeze(-1).repeat(1,1,dataset.n_classes)-self.c) ** 2, dim=1)
                        
                        loss_val += torch.sum(dist.gather(1, labels.view(-1,1))).item()
                        #loss_val += torch.mean(torch.sum(dist, dim=0)).item()
                        #loss_val += torch.sum(torch.mean(dist[labels != torch.argmin(dist,dim=1)], dim=0))
                        
                        idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                    labels.cpu().data.numpy().tolist(),
                                                    dist.cpu().data.numpy().tolist()))

                val_train_time = time.time() - val_start_time
                _, labels, scores = zip(*idx_label_score)
                #labels = np.array(np.argmax(labels,1))
                labels = np.array(labels)
                scores = np.array(scores)

                #val_acc = accuracy_score(labels, np.argmin(scores, axis=1))
                val_acc = top_k_accuracy_score(labels,scores,k=1)
                # if get better results save it
                if val_acc > best_acc and self.restore_best:
                    net_dict = copy.deepcopy(net.state_dict())
                    best_acc = val_acc
                    ep = epoch + 1
        
                # log epoch statistics
                logger.info('  Epoch {}/{}\t Time: {:.3f}\t Top1 on val_dataset: {:.2f}%'
                        .format(epoch + 1, self.n_epochs, val_train_time, val_acc*100.))
                
                #self.update_vis_plot(epoch*len(train_loader), [loss_val.item()], self.val_plot, 'append')
                #self.update_vis_plot(epoch*len(train_loader), [val_acc], self.val_acc_plot, 'append')

            # lr scheduler
            scheduler.step()
            if epoch+1 in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]*0.1))
                
            if (epoch+1)%5 == 0:
                logger.info('Center c Updating... ')
                self.c = self.update_center_c(train_loader, net, dataset.n_classes, step=0.1)
                if self.objective == 'multi-class':
                    for j in range(dataset.n_classes):
                        l=list()
                        for i in range(dataset.n_classes):
                            if i != j:
                                s=sum((self.c[...,j]-self.c[...,i]) ** 2)
                                l.append(s) 
                        l=np.array(l)
                        i=np.argmin(l)
                        if i >= j:
                            i += 1
                        logger.info('Minimal dist for {:d} class to {:d} class: {:.5f}'.format(j, i, np.min(l)))
                logger.info('Center c Updated. ')
                
            if self.enable_vis:
                if hasattr(dataset, "validation_set"):
                    self.update_vis_plot(epoch, [loss_epoch,loss_val], self.epoch_plot, 'append')
                    self.update_vis_plot(epoch, [train_acc,val_acc], self.epoch_acc_plot, 'append')
                else:
                    self.update_vis_plot(epoch, [loss_epoch,0], self.epoch_plot, 'append')
                    self.update_vis_plot(epoch, [train_acc,0], self.epoch_acc_plot, 'append')

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        
        if hasattr(dataset, "validation_set") and self.restore_best:
            logger.info('Restore best weights from Epoch {}/{} Top1: {:.2f}'.format(ep,self.n_epochs,best_acc*100.))
            state_dict = net.state_dict()
            state_dict.update(net_dict)
            net.load_state_dict(state_dict)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        points = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = net(inputs)
                if self.objective == 'one-class':
                    dist = torch.sum((outputs - self.c) ** 2, dim=1)
                else:
                    dist = torch.sum((outputs.unsqueeze(-1).repeat(1,1,dataset.n_classes)-self.c) ** 2, dim=1)

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            dist.cpu().data.numpy().tolist()))
                points += list(outputs.cpu().data.numpy().tolist())
                
        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels_l, scores = zip(*idx_label_score)
        #labels = np.array(np.argmax(labels,1))
        labels = np.array(labels_l)
        scores = np.array(scores)

        #self.test_acc = accuracy_score(labels, np.argmin(scores, axis=1))
        self.test_acc = top_k_accuracy_score(labels, scores,k=1)
        logger.info('Test set Top1: {:.2f}%'.format(100. * self.test_acc))
        
        # Plot points
        # points += self.c.cpu().data.numpy().tolist()
        # label_l += torch.ones(self.c.shape[])
        outputs = torch.Tensor(points)
        #torch.Tensor(self.c)[0:3].transpose(0,1).cpu()
        
        if self.enable_vis:
            logger.info('Plotting points...')
            for idx in range(0,min(6,self.c.shape[0]),3):
                self.viz.scatter(X=outputs[..., idx:idx+3].cpu(),
                                 Y=torch.Tensor(labels+1).cpu(),
                                 opts=dict(xlabel='Feature {}'.format(idx+1),
                                           ylabel='Feature {}'.format(idx+2),
                                           zlabel='Feature {}'.format(idx+3),
                                           title='Points',
                                           markersize=3))
                self.viz.scatter(X=torch.Tensor(self.c)[idx:idx+3].transpose(0,1).cpu(),
                                 opts=dict(xlabel='Feature {}'.format(idx+1),
                                           ylabel='Feature {}'.format(idx+2),
                                           zlabel='Feature {}'.format(idx+3),
                                           title='Centers',
                                           markersize=7,
                                           markersymbol='cross'))

        logger.info('Finished testing.')
        
    def init_center_c(self, train_loader: DataLoader, net: BaseNet, n_classes: int, eps: float=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        if self.objective == 'one-class':
            n_samples = 0
            c = torch.zeros(net.rep_dim, device=self.device)
        else:
            n_samples = torch.zeros(n_classes)
            c = torch.zeros(net.rep_dim, n_classes, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, labels, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                if self.objective == 'one-class':
                    n_samples += outputs.shape[0]
                    c += torch.sum(outputs, dim=0)
                else:
                    for i in range(n_classes):
                        n_samples[i] += outputs[labels == i].shape[0]
                        c[..., i] += torch.sum(outputs[labels == i], dim=0)

        if self.objective == 'one-class':        
            c /= n_samples
        else:
            for i in range(n_classes):
                c[..., i] /= n_samples[i]

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    def update_center_c(self, train_loader: DataLoader, net: BaseNet, n_classes: int, step: float=0.1):
        """Update hypersphere center c by forward pass on the data."""
        # n_samples = torch.zeros(n_classes)
        # c = torch.zeros(net.rep_dim, n_classes, device=self.device)

        # net.eval()
        # with torch.no_grad():
        #     for data in train_loader:
        #         # get the inputs of the batch
        #         inputs, labels, _ = data
        #         inputs = inputs.to(self.device)
        #         outputs = net(inputs)
        #         for i in range(n_classes):
        #             n_samples[i] += outputs[labels == i].shape[0]
        #             c[..., i] += torch.sum(outputs[labels == i], dim=0)
        # for i in range(n_classes):
        #     c[..., i] /= n_samples[i]    
        # diff = self.c-c 
        # return self.c + step * diff
        diff = torch.Tensor(self.c.shape[0],n_classes,n_classes)
        for j in range(n_classes):
            for i in range(n_classes):
                    diff[...,j,i]=self.c[...,j]-self.c[...,i]
        avg = torch.mean(diff, dim=2)
        return self.c + step * torch.Tensor(avg)
        
    def create_vis_plot(self, _xlabel, _ylabel, _title, _legend):
        return self.viz.line(X=torch.zeros((1,len(_legend))).cpu(),
                             Y=torch.zeros((1,len(_legend))).cpu(),
                             opts=dict(xlabel=_xlabel,
                                       ylabel=_ylabel,
                                       title=_title,
                                       legend=_legend))


    def update_vis_plot(self, iteration, loss, window1, update_type, epoch_size=1):
        self.viz.line(X=torch.ones((1,len(loss))).cpu() * iteration,
                 Y=torch.Tensor(loss).unsqueeze(0).cpu() / epoch_size,
                 win=window1,
                 update=update_type)
        
    def create_log_at(self, file_path, current_env, new_env=None):
        """https://github.com/theevann/visdom-save/blob/master/vis.py"""
        
        new_env = current_env if new_env is None else new_env
        vis = visdom.Visdom(env=current_env)

        data = json.loads(vis.get_window_data())
        if len(data) == 0:
            print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
            return

        file = open(file_path, 'w+')
        for datapoint in data.values():
            output = {
                'win': datapoint['id'],
                'eid': new_env,
                'opts': {}
                }

            if datapoint['type'] != "plot":
                output['data'] = [{'content': datapoint['content'], 'type': datapoint['type']}]
                if datapoint['height'] is not None:
                    output['opts']['height'] = datapoint['height']
                    if datapoint['width'] is not None:
                        output['opts']['width'] = datapoint['width']
            else:
                output['data'] = datapoint['content']["data"]
                output['layout'] = datapoint['content']["layout"]

            to_write = json.dumps(["events", output])
            file.write(to_write + '\n')
        file.close()
        vis.delete_env(env=current_env)
        
