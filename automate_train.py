from main import main

#experiments = ['baseline', 'BN', 'GC', 'N', 'AA', 'AC', 'R2']
e = 'R2'

data = 'mydata'
net_name = 'my_LeNet'
versions = ['100','200','300','400']
base_dir = 'log/'
data_path = 'data/'
load_config = None
load_model = None
device = 'cuda'
seed = 0
optimizer_name = 'adam'
lr = '0.001'
n_epochs = 100
lr_milestone = [50]
batch_size = 128
weight_decay = 1e-6
n_jobs_dataloader = 5
for v in versions:
    dataset_name=data+v
    xp_path=base_dir+e+v+'/'
    main(dataset_name, net_name, xp_path, data_path, load_config, load_model, device, seed,
         optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, n_jobs_dataloader)

