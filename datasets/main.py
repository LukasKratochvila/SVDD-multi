from .mydata import MYDATA_Dataset
from .cifar10 import CIFAR10_Dataset
from .mnist import MNIST_Dataset
from .joybuy import JOYBUY_Dataset

def implemented_datasets():
    return ('mnist_one','joybuy','mnist','cifar10','mydata100', 'mydata200', 'mydata300', 'mydata400','mydata100aa','mydata200aa','mydata300aa','mydata400aa','mydata100H', 'mydata300H','mydata100Haa','mydata300Haa')

def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    assert dataset_name in implemented_datasets()

    dataset = None
    
    if dataset_name == 'mnist_one':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class, one_class=True)
        
    if dataset_name == 'joybuy':
        dataset = JOYBUY_Dataset(root=data_path)
    
    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path)
        
    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path)
        
    if dataset_name == 'mydata100':
        dataset = MYDATA_Dataset(root=data_path, version=100, validation=True)
        
    if dataset_name == 'mydata200':
        dataset = MYDATA_Dataset(root=data_path, version=200, validation=True)
    
    if dataset_name == 'mydata300':
        dataset = MYDATA_Dataset(root=data_path, version=300, validation=True)
    
    if dataset_name == 'mydata400':
        dataset = MYDATA_Dataset(root=data_path, version=400, validation=True)

    if dataset_name == 'mydata100aa':
        dataset = MYDATA_Dataset(root=data_path, version=100, validation=True, autoaugment=True)

    if dataset_name == 'mydata200aa':
        dataset = MYDATA_Dataset(root=data_path, version=200, validation=True, autoaugment=True)

    if dataset_name == 'mydata300aa':
        dataset = MYDATA_Dataset(root=data_path, version=300, validation=True, autoaugment=True)
    
    if dataset_name == 'mydata400aa':
        dataset = MYDATA_Dataset(root=data_path, version=400, validation=True, autoaugment=True)
        
    if dataset_name == 'mydata100H':
        dataset = MYDATA_Dataset(root=data_path, version=100, Res=0, validation=True)
    
    if dataset_name == 'mydata300H':
        dataset = MYDATA_Dataset(root=data_path, version=300, Res=0, validation=True)

    if dataset_name == 'mydata100Haa':
        dataset = MYDATA_Dataset(root=data_path, version=100, Res=0, validation=True, autoaugment=True)
    
    if dataset_name == 'mydata300Haa':
        dataset = MYDATA_Dataset(root=data_path, version=300, Res=0, validation=True, autoaugment=True)
        
        
    return dataset
