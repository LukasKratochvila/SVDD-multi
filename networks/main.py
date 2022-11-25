from .my_LeNet import MY_LeNet, MY_LeNetBN
from .my_LeNet_HRes import MY_LeNet as MY_LeNet_HRes, MY_LeNet_Autoencoder

def implemented_networks():
    return ('mnist_LeNet','my_LeNet', 'my_LeNetBN', 'my_LeNet_Hres')

def build_network(net_name, rep_dim):
    """Builds the neural network."""

    assert net_name in implemented_networks()

    net = None
    if net_name == 'mnist_LeNet':
        net = MY_LeNetBN(rep_dim=rep_dim, in_channels=1, in_size=28)
        
    if net_name == 'my_LeNet':
        net = MY_LeNet(rep_dim=rep_dim)
        
    if net_name == 'my_LeNetBN':
        net = MY_LeNetBN(rep_dim=rep_dim)
        
    if net_name == 'my_LeNet_Hres':
        net = MY_LeNet_HRes()
        
    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    assert net_name in implemented_networks()

    ae_net = None

    if net_name == 'my_LeNet':
        ae_net = MY_LeNet_Autoencoder()
        
    return ae_net
