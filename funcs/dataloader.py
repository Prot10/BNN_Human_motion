from torch.utils.data import Dataset
import torch


from .utils import h36motion3d as datasets
from torch.utils.data import DataLoader

datas = 'h36m' # dataset name
PATH = './data/h3.6m/h3.6m/dataset'
# input_n=10 # number of frames to train on (default=10)
# output_n=25 # number of frames to predict on
# input_dim=3 # dimensions of the input coordinates(default=3)
# skip_rate=1 # # skip rate of frames
# joints_to_consider=22


def load_dataset(input_n=10, output_n=25, skip_rate=1, batch_size=256):
    
    '''Function that returns the two splits of the Human Motion dataset'''
    
    print('Loading Train Dataset...')
    dataset = datasets.Datasets(PATH, input_n, output_n, skip_rate, split=0)
    
    print('Loading Validation Dataset...')
    vald_dataset = datasets.Datasets(PATH, input_n, output_n, skip_rate, split=1)

    print('Loading Test Dataset...')
    test_dataset = datasets.Datasets(PATH, input_n, output_n, skip_rate, split=2)
    
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)#

    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    vald_loader = DataLoader(vald_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    print('>>> Test dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
    return data_loader, vald_loader, test_loader


	
def load_test(actions:list,input_n=10, output_n=25, skip_rate=1, batch_size=256):
    test_dataset = datasets.Datasets(PATH, input_n, output_n, skip_rate, split=2, actions=actions)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True)
    return test_loader


all_actions =  ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
def load_testset(input_n=10, output_n=25, skip_rate=1, batch_size=256):
    actions = all_actions
    ds = {action: datasets.Datasets(PATH, input_n, output_n, skip_rate, split=2, actions=[action]) for action in actions}

    return {a: DataLoader(d, batch_size=batch_size, num_workers=2, shuffle=False, pin_memory=True) for a,d in ds.items()}