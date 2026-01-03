import torch
from torch.utils.data import Dataset,dataloader

class customDataset(Dataset):
    def __init__(self,data_dict,transform=None):
        self.data_dict=data_dict
        self.transform=transform
    def __len__(self):
        return len(self.data_dict['labels'])
    def __getitem__(self,idx):
        image=self.data_dict['data'][idx]
        label=self.data_dict['labels'][idx]
        if self.transform:
            image=self.transform(image)
            #label=label.ToTensor()
        return (image,label)
    
