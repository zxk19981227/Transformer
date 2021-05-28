import pickle
from torch import tensor
from torch.utils.data import Dataset
class Dataload(Dataset):
    def __init__(self,path:str,src:str,trg:str):
        """

        :param path: path to data
        :param src: end name of source data
        :param trg: end name of trg data
        """
        input_lines=pickle.load(open(path+'.'+src,'rb'))
        target_lines=pickle.load(open(path+'.'+trg,'rb'))
        self.src=[tensor([1]+each+[2]) for each in input_lines]
        self.trg=[tensor([1]+each+[2]) for each in target_lines]
        assert len(self.src)==len(self.trg)
    def __getitem__(self, item):
        return self.src[item],self.trg[item]
    def __len__(self):
        return len(self.src)
