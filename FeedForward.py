from torch.nn import Linear,Module,Dropout
from torch.nn.functional import relu
from torch import Tensor
class FeedForward(Module):
    def __init__(self,input_dim:int,inner_dim:int,dropout:float):
        """

        :param input_dim: dimensions of input tensor
        :param inner_dim: dimensions of hidden features in feedforward
        """
        super(FeedForward, self).__init__()
        self.input_linear=Linear(input_dim,inner_dim)
        self.relu=relu
        self.dropout=Dropout(dropout)
        self.output_linear=Linear(inner_dim,input_dim)
    def forward(self,features:Tensor):
        """

        :param features: features to process ,shape:(...,...,input_dim)
        :return: Tensor of shape(...,...,input_dim)
        """
        return self.output_linear(self.dropout(self.relu(self.input_linear(features))))