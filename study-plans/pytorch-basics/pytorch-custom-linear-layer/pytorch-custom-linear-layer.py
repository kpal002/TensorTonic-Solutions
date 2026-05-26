import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Returns: y = x W^T + b without using nn.Linear
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(

            torch.empty(out_features)

        )

        # initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_features

        bound = 1 / math.sqrt(fan_in)

        nn.init.uniform_(self.bias, -bound, bound)
        

    def forward(self, x):

        return x @ self.weight.t() + self.bias
