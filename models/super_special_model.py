import os.path
import sys
from os import path
from model_impl.HART import HartModel, HartClassificationModel
from torch import nn

# layers = [module for module in model.modules()]

print('testin')

class HartClassificationModelSparCL(nn.Module):
    def __init__(self, n_classes:int, input_shape=(128,6), mlp_head_units = [1024], **kwargs ):

        super(HartClassificationModelSparCL, self).__init__(**kwargs)
        self.hart_core = HartModel(input_shape,mlp_head_units=mlp_head_units, **kwargs)
        self.logits = nn.Linear(in_features=mlp_head_units[-1], out_features=n_classes)

    def forward(self, x):
        out_1 = self.hart_core(x.float())
        return self.logits(out_1)

def generate_irregular_sparsity(model, dest, filename, sparsity):
    names = []
    for i, val in enumerate(model.named_parameters()):
        if "hart_core" in val[0] and "weight" in val[0] and "norm" not in val[0] and i > 7:
            names.append(val[0])
    print(names)
    output_to_yaml(names, dest,filename, sparsity)

def output_to_yaml(layers, dest, filename, sparsity:float):
    with open(os.path.join(dest,filename), 'a+') as f:
        # print to the top
        f.write("prune_ratios: \n")
        for layer in layers:
            f.write(f"  {layer}:\n    {sparsity}\n")
        f.write("rho:\n  0.001")

if __name__ == '__main__':
    model = HartClassificationModelSparCL(6)
    generate_irregular_sparsity(model, '../profiles/hart/irr', 'hart_0.75.yaml', 0.75)