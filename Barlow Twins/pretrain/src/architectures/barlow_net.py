from torch import nn
import torch
from torchvision import models


class Barlow_arch(nn.Module):

    """
    Attributes:
        backbone:
            Backbone model to extract features from images.
            - resnet50_pre takes a pretrained model on CheXpert dataset and performs unsupervised pretraining again with ImageCLEF dataset
            - otherwise it will select the model arch. mentioned in the "barlow_model.yaml" file. 
        num_features:
            Dimension of the embedding (before the projection head).
        proj_head_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        proj_head_out:
            Dimension of the output (after the projection head).
    """


    def __init__(self, hparams: dict):
        super().__init__()

        if hparams["backbone"] == "resnet50_pre":

            pretrained_dict = torch.load('/home/numansaeed/Documents/ML_Project (Don\'t Delete)/CheXpert Project/Barlow Twins/last.ckpt')["state_dict"]

            state_dict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("model.network."):
                    k = k.replace("model.network.", "")
                    state_dict[k] = v

            self.network = models.resnet50()
            del self.network.fc     

            self.network.load_state_dict(state_dict)    

            self.network.fc =  nn.Sequential(
                nn.Linear(2048, 1000))          

            self.num_features = 2048 #final output dim from conv layers

        
        else:
            self.network = models.__dict__[hparams["backbone"]](pretrained = hparams['pretrained'],zero_init_residual = True)

            self.num_features = 2048 #final output dim from conv layers

        if hasattr(self.network, "fc"):  # ResNet models
                self.network.fc = nn.Identity()

        elif hasattr(self.network, "classifier"):  # Densenet models
                self.network.classifier = nn.Identity()



        self.proj_head_dim = hparams["proj_head_dim"] # projection hidden layer dimension
        self.proj_output_dim = hparams["proj_head_out"] # projection output layer dimension


        self.proj_mlp =  nn.Sequential(
            nn.Linear(self.num_features, self.proj_head_dim, bias = False),
            nn.BatchNorm1d(self.proj_head_dim),
            nn.ReLU(inplace= True),

            nn.Linear(self.proj_head_dim, self.proj_head_dim, bias = False),
            nn.BatchNorm1d(self.proj_head_dim),
            nn.ReLU(inplace = True),

            nn.Linear(self.proj_head_dim, self.proj_output_dim, bias = False)
        )


    def forward(self, x0: torch.Tensor,
                      x1: torch.Tensor = None,
                      return_features: bool = False):
        
        """
        Forward pass through BarlowNetwork.
        Extracts features with the backbone and applies the projection
        head to the output space. If both x0 and x1 are not None, both will be
        passed through the backbone and projection. If x1 is None, only x0 will
        be forwarded.
        Barlow Twins only implement a projection head unlike SimSiam.
        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns:
            The output projection of x0 and (if x1 is not None)
            the output projection of x1. 
        """

        z1 = self.proj_mlp(self.network(x0))
        z2 = self.proj_mlp(self.network(x1))


        return z1, z2
