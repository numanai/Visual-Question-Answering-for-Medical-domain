
import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.classification import Accuracy
from torch.optim import Optimizer

from src.architectures.barlow_net import Barlow_arch


class BarlowTwins(pl.LightningModule):
    """

    BarlowTwins network main class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = Barlow_arch(hparams=self.hparams)

        # loss function
        self.criterion = BarlowTwinsLoss(lambda_param= self.hparams['lambda_param'])
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        #self.val_accuracy = Accuracy()
        #self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            #"val/acc": [],
            "train/loss": [],
            #"val/loss": [],
        }

    def forward(self, x0, x1) -> torch.Tensor:
        return self.model(x0, x1)


    def training_step(self, batch, batch_idx: int):
        
        
        x = batch # The batch from datamodule returns a dictionary which contains images, label and metadata
        img0 = x['image'][0] # First batch of images with random augmentation
        img1 = x['image'][1] # Second batch of same images with random augmentation



        z1, z2 = self.forward(img0, img1)
        loss = self.criterion(z1, z2)

        # log train metrics to your loggers!
        #acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        #self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss


    def configure_optimizers(self) :
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        """
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.hparams['max_epochs'])

        # if we want to use the scheduler defined above, then we need to return it together with optim. 


        return optim


#Source https://docs.lightly.ai/
class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, lambda_param=3.9e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

        gpus = 1 if torch.cuda.is_available() else 0
        device = 'cuda' if gpus else 'cpu'

        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss