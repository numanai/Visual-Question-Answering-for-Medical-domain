from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
from torch.optim import Optimizer

from src.architectures.moco_net import MoCo_arch


class MoCo(pl.LightningModule):
    """
    
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = MoCo_arch(hparams=self.hparams)

        """
        # NCE loss function
        def infoNCE_loss(q, k, queue):
            
            t = 0.05 # temperature coefficient

            # batch size of query encoder
            N = q.shape[0]

            # representation vector dimensionality
            C = q.shape[1]

            # we can use bmm for batch matrix multiplication 
            # numerator
            pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C),k.view(N,C,1).view(N,1)),t))

            #denomicator
            neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C),torch.t(queue)),t)), dim = 1)
            

            return torch.mean(-torch.log(torch.div(pos,pos+neg)))
        """

        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metric_hist = {
            "train/acc": [],
            "val/acc": [],
            "train/loss": [],
            "val/loss": [],
        }

    def forward(self, image0, image1):
        return self.model(image0, image1) #these two images are passed to MoCo_arch??

    #What does -> mean?
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        
        image0, image1 = batch["image0"], batch["image1"]

        output, target = self.forward(image0, image1)

        # log train metrics to your loggers!
        loss_val = self.loss_fn(output, target)


        acc = self.train_accuracy(output, target)
        self.log("train/loss", loss_val, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss_val, "preds": output, "targets": target}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers

        """
        optim = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.model.parameters()
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 10)

        return [optim], [scheduler]
    

    # [OPTIONAL METHOD]
    def training_epoch_end(self, outputs: List[Any]) -> None:
        # log best so far train acc and train loss
        self.metric_hist["train/acc"].append(self.trainer.callback_metrics["train/acc"])
        self.metric_hist["train/loss"].append(self.trainer.callback_metrics["train/loss"])
        self.log("train/acc_best", max(self.metric_hist["train/acc"]), prog_bar=False)
        self.log("train/loss_best", min(self.metric_hist["train/loss"]), prog_bar=False)

    # [OPTIONAL METHOD]
    # IF WE HAVE VALIDATION STEP THEN WE CAN USE THIS
    # def validation_epoch_end(self, outputs: List[Any]) -> None:
    #     # log best so far val acc and val loss
    #     self.metric_hist["val/acc"].append(self.trainer.callback_metrics["val/acc"])
    #     self.metric_hist["val/loss"].append(self.trainer.callback_metrics["val/loss"])
    #     self.log("val/acc_best", max(self.metric_hist["val/acc"]), prog_bar=False)
    #     self.log("val/loss_best", min(self.metric_hist["val/loss"]), prog_bar=False)

    
