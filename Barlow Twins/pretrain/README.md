# Barlow Twins Pretraining

This directory contains the code for pretraining models using ImageClEF
and CheXpert datasets by applying the Barlow Twins texhnique. 

The Barlow Twins pretraining technique was described in the following
paper:

[Zbontar, Jure, et al. "Barlow twins: Self-supervised learning via redundancy reduction (Zbontar, J. et al., 2021)](https://arxiv.org/abs/2103.03230)

## Usage

The training can be run using the script in `train.py`. 

Train model with default configuration
```yaml
python train.py
```

You can override any parameter from command line like this
```yaml
python train.py trainer.max_epochs=100 optimizer.lr=0.001 datamodule.dataset = CheXpert
```

Train on GPU
```yaml
python train.py trainer.gpus=1
```
<br>


By default, the script will train for 400 epochs on all available GPUs with a batch size
of 128. The default dataset used is CheXpert which should be downloaded and saved in the
data folder or update the `data dir` path in `config.yaml` file. To pretrain with ImageCLEF
dataset, you can override the dataset parameter from command line like this
```yaml
python train.py trainer.max_epochs=100 optimizer.lr=0.001 datamodule.dataset = ImageCLEF
```
or you can update dataset name in  `barlow_datamodule.yaml`

The script doesn't include any validation loops, but it does track the
contrastive loss via wandb and you'll see the training loss of the contrastive
classifier online. If you want to save wandb logs locally, you can set `offline: True` in
`wandb.yaml' file.

After pretraining, you can copy your model to the `finetune` to test the accuracy
on your downstream task.
