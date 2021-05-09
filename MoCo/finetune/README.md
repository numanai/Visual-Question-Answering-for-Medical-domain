<div align="center">

# Fine-tuning with ImageCLEF 2020 dataset

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>

</div>


## Introduction
This directory contains the code used for fine-tuning the learned feature representations on VQA-Med task using ImageCLEF 2020 dataset.
``` moco_model.ckpt ``` is a checkpoint file containing the weights of a pretrained MoCo model, this can be substituted with your own model weights.
``` utils.py ``` is a helper function used to clean the text in the dataset.
``` DenseNet121_MoCo_Classifier.ipynb ``` is a Jupyter notebook containing the code for fine-tuning.
The dataset used for fine-tuning is located in Data directory, and you can organize your own data using the following structure:
```
├── Data                 
│   ├── train_2020
│   |   ├── All_QA_Pairs_train_2020.txt
│   |   └── train_images
│   |            
│   ├── val_2020             
│   |   ├── All_QA_Pairs_val_2020.txt
│   |   └── val_images
│   └──
└──
```
