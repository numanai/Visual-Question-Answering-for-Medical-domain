<div align="center">

# Visual Question Answering for Medical Domain

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>

</div>

## Description
Visual Question Answering (VQA) is a rising interdisciplinary problem that demands the knowledge of both Computer Vision (CV) and Natural Language Processing (NLP). Many domain-specific VQA tasks have emerged in the last few years, and VQA in the medical domain is one such that plays a significant role in providing medical assistance to both doctors and patients. For example, doctors could use VQA model answers as assistance in medical diagnosis, while patients could ask questions from VQA related to their medical images to better understand their physical condition. <br>
A VQA system takes as input an image and a natural language question about the image and produces an answer consistent with the visual content of a given image. To facilitate the research of implementing VQA in medical domain, ImageCLEF, which is part of the Conference and Labs of the Evaluation Forum (CLEF), has been conducting annual VQA-Med challenges since 2018. In this project, we will be using the dataset coming from VQA-Med 2020 challenge for training. As far as our knowledge goes, recent VQA-Med challenge participants have not used self-supervised learning (SSL) techniques and instead have focused on transfer learning, ensemble models, etc. Thus, towards the possible solution of the VQA-Med problem, in this project we implement two contrastive learning frameworks, MoCo and Barlow Twins pretrained on different medical datasets and fine-tuned on VQA-Med 2020 dataset.

This repository contains two folders: Barlow Twins and MoCo, each containing the code baselines for Barlow Twins and MoCo contrastive learning frameworks.

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/numanai/Visual-Question-Answering-for-Medical-domain
cd Visual-Question-Answering-for-Medical-domain

# [OPTIONAL] create conda environment
conda env create -f conda_env_gpu.yaml -n your_env_name
conda activate your_env_name

# install requirements
pip install -r requirements.txt
```
The instructions to run the pretraining and finetuning codes for both SSL methods, i.e., MoCo and Barlow Twins, can be found in the respective directories.

## Demo
We further provide some of the sample inputs and outputs of the model. Inputs are the natural language questions about the images, and answers are the medical diseases. Some of the answers are consistent with the ground truth, while others are not.

<p align="middle">
  <img src="./demo_images/demo1.png" width="100" />
  <img src="./demo_images/demo2.png" width="100" /> 
  <img src="./demo_images/demo3.png" width="100" />
</p>

## Datasets
#### Datasets used for pretraining:
1. ImageCLEF from 2018 till 2020 - [ImageCLEF-2018](https://www.aicrowd.com/clef_tasks/8/task_dataset_files?challenge_id=155), [ImageCLEF-2019](https://github.com/abachaa/VQA-Med-2019), [ImageCLEF-2020](https://github.com/abachaa/VQA-Med-2020). Please note that a signed and approved End User Agreement (EUA) is required to use these datasets.
2. [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
3. [MIMIC-CXR](https://mimic-cxr.mit.edu/)
CheXpert and MIMIC-CXR datasets were not explicitly used in this project, rather the weights of the model pretrained on them were adapted to initialize our own models.
#### Datasets used for fine-tuning:
1. [ImageCLEF-2020](https://github.com/abachaa/VQA-Med-2020) from ImageCLEF-2020 VQA in medical domain challenge

## Authors
```yaml
Elnura Zhalieva (ryuzakizh) and Numan Saeed (numanai)
```
