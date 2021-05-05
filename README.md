<div align="center">

# Visual Question Answering for Medical Domain

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>

</div>

## Description
Visual Question Answering (VQA) is a rising interdisciplinary problem that demands the knowledge of both Computer Vision (CV) and Natural Language Processing (NLP). Many domain-specific VQA tasks have emerged in the last few years, and VQA in the medical domain is one such that plays a significant role in providing medical assistance to both doctors and patients. For example, doctors could use VQA model answers as assistance in medical diagnosis, while patients could ask questions from VQA related to their medical images to better understand their physical condition.
A VQA system takes as input an image and a natural language question about the image and produces an answer consistent with the visual content of a given image. A lack of large-scale labeled medical data makes the application of VQA in the medical domain even more complicated. To facilitate the research of implementing VQA in medical domain, ImageCLEF, which is part of the Conference and Labs of the Evaluation Forum (CLEF), has been conducting annual VQA-Med challenges since 2018. In this project, we will be using the dataset coming from VQA-Med 2020 challenge for training. As far as our knowledge goes, recent VQA-Med challenge participants have not used self-supervised learning (SSL) techniques and instead have focused on transfer learning, ensemble models, etc. Thus, towards the possible solution of the VQA-Med problem, in this project we implement two contrastive learning frameworks, MoCo and Barlow Twins pretrained on different medical datasets and fine-tuned on VQA-Med 2020 dataset.


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

## Authors
```yaml
Elnura Zhalieva and Numan Saeed
```
