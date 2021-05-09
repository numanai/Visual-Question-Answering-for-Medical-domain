# ImageCLEF - Downstream Task 


**We provide the code for finetuning and testing the pretrained model**
 

### Dataset
Use the MedVQA data provided by ImageCLEF. 
To generate the dataset for training and testing, 
1. Download images from [`ImageCLEF-VQAMed`](https://www.imageclef.org/2021/medical/vqa) and update the path for the datasets.
2. Define the transforms you want to use for augmentations. 
3. Run the dataloader script in `finetune.ipynb` and load the dataset.


### How to train
*First Step* Define the model in ResNetModelSSL class and load the pretrained weights (.ckpt).

*Second Step* Run the training loop
