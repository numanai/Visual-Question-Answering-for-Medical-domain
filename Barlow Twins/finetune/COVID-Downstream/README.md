# COVID CT - Downstream Task 


**We provide the code for finetuning and testing the pretrained model**
 

### Dataset
Use the split in `Data-split`. 
To generate the dataset for training and testing, 
1. Download images from repo [`Images-processed`](https://github.com/UCSD-AI4H/COVID-CT/tree/master/Images-processed)
2. Download txt files for image names in train, val, and test set from repo [`Data-split`](https://github.com/UCSD-AI4H/COVID-CT/tree/master/Data-split)
3. Use the dataloader defined the script `finetune.ipynb` and load the dataset.


### How to train
*First Step* Download and update the path for the dataset.

*Second Step* Copy the pretrained model file `.ckpt` and update the path

*Third Step* Run the training function 


The code was inspired from https://github.com/UCSD-AI4H/COVID-CT/tree/master/baseline%20methods/Self-Trans
