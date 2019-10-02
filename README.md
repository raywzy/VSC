# Transductive Zero-Shot Learning with Visual Structure Constraint

This is the offical implementation of our paper [Transductive Zero-Shot Learning with Visual Structure Constraint (NeurIPS 2019)] using PyTorch.

###Runing

1. Finetune the pretrained model on the training dataset. This is useful fine-grained dataset such as CUB since ImageNet is a general dataset:
    
    ```
    python Tools/Finetune.py --data_dir [root of dataset] --train_class_list [file saving the training class name] --gpu_id [Your device information]
    ```
2. Extract the features of each image

    ```
    python Tools/Feature_extractor.py --data_dir [root of dataset] --pretrain_model [ignore if you use ImageNet pretrain directly] --mode [SS|PS] --gpu_id [Your device information]
    ```
