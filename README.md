# Transductive Zero-Shot Learning with Visual Structure Constraint

This is the offical implementation of our paper [Transductive Zero-Shot Learning with Visual Structure Constraint (NeurIPS 2019)] using PyTorch.

### Runing

1. (Optional) Finetune the pretrained model on the training dataset. This is useful for fine-grained dataset such as CUB since ImageNet is a general dataset:
    
    ```
    python Tools/Finetune.py --data_dir [root of dataset] --train_class_list [file of seen class name] --gpu_id [Your device information]
    ```
2. Extract the features of each image

    ```
    python Tools/Feature_extractor.py --data_dir [root of dataset] --pretrain_model [ignore if you use ImageNet pretrain directly] --mode [SS|PS] --gpu_id [Your device information]
    ```
3. Extract Visual Center

    python Tools/VC_extractor.py --train_class_list [file of seen class name] --test_class_list [file of unseen class name] --data_dir [root of dataset] --feature_name [name of json file] --dataset_name [AwA|...] --mode [SS|PS]

4. Cluster

        