# Transductive Zero-Shot Learning with Visual Structure Constraint

This is the offical implementation of our paper [Transductive Zero-Shot Learning with Visual Structure Constraint (NeurIPS 2019)] using PyTorch.


## Requirements
```
Python 3.6.3
Pytorch 0.4.1
CUDA 8.0.61
Scipy 1.2.1
Scikit-learn 0.20.3
```
All experiments are conducted on single TITAN XP.


## Running

1. (Optional) Finetune the pretrained model on the training dataset. This is useful for fine-grained dataset such as CUB since ImageNet is a general dataset:
    
    ```
    python Tools/Finetune.py --data_dir [root of dataset] --train_class_list [file of seen class name] --gpu_id [Your device information]
    ```
2. Extract the features of each image

    ```
    python Tools/Feature_extractor.py --data_dir [root of dataset] --pretrain_model [ignore if you use ImageNet pretrain directly] --mode [SS|PS] --gpu_id [Your device information]
    ```
3. Extract Visual Center
    ```
    python Tools/VC_extractor.py --train_class_list [file of seen class name] --test_class_list [file of unseen class name] --data_dir [root of dataset] --feature_name [name of json file] --dataset_name [AwA|...] --mode [SS|PS]
    ```
4. Cluster in target feature space

    ```
    python Cluster/Cluster.py --test_class_list [file of unseen class name] --mode[SS|PS] --data_dir [root of dataset] --feature_name [name of json file] --cluster_method [Kmeans|Spectral] --center_num [unseen class num] --dataset_name [AwA|...]
    ```
5. Train the model
    ```
    python Train_Model.py --lamda [The coefficient of VSC term] --data_path [root of dataset] --method [VCL|CDVSc|BMVSc|WDVSc] --GPU [Your device information] --dataset [AwA1|AwA2|CUB|SUN|SUN10] --split_mode [standard_split|proposed_split] --train_center [VC of seen classes] --cluster_center [Approximated VC of unseen classes] --save_dir [save place]
    ```
6. Test the results
    ```
    python Eval_Model.py --GPU [Your device information] --dataset [AwA|...] --split_mode [standard_split|proposed_split] --checkpoint_fn [The saving dir]
    ```

### Notice: 
For CDVSc and BMVSc, please cross-validate the parameter according to the paper. For WDVSc, you could set lamda to 0.001 directly.
There may exist some variance while performing cluster. To achieve similar results with original paper, we suggest to use [our cluster](https://drive.google.com/open?id=1tpXoPS8KMgsVgDVW0x_rC19s9L7ExMqv) results directly.       


## Citation

If you find our work is helpful for your research, please cite the following paper.

```bibtex
@inproceedings{wan2019transductive,
title={Transductive Zero-Shot Learning with Visual Structure Constraint},
author={Wan, Ziyu and Chen, Dongdong and Li, Yan and Yan, Xingguang and Zhang, Junge and Yu, Yizhou and Liao, Jing},
booktitle={Thirty-third Conference on Neural Information Processing Systems (NeurIPS)},
year={2019}
}
```
## Acknowledgement

Some parts of the code are borrowed from [DGP](https://github.com/cyvius96/DGP).

## Contact
Feel free to contact me if there are any questions: ziyuwan2-c@my.cityu.edu.hk