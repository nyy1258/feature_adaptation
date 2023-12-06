## Requirements
### Installation
Create a conda environment and install dependencies:
```bash
git clone https://github.com/gaopengcuhk/Tip-Adapter.git
cd Tip-Adapter

conda create -n tip_adapter python=3.7
conda activate tip_adapter

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
```

### Dataset
Follow [DATASET.md](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/DATASET.md) to install ImageNet and other 10 datasets referring to CoOp.

1. officehome dataset:
You can refer "move_file.py" to move the data to the data folder

DATA -
     |OfficeHome
	     |---- train
	     |      |------ 0_Ruler ~ 64_Printer
	     |
	     |---- val
	     |      |------ 0_Ruler ~ 64_Printer
	     |
	     |---- test
	            |------ 0_Ruler ~ 64_Printer
	            
## Get Started
### Configs
The running configurations can be modified in `configs/dataset.yaml`, including shot numbers, visual encoders, and hyperparamters. 

For simplicity, we provide the hyperparamters achieving the overall best performance on 1\~16 shots for a dataset, which accord with the scores reported in the paper. If respectively tuned for different shot numbers, the 1\~16-shot performance can be further improved. You can edit the `search_scale`, `search_step`, `init_beta` and `init_alpha` for fine-grained tuning.

Note that the default `load_cache` and `load_pre_feat` are `False` for the first running, which will store the cache model and val/test features in `configs/dataset/`. For later running, they can be set as `True` for faster hyperparamters tuning.

For example:
    Dataset: OfficeHome ==> change configs/officehome.yaml for configuration

### Running
For ImageNet dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main_imagenet.py --config configs/imagenet.yaml
```
For other 10 datasets:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/dataset.yaml
```
The fine-tuning of Tip-Adapter-F will be automatically conducted after the training-free Tip-Adapter.

For example:
    Dataset: OfficeHome 
    Command: CUDA_VISIBLE_DEVICES=0 python main.py --config configs/officehome.yaml

## Result
"result.log" contains my experiment result 
You can check Tip-adapator result of OfficeHome accuracy in "result.log" 

## plot tsne
You can produce tsne image during training by change plot_tsne flag = True in main.py
 
## Citation
```bash
@article{zhang2021tip,
  title={Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling},
  author={Zhang, Renrui and Fang, Rongyao and Gao, Peng and Zhang, Wei and Li, Kunchang and Dai, Jifeng and Qiao, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2111.03930},
  year={2021}
}
```
