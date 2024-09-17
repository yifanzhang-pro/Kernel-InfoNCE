## Introduction

Official implementation of ICLR 2024 paper "Contrastive Learning Is Spectral Clustering On Similarity Graph" (https://arxiv.org/abs/2303.15103) .


## Installation

Requirement:
- Conda

Once installed conda, you can create the `contrastive` environment using 
`conda env create -f environment.yaml`. 



## Random Search
Just run 
`python random_search.py`

You can overwrite any pretraining arguments while random searching. For example, you want to random search the hyperparameters for CIFAR100 with lars optimizer in 100 epochs, you can run `python random_search.py --dataset cifar100 --optimizer lars --max_epochs 100`

For more details, see the argument help of `random_search.py`

## Pretraining

Once you have got the best parameter by random search, you can run `python simclr_module.py [args]` to pretrain.

For more details, see the argument help of `simclr_module.py`. 


## Linear Probing

For linear probe, run `python simclr_finetune.py --ckpt_path [path/to/your/ckpt] [args]`

For more details, see the argument help of `simclr_finetune.py`. For most cases, you may only need to change `dataset`, `data_dir`, `ckpt_path` three arguments.

## Acknowledgement

This repo is mainly based on [Pytorch Lightning](https://github.com/Lightning-AI/lightning). Many thanks to their wonderful work!


## Citations
Please cite the paper and star this repo if you use Kernel-InfoNCE and find it interesting/useful, thanks! Feel free to contact zhangyif21@mails.tsinghua.edu.cn | yangjq21@mails.tsinghua.edu.cn or open an issue if you have any questions.

```bibtex
@article{tan2023contrastive,
  title={Contrastive Learning Is Spectral Clustering On Similarity Graph},
  author={Tan, Zhiquan and Zhang, Yifan and Yang, Jingqin and Yuan, Yang},
  journal={arXiv preprint arXiv:2303.15103},
  year={2023}
}
```
