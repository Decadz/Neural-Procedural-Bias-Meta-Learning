<h1 align="center">
Neural Procedural Bias Meta-Learning
</h1>

This repository contains code for reproducing the experiments in the paper "[*Meta-Learning Neural Procedural Biases*]()" by Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang. A [PyTorch](https://pytorch.org/) + [Higher](https://github.com/facebookresearch/higher) implementation of the newly proposed *Neural Procedural Bias Meta-Learning* (NPBML) algorithm.

![npbml-header-image](https://github.com/Decadz/Neural-Procedural-Bias-Meta-Learning/assets/23614094/6d6bb9c6-f7b9-4a59-8656-3d52a4aa1f3d)

## Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/Decadz/Neural-Procedural-Bias-Meta-Learning.git
cd Neural-Procedural-Bias-Meta-Learning
```

2. Install the necessary libraries and dependencies:
```bash
pip install requirements.txt
```

## Usage

1. Download the desired meta-learning datasets and place them in the "../experiments/resources/datasets/" folder. Each dataset should include "train", "val", and "test" subfolders. All datasets are available via the provided Google Drive [link](https://drive.google.com/drive/folders/1rIkInpp1NBnDytsKOLOFw_Sm5B2Fsxtk?usp=drive_link), except for TieredImageNet, which has a separate download [link](https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html). To automatically download all datasets except TieredImageNet, run the provided script:
```
python experiments/resources/datasets/download_datasets.py
```

2. To run the NPBML algorithm you will first need to pretrain the backbone encoder and relation network. To do this run the following commands via the terminal command (you can find a list of the available arguments in the following files [[1]](https://github.com/Decadz/Neural-Procedural-Bias-Meta-Learning/blob/main/experiments/resources/__init__.py) and [[2]](https://github.com/Decadz/Neural-Procedural-Bias-Meta-Learning/blob/main/source/__init__.py)). Alternatively, you can download the models used in our experiments from the following Google Drive [link](https://drive.google.com/drive/folders/1496Z8XcwhhTaBUI5Ay2bt9OH9FszsMHU?usp=drive_link).
```
python experiments/run_pretraining.py --dataset dataset_name --model model_name --seeds [seeds] --device device
```
```
python experiments/run_relation.py --dataset dataset_name --model model_name --seeds [seeds] --device device
```

3. Following this, a few-shot learning task can be executed via the following terminal command. After the experiment has finished the results will be output into a .json file which can be found in the following directory "../experiments/results/dataset_name/". Note, our experiments natively report the error rate metric, *i.e.*, 1-accuracy.
```
python experiments/run_npbml.py --dataset dataset_name --model model_name --num_ways N --num_shots M --seeds [seeds] --device device
```

### Code Reproducibility: 

The code has not been comprehensively checked and re-run since refactoring. If you're having any issues, find a problem/bug or cannot reproduce similar results as the paper please [open an issue](https://github.com/Decadz/Neural-Procedural-Bias-Meta-Learning/issues) or email me.

## References

If you use our library or find our research of value please consider citing our paper with the following Bibtex entry:

```
@article{raymond2024meta,
  title={Meta-Learning Neural Procedural Biases},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  journal={arXiv preprint arXiv:},
  year={2024}
}
@article{raymond2024thesis,
  title={Meta-Learning Loss Functions for Deep Neural Networks},
  author={Raymond, Christian},
  journal={arXiv preprint arXiv:2406.09713},
  year={2024}
}
```
