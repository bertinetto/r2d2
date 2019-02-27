### Meta-learning with differentiable closed-form solvers.

#### Paper
* OpenReview: https://openreview.net/forum?id=HyxnZh0ct7 

Please refer to it as:
```
@inproceedings{
bertinetto2018metalearning,
title={Meta-learning with differentiable closed-form solvers},
author={Luca Bertinetto and Joao F. Henriques and Philip Torr and Andrea Vedaldi},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=HyxnZh0ct7},
}
```

#### Data setup
* In `scripts/train/conf/fewshots.yaml`, specify the location of your custom `$DATASET_PATH`  (`data.root_dir`).
* Download [Omniglot](https://gofile.io/?c=No7UF3), [CIFAR-FS](https://gofile.io/?c=pwbukP) and [<i>mini</i>ImageNet](https://gofile.io/?c=KbY4eQ)  the above format. Original datasets from [here](https://github.com/brendenlake/omniglot/tree/master/python) and [here](https://www.cs.toronto.edu/~kriz/cifar.html).
* Download and extract one or more datasets in your custom `$DATASET_PATH` folder, the code assumes the following structure (example):
```
$DATASET_PATH
├── miniimagenet
│   ├── data
│   │   ├── n01532829
|   |   |── ...
│   │   └── n13133613
│   ├── splits
│   │   └── ravi-larochelle
|   |   |   ├── train.txt
|   |   |   ├── val.txt
|   |   |   └── test.txt
├── omniglot
|   ...
├── cifarfs 
|   ...
```

 
#### Repo setup (with Conda)

* Set up conda environment: `conda env create -f environment.yml`.
* `source activate fsrr`
* Install [torchnet](https://github.com/pytorch/tnt): `pip install git+https://github.com/pytorch/tnt.git@master`.
* Install the repo package: `pip install -e .`
* `source deactivate fsrr`

#### Run
`scripts/train/experiments.sh` contains all the experiments of the paper  (train+eval) in blocks of three lines, e.g.
```
expm_folder=mini_r2d2 
python run_train.py --log.exp_dir $expm_folder --data.dataset miniimagenet --data.way 16 --model.drop 0.1 --base_learner.init_adj_scale 1e-4 
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 1 --model.model_path ../train/results/$expm_folder/best_model.1shot.t7 
python ../eval/run_eval.py --data.test_episodes 10000 --data.test_way 5 --data.test_shot 5 --model.model_path ../train/results/$expm_folder/best_model.5shot.t7

```

##### Note
Some of the files of this repository (e.g. data loading and training boilerplate routines) are the result of a modification of [prototypical networks code](https://github.com/jakesnell/prototypical-networks) and contain a statement in their header.


