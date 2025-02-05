<div align="center">    
 
# Fooling CAM via Adversarial Model Manipulation

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.12+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.5+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 2.4+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>


[![Conference](https://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)]
</div>


 
## Description
This repository provides a new implementation of the paper Fooling Neural Network Interpretations via Adversarial Model Manipulation (https://arxiv.org/abs/1902.02041). It includes code for fooling deep neural networks, including recently published architectures, as well as various CAM-based interpretation methods implemented in the TorchCAM (https://github.com/frgfm/torch-cam) repository. The implementation is built using PyTorch Lightning.

## Things to do before running

```bash
# clone project   
git clone https://github.com/joshua840/AMM.git

# create environment
conda env create -f env.yaml 
conda activate torch2.5_cuda12.4
```

- Update `configs/AMM.yaml`
  - Specify values for `dataset`, `data_dir`, `model`, and `h_target_layer`.
- Update `Logger` option in `configs/trainer.yaml`
  - Using `NeptuneLogger`
    - Set `api_key`, `project`, and `name`, by following the instructions in (https://docs.neptune.ai/setup/).
  - Using the other loggers
    - Check the supported loggers in (https://lightning.ai/docs/pytorch/stable/extensions/logging.html)
    - Loggers such as `WandB`, `Comet`, `Tensorboard`, and others are available.
- (Optional) Mini ImageNet dataset (4GB) for demo run
  - Download (https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000?resource=download)
  - set `dataset=imagenet` and `data_dir=PATH/TO/ImageNet`
 
After finishing the above setting, you can directly run the following code:
```bash
# run experiments
bash scripts/amm.sh
```

The checkpoints will be saved in `.neptune` directory.

## Configuration tips

### Trainer args
In Lightining, the `Trainer` class includes arguments that are commonly used for model training. For more details on the `Trainer` class, please refer the following API documentation.:(https://lightning.ai/docs/pytorch/stable/common/trainer.html).

The argument lists are available as well using the following command:
```bash
python -m src.main -h
```


### LightningModule args and using YAML files
In Lightining, arguments of module classes inherenting from `Lightning.pytorch.LightningModule` are automatically registered in argparse lists. This feature keeps the codes clean by eliminating redundant argparse declarations. 

Another advantage of Lightning is built-in support for using YAML files to pass the hyperparameters, including class-level arguments. 

For more details, please refer to the Lightning tutorials 
- (https://lightning.ai/docs/pytorch/stable/common/hyperparameters.html)
- (https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate.html)




### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
