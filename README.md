#  LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions 

## Installation

Install the dependencies in ``requirements.txt``and download the *wordnet* as follows:

```
import nltk
nltk.download('wordnet')
```

## Quickstart

For a quick demo, see [DEMO](demo/).

## Hydra Usage

The repository uses [Hydra](https://hydra.cc) framework to manage experiments.
We provide three main experiments:

``train.py``: Trains a model from scratch.
``eval.py``: Evaluates a pre-trained model.
``gen.py``: Generates samples with the given model.

Use ``--help`` to list all experiment parameters and help on Hydra usage:

```
python train.py --help
```

Hydra will output experiment results under ``outputs`` folder.
Outputs of train and eval tasks will be under ``run``folder while results of generation experiments will be scaffolded first by date and then time of the day.

## Examples

Train 32 nonlinear directions with ``BigGAN`` and class ``bulbul`` by targeting feature layer ``generator.layers.4``.

```
python train.py k=32 generator=biggan generator.feature_layer=generator.layers.4 generator.class_name=bulbul model=nonlinear hparams.batch_size=16 model.alpha='[-3.0,+3.0]'
```

Train 100 nonlinear directions with ``StyleGAN2`` and class ``ffhq`` by targeting feature layer ``conv1``.

```
python train.py k=100 generator=stylegan2 generator.feature_layer=conv1 hparams.batch_size=8 generator.class_name=ffhq model=nonlinear model.alpha=1
```

Generate 5 images from the first 4 directions of a pre-trained model located at ``$PATH``

```
python gen.py \
        --config-path=$PATH
        --config-name=config \
        checkpoint="$PATH/best_model.pt"
        +n_samples=5 \
        +alphas="[-15,-10,-5,5,10,15]" \
        +iterative=False \
        +image_size=256 \
        +n_dirs=[0,1,2,3] \
```

You can also use ``feed_layers`` (takes a list of layer indices) option to only activate changes in certain layers of the generator (both in training and generation).
