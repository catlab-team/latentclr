#  LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions 

![stylegan_ours](https://user-images.githubusercontent.com/34350876/136622590-1e6dc067-c2ba-4b80-a1e2-507bdede1143.png)

> https://arxiv.org/abs/2104.00820

> Recent research has shown that it is possible to find interpretable directions in the latent spaces of pre-trained Generative  Adversarial Networks (GANs). These directions enable controllable image generation and support a wide range of semantic editing operations, such as zoom or rotation. The discovery of such directions is often done in a supervised or semi-supervised manner and requires manual annotations which limits their use in practice. In comparison, unsupervised discovery allows finding subtle directions that are difficult to detect a priori.
In this work, we propose a contrastive learning-based approach to discover semantic directions in the latent space of pre-trained GANs in a selfsupervised manner. Our approach finds semantically meaningful dimensions compatible with state-of-the-art methods.


## Installation

Install the dependencies in ``env.yml``
``` bash
$ conda env create -f env.yml
$ conda activate latentclr-env
```

and download the *wordnet* as follows:
``` python
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
        --config-path="$PATH/.hydra"
        --config-name=config \
        checkpoint="$PATH/best_model.pt"
        +n_samples=5 \
        +alphas="[-15,-10,-5,5,10,15]" \
        +iterative=False \
        +image_size=256 \
        +n_dirs=[0,1,2,3] \
```

You can also use ``feed_layers`` (takes a list of layer indices) option to only activate changes in certain layers of the generator (both in training and generation).


## Citation

If you use this code for your research, please cite our paper:
```
@misc{yüksel2021latentclr,
      title={LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions},
      author={Oğuz Kaan Yüksel and Enis Simsar and Ezgi Gülperi Er and Pinar Yanardag},
      year={2021},
      eprint={2104.00820},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


