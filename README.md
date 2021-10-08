# LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions

This is the repository that contains source code for the [LatentCLR website](https://catlab-team.github.io/latentclr/).

![image](https://user-images.githubusercontent.com/34350876/136621888-5fe74002-8a1f-40d5-8087-82cb1b4fe31d.png)

Recent research has shown that it is possible to find interpretable directions in the latent spaces of pre-trained Generative Adversarial Networks (GANs). These directions enable controllable image generation and support a wide range of semantic editing operations, such as zoom or rotation. The discovery of such directions is often done in a supervised or semi-supervised manner and requires manual annotations which limits their use in practice. In comparison, unsupervised discovery allows finding subtle directions that are difficult to detect a priori.

In this work, we propose a contrastive learning-based approach to discover semantic directions in the latent space of pre-trained GANs in a selfsupervised manner. Our approach finds semantically meaningful dimensions compatible with state-of-the-art methods.



If you find our work useful please cite:
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

# Website License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Template is from : https://github.com/nerfies/nerfies.github.io
