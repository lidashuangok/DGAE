# DGAE
Code for the paper ["Dual-decoder graph autoencoder for unsupervised graph representation learning"](https://www.sciencedirect.com/science/article/abs/pii/S0950705121008261)

Sun, Dengdi, Dashuang Li, Zhuanlian Ding, Xingyi Zhang and Jin Tang. “[Dual-decoder graph autoencoder for unsupervised graph representation learning.”](https://www.sciencedirect.com/science/article/abs/pii/S0950705121008261) Knowl. Based Syst. 234 (2021): 107564.

 The model flow is as follows ：

![](dga_flow.png)



The base code is a  the Variational Graph Auto-Encoder model described in the paper:
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

The code in this repo is based on or refers to https://github.com/tkipf/gae, https://github.com/tkipf/gcn and https://github.com/vmasrani/gae_in_pytorch.

## Requirements
* TensorFlow (1.0 or later)
* python 3
* networkx
* scikit-learn
* scipy

## Data
In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `load_data()` function in `input_data.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid


## Run from

```bash
python run.py
```


## Reference

If you make advantage of the DGAE model or use the datasets released in our paper, please cite the following in your manuscript:

```
@article{SUN2021107564,
title = {Dual-decoder graph autoencoder for unsupervised graph representation learning},
journal = {Knowledge-Based Systems},
volume = {234},
pages = {107564},
year = {2021},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2021.107564},
url = {https://www.sciencedirect.com/science/article/pii/S0950705121008261},
author = {Dengdi Sun and Dashuang Li and Zhuanlian Ding and Xingyi Zhang and Jin Tang}
}
```