#CoSMIG
Communicative Subgraph Representation Learning for Multi-Relational Inductive Drug-Gene Interaction Prediction

![alt text](https://github.com/biomed-AI/CoSMIG/raw/master/framework.jpg "Illustration of CoSMIG")


This is the standalone code for our paper: [Communicative Subgraph Representation Learning for Multi-Relational Inductive Drug-Gene Interaction Prediction]()

## Requirements

Stable version: Python 3.7.9 + PyTorch 1.7.1+cu110 + PyTorch_Geometric 1.6.3.


Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.

Also you can  install the required packages follow there instructions (tested on a linux terminal):

`conda env create -f environment.yaml`


## Datasets

Data (from DrugBank and DGIdb) could be download from [here](https://github.com/biomed-AI/CoSMIG/raw/master/data)

### Statistic of DGI Dataset
|Dataset|DrugBank|DGIdb|
|:-:|:-:|:-:|
|#Drug|425|1185|
|#Gene|11284|1164|
|#Interactions|80924|11266|
|Interaction type|2|14|

## Usages
For training on DrugBank on the transductive scenario:
```
CUDA_VISIBLE_DEVICES=0 python main.py --data-name DrugBank --testing --dynamic-train --dynamic-test --dynamic-val --save-results --max-nodes-per-hop 200
```


For training on DGIdb on the inductive scenario:
```
CUDA_VISIBLE_DEVICES=0 python main.py --data-name DGIdb --testing --mode inductive --dynamic-train --dynamic-test --dynamic-val --save-results --max-nodes-per-hop 200
```

More parameters could be found by:
```
python main.py -h
```

## Reference
If you find the code useful, please cite our paper.
```
```

## Contact
Jiahua Rao (raojh6@mail2.sysu.edu.cn) and Yuedong Yang (yangyd25@mail.sysu.edu.cn)