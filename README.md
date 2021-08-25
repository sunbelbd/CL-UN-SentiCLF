# Cross-Lingual Unsupervised Sentiment Classification with Multi-View Transfer Learning

## Introduction
This repository contains code that supports experiments in our ACL 2020 paper "Cross-Lingual Unsupervised Sentiment Classification with Multi-View Transfer Learning". Note that this is the PaddlePaddle version of the implementation, which is largely motivated and modified on unsupervised machine translation from the XLM codebase by Facebook AI Research. Great appreications to them! 

There is also a Pytorch version, which is available upon request.


## Install and usage

### Install dependencies

```
bash scripts/install-tools.sh
```

### Data preprocessing
We use English and German as an example to show the data processing.  
Download [de-en data](https://drive.google.com/file/d/1gsiysHgTcTkYfNJWd33IbMNCeDX62lY-/view?usp=sharing) and unzip it to the code root directory. Running the following command, the script will tokenize labeled training, valid and test data and convert them to BPE format. 
```
bash prepare-clf-data.sh --src en --tgt de --reload_codes ./pretrain/pretrain_deen/codes_ende --reload_vocab ./pretrain/pretrain_deen/vocab_ende --product books (dvd or music)
```

Similarly, monolingual unlabeled data can be processed and placed in ./data/processed/de-en folder. 

### Usage
We also use English and German as an example to show how to use it.
```
bash runPaddle_de.sh --clf_steps 50 --num_gpu 1 --exp_name unsupMTDiscCLF_ende --data_category books (dvd or music) --train_dis True --train_encdec True --train_bt True --clf_atten False --clf_mtv True --tokens_per_batch 600
```

## Other Recourses
Shell scripts in scripts/ are some examples of how you submit jobs in SLURM. There are also scripts useful to downloading Wikipedia data, tokenize and binarize them. The pretrained checkpoints provided is intialized by XLM-100 and continuously pretrained using MLM objective on a mixture of monolingual and unlabeled product review data in using [XLM](https://github.com/facebookresearch/XLM)

## Reference
If you find the code useful, please consider citing it as follows:

```
@inproceedings{DBLP:conf/acl/FeiL20,
  author    = {Hongliang Fei and Ping Li},
  title     = {Cross-Lingual Unsupervised Sentiment Classification with Multi-View Transfer Learning},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2020, Online, July 5-10, 2020},
  pages     = {5759--5771},
  publisher = {Association for Computational Linguistics},
  year      = {2020},
  url       = {https://doi.org/10.18653/v1/2020.acl-main.510}
}
```
