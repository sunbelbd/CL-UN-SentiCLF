#  Copyright (c) 2020-present, Baidu, Inc.
#  All rights reserved.
#  #
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#  #
#  Acknowledgement: The code is modified based on Facebook AI's XLM.

import argparse
import os

from src.evaluation.clf import CLF
from src.model.embedder import SentenceEmbedder
from src.utils import bool_flag, initialize_exp

XNLI_TASKS = ['XNLI']

# parse parameters
parser = argparse.ArgumentParser(description='Train on GLUE or XNLI')

# main parameters
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")
parser.add_argument("--max_epoch", type=int, default=100000,
                    help="Maximum epoch size")

# evaluation task / pretrained model
parser.add_argument("--transfer_tasks", type=str, default="",
                    help="Transfer tasks, example: 'MNLI-m,RTE,XNLI' ")
parser.add_argument("--model_path", type=str, default="",
                    help="Model location")
parser.add_argument("--data_category", type=str, default="",
                    help="Mainly used for amazon review data {books, dvd, music}")
# data
parser.add_argument("--data_path", type=str, default="",
                    help="Data path")
parser.add_argument("--target_lang", type=str, default="",
                    help="")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--min_count", type=int, default=0,
                    help="Minimum vocabulary count")

# batch parameters
parser.add_argument("--max_len", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--group_by_size", type=bool_flag, default=False,
                    help="Sort sentences by size during the training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")
parser.add_argument("--max_batch_size", type=int, default=0,
                    help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
parser.add_argument("--tokens_per_batch", type=int, default=-1,
                    help="Number of tokens per batch")
parser.add_argument("--clf_output_dim", type=int, default=2,
                    help="Classifier output dimension (# classes)")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--weighted_training", type=bool_flag, default=False,
                    help="Use a weighted loss during training")
parser.add_argument("--dropout", type=float, default=0,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                    help="Embedder (pretrained model) optimizer")
parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                    help="Projection (classifier) optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")
parser.add_argument("--epoch_size", type=int, default=-1,
                    help="Epoch size (-1 for full pass over the dataset)")

# debug
parser.add_argument("--debug_train", type=bool_flag, default=False,
                    help="Use valid sets for train sets (faster loading)")
parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                    help="Debug multi-GPU / multi-node within a SLURM job")

# parse parameters
params = parser.parse_args()
if params.tokens_per_batch > -1:
    params.group_by_size = True

# check parameters
assert os.path.isdir(params.data_path)
assert os.path.isfile(params.model_path)

# reload pretrained model: use XLM-100 languages
# embedder contains pretrained model dico, how to align with the finetuned data?
embedder = SentenceEmbedder.reload(params.model_path, params)

# reload langs from pretrained model
params.n_langs = embedder.pretrain_params['n_langs']
params.id2lang = embedder.pretrain_params['id2lang']
params.lang2id = embedder.pretrain_params['lang2id']
tgt_lang = params.target_lang if params.target_lang != 'jp' else 'ja'

assert tgt_lang in params.lang2id
# initialize the experiment / build sentence embedder
logger = initialize_exp(params)
scores = {}

# prepare trainers / evaluators
clf = CLF(embedder, scores, params)

# run each product over each lanugage
# for task in params.transfer_tasks:
#     if task in GLUE_TASKS:
#         glue.run(task)
#     if task in XNLI_TASKS:
#         xnli.run()
clf.run()
