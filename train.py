#  Copyright (c) 2020-present, Baidu, Inc.
#  All rights reserved.
#  #
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#  #
#  Acknowledgement: The code is modified based on Facebook AI's XLM.


import argparse
import json

import paddle.fluid as fluid

from src.data.loader import check_data_params, load_data
from src.evaluation.evaluator import EncDecEvaluator
from src.model import check_model_params, build_model
from src.slurm import init_signal_handler, init_distributed_mode
from src.trainer import EncDecTrainer
from src.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
import time


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. "
                             "Level of optimization. -1 to disable.")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=True,
                        help="Only use an encoder")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--dec_layers", type=int, default=2,
                        help="Number of Transformer decoder layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=False,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--data_category", type=str, default="",
                        help="Mainly used for amazon review data {books, dvd, music}")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--max_positions", type=int, default=256,
                        help="Maximum length of position embeddings")
    parser.add_argument("--group_by_size", type=bool_flag, default=False,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, "
                             "0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--sample_clf_prob", type=float, default=0.4,
                        help="Probability of sampling from labeled data for each batch")
    parser.add_argument("--train_clf", type=bool_flag, default=True,
                        help="Train classifier")
    parser.add_argument("--train_dis", type=bool_flag, default=True,
                        help="Train discriminator")
    parser.add_argument("--train_dae", type=bool_flag, default=True,
                        help="Train denosing auto-encoder")
    parser.add_argument("--train_bt", type=bool_flag, default=False,
                        help="Train backtranslation")

    parser.add_argument("--dis_steps", type=int, default=1,
                        help="Number of discriminator training iterations per epoch")
    parser.add_argument("--clf_steps", type=int, default=100,
                        help="Number of classifier training iterations per epoch")
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=200000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="valid_en_clf_acc,15",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # classifier parameters
    parser.add_argument("--clf_layers", type=int, default=2,
                        help="Number of hidden layers in the classifier. "
                             "0 indicates a logistic regression.")
    parser.add_argument("--clf_hidden_dim", type=int, default=128,
                        help="Classifier hidden layers dimension")
    parser.add_argument("--clf_output_dim", type=int, default=2,
                        help="Classifier output dimension (# classes)")
    parser.add_argument("--clf_batch_norm", type=bool_flag, default=True,
                        help="Batch normalization or not")
    parser.add_argument("--clf_attention", type=bool_flag, default=False,
                        help="Use dot attention for classifier's input")
    parser.add_argument("--clf_mtv", type=bool_flag, default=False,
                        help="Use multi-view representation for classifier training")
    parser.add_argument("--clf_dropout", type=float, default=0,
                        help="Classifier dropout")
    parser.add_argument("--clf_clip", type=float, default=0,
                        help="Clip classifier weights (0 to disable)")

    # discriminator parameters
    parser.add_argument("--dis_layers", type=int, default=2,
                        help="Number of hidden layers in the discriminator")
    parser.add_argument("--dis_hidden_dim", type=int, default=128,
                        help="Discriminator hidden layers dimension")
    parser.add_argument("--dis_dropout", type=float, default=0,
                        help="Discriminator dropout")
    parser.add_argument("--dis_clip", type=float, default=0,
                        help="Clip discriminator weights (0 to disable)")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_dis", type=float, default=0.5,
                        help="discriminator coefficient")
    parser.add_argument("--lambda_div", type=float, default=0.1,
                        help="discriminator coefficient")
    parser.add_argument("--lambda_clf", type=float, default=1.0,
                        help="Classifier coefficient")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")

    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_dis", type=bool_flag, default=True,
                        help="Reload pretrained discriminator")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    return parser


def main(params):
    # initialize the multi-GPU / multi-node training
    # use_gpu = True
    # place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

    # init_distributed_mode(params)
    # initialize the experiment
    logger = initialize_exp(params)

    # initialize SLURM signal handler for time limit / pre-emption
    init_signal_handler()
    # load data
    data = load_data(params)

    with fluid.dygraph.guard():
        # build model: encoder and decoder branch

        encoder, decoder, discriminator, classifier = build_model(params, data['dico'])
        # encoder, decoder, discriminator = build_model(params, data['dico'])

        # build trainer, reload potential checkpoints / build evaluator

        trainer = EncDecTrainer(encoder, decoder, discriminator, classifier, data, params)
        # trainer = EncDecTrainer(encoder, decoder, discriminator, data, params)
        evaluator = EncDecEvaluator(trainer, data, params)
        # evaluator = CLFEvaluator(encoder, classifier, data, params)

        # evaluation
        if params.eval_only:
            scores = evaluator.run_all_evals(trainer)
            for k, v in scores.items():
                logger.info("%s -> %.6f" % (k, v))
            logger.info("__log__:%s" % json.dumps(scores))
            exit()

    # set sampling probabilities for training
    set_sampling_probs(data, params)

    # language model training
    for epoch in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % epoch)

        trainer.n_sentences = 0

        while trainer.n_sentences < trainer.epoch_size:
            # time.sleep(30)
            # discriminator step: used in UMT!!!
            with fluid.dygraph.guard():
                if params.train_dis and params.lambda_dis > 0 and params.dis_steps > 0:
                    # logger.info("Running discriminator step at epoch %d " % epoch)
                    for _ in range(params.dis_steps):
                        trainer.discriminator_step()

            # classifier step
            # train_all: tuning enc, dec and clf in clf_step(). mt_step() is disabled.
            # train_clf: tining clf only in clf_step(). enc, dec are tuned in mt_step()
            with fluid.dygraph.guard():
                if params.train_clf and params.clf_steps > 0:
                    # logger.info("Running classifier step at epoch %d" % epoch)
                    for _ in range(params.clf_steps):
                        if params.train_dae:  # tune enc-dec separately
                            trainer.classifier_step(mode="train_clf")
                        else:  # train enc-dec together with clf training
                            trainer.classifier_step(mode="train_all")

            # denoising auto-encoder steps: used in UMT!!!
            with fluid.dygraph.guard():
                if params.train_dae:
                    # logger.info("Running AE step at epoch %d" % epoch)
                    for lang in shuf_order(params.ae_steps):
                        trainer.mt_step(lang, lang, params.lambda_ae)

            # back-translation steps: used in UMT!!!
            with fluid.dygraph.guard():
                if params.train_bt:
                    # logger.info("Running BT step at epoch %d" % epoch)
                    for lang1, lang2, lang3 in shuf_order(params.bt_steps):
                        # print("bt step: %s-%s-%s" %(lang1, lang2, lang3))
                        trainer.bt_step(lang1, lang2, lang3, params.lambda_bt)

            trainer.iter()

        logger.info("============ End of epoch %i ============" % epoch)

        # evaluate all modules
        with fluid.dygraph.guard():
            scores = evaluator.run_all_evals(trainer, epoch)

            # print / JSON log
            for k, v in scores.items():
                logger.info("%s -> %.6f" % (k, v))
            # if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

            # end of epoch
            # trainer.save_best_model(scores)
            # trainer.save_periodic()
            trainer.end_epoch(scores)


if __name__ == '__main__':
    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    main(params)
