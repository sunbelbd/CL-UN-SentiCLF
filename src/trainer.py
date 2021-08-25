#  Copyright (c) 2020-present, Baidu, Inc.
#  All rights reserved.
#  #
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#  #
#  Acknowledgement: The code is modified based on Facebook AI's XLM.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Handle optimizer, with torch_no_grad() block
Handle evaluation code.
Tomorrow.
"""
import math
import os
import time
from collections import OrderedDict
from logging import getLogger
from random import random

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.layers import reduce_max, reduce_sum, transpose, reduce_mean, \
    reshape, concat, kldiv_loss, log, softmax, zeros, fill_constant, unsqueeze, \
    softmax_with_cross_entropy

from .optim import get_optimizer
from .utils import parse_lambda_config, update_lambdas
from .utils import to_cuda, concat_batches, masked_select

logger = getLogger()


class Trainer(object):

    def __init__(self, data, params):
        """
        Initialize trainer.
        """
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # set optimizers: convert to paddle optimizers
        self.set_optimizers()

        # stopping criterion used for early stopping
        if params.stopping_criterion != '':
            split = params.stopping_criterion.split(',')
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == '_':
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # probability of masking out / randomize / not modify words to predict
        # params.pred_probs = torch.FloatTensor([params.word_mask, params.word_keep, params.word_rand])
        params.pred_probs = [params.word_mask, params.word_keep, params.word_rand]

        # probabilty to predict a word
        counts = np.array(list(self.data['dico'].counts.values()))
        params.mask_scores = np.maximum(counts, 1) ** -params.sample_alpha
        params.mask_scores[params.pad_index] = 0  # do not predict <PAD> index
        params.mask_scores[counts == 0] = 0  # do not predict special tokens

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sentences = 0
        # training progress report dict
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0), ('dis_loss', []), ('clf_loss', [])] +
            [('CLM-%s' % l, []) for l in params.langs] +
            # [('CLM-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys()] +
            # [('CLM-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys()] +
            [('MLM-%s' % l, []) for l in params.langs] +
            # [('MLM-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys()] +
            # [('MLM-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys()] +
            [('PC-%s-%s' % (l1, l2), []) for l1, l2 in params.pc_steps] +
            [('AE-DIS-CLF-%s' % lang, []) for lang in params.ae_steps] +
            [('MT-%s-%s' % (l1, l2), []) for l1, l2 in params.mt_steps] +
            [('BT-DIS-CLF-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps]
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # initialize lambda coefficients and their configurations
        parse_lambda_config(params)

    def set_parameters(self):
        """
        Set parameters.
        """
        params = self.params
        self.parameters = {}
        named_params = []
        params_names_debug = {}
        for name in self.MODEL_NAMES:
            # named_params.extend([(k, p) for k, p in getattr(self, name).params if not p.stop_gradient])
            self.parameters[name] = [p for k, p in getattr(self, name).named_parameters() if not p.stop_gradient]
            params_names_debug[name] = [k for k, p in getattr(self, name).named_parameters() if not p.stop_gradient]
            named_params.extend(self.parameters[name])
            # print("Debug, %s parameter list" % name)
            # print("\n".join(params_names_debug[name]))

        # model
        self.parameters['model'] = named_params


        # log
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}

        # model optimizer: change to paddle accordingly
        # May just assign an optimizers w/o calling a func
        # for name in self.MODEL_NAMES:
        self.optimizers["model"] = get_optimizer(params.optimizer, self.parameters['model'])

        # log
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def optimize(self, loss, model_name='model'):
        """
        Optimize.
        Hold on, convert to paddle accordingly
        """
        # check NaN
        if (loss != loss).detach().numpy().any():
            logger.warning("NaN detected")
            # exit()
            return

        # params = self.params

        # # optimizers
        # names = self.optimizers.keys()
        # optimizers = [self.optimizers[k] for k in names]

        # fluid.clip.set_gradient_clip(
        #     fluid.clip.GradientClipByGlobalNorm(clip_norm=params.clip_grad_norm))
        # regular optimization

        # print("Start backward")
        loss.backward()
        # print("Finish backward")
        self.optimizers[model_name].minimize(loss)

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        update_lambdas(self.params, self.n_total_iter)
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        # s_lr = " - "
        # for k, v in self.optimizers.items():
        #     s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat)

    def get_iterator(self, iter_name, lang1, lang2, stream):
        """
        Create a new iterator for a dataset and save it in self.iterator dictionary
        """
        logger.info("Creating new training data iterator (%s) ..." % ','.join([str(x)
                                                                               for x in [iter_name, lang1, lang2]
                                                                               if x is not None]))
        # assert stream or not self.params.use_memory or not self.params.mem_query_batchnorm
        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1]['train'].get_iterator(shuffle=True)
            else:
                iterator = self.data['mono'][lang1]['train'].get_iterator(
                    shuffle=True,
                    group_by_size=self.params.group_by_size,
                    n_sentences=-1,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            # _lang1, _lang2 = (lang1, lang2)  # remove the input order constrain
            iterator = self.data['para'][(_lang1, _lang2)]['train'].get_iterator(
                shuffle=True,
                group_by_size=self.params.group_by_size,
                n_sentences=-1,
            )
        # save the new iterator in self.iterator dictionary
        self.iterators[(iter_name, lang1, lang2)] = iterator
        return iterator

    def get_batch(self, iter_name, lang1, lang2=None, stream=False, drop_last=False):
        """
        Return a batch of sentences from a dataset.
        """
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None
        iterator = self.iterators.get((iter_name, lang1, lang2), None)
        # check if the iterator exits. If no, create a new one
        if iterator is None:
            # print("None iterator, create a new one")
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
        # else:
        #     print("Found iterator (%s, %s, %s)" % (iter_name, lang1, lang2))
        # get next batch
        try:
            x = next(iterator)
            if drop_last and x[1].shape[0] == 1:
                # print("Creating new iterator bc drop last. Current x[1]'s shape", x[1].shape)
                iterator = self.get_iterator(iter_name, lang1, lang2, stream)
                x = next(iterator)
        except StopIteration:  # reaching end, restart the iterator.
            print("Stop iteration! Restart iterator")
            iterator = self.get_iterator(iter_name, lang1, lang2, stream)
            x = next(iterator)
        return x if lang2 is None or lang1 < lang2 else x[::-1]

    def get_batch_xy(self, iter_name, lang, splt, drop_last=False):
        """
        Return a batch of ((sentences, lens), ylabels) from a classification dataset.
        Classification data params scheme
        params.clf_dataset = {
            lang: {
            splt: {
                catg: os.path.join(params.data_path, '%s_%s_%s.bpe.pth' % (splt, catg, lang)) for catg in ['x', 'y']
            } for splt in (['train', 'valid'] if lang == 'en' else ['test'])
            } for lang in params.langs if lang in required_mono
        }
        """
        assert lang in self.params.langs
        assert iter_name in self.data
        assert splt in ['valid', 'test'] and lang != 'en' or splt == 'train' and lang == 'en'
        iterator = self.iterators.get((iter_name, lang, splt), None)
        # check if the iterator exits. If no, create a new one
        if iterator is None:
            logger.info("Creating new training data (x,y) iterator (%s) ..." % ','.join([str(x)
                                                                                   for x in [iter_name, lang]
                                                                                   if x is not None]))
            iterator = self.data[iter_name][lang][splt]['x'].get_iterator(
                shuffle=(splt == 'train'), group_by_size=self.params.group_by_size, return_indices=True)
            self.iterators[(iter_name, lang, splt)] = iterator
        # get next batch
        try:
            x, indices = next(iterator)
            if drop_last and x[1].shape[0] == 1:
                iterator = self.data[iter_name][lang][splt]['x'].get_iterator(
                    shuffle=(splt == 'train'), group_by_size=self.params.group_by_size, return_indices=True)
                self.iterators[(iter_name, lang, splt)] = iterator
                x, indices = next(iterator)
        except StopIteration:  # reaching end, restart the iterator.
            # print("Stop iteration! Restart iterator")
            iterator = self.data[iter_name][lang][splt]['x'].get_iterator(
                shuffle=(splt == 'train'), group_by_size=self.params.group_by_size, return_indices=True)
            self.iterators[(iter_name, lang, splt)] = iterator
            x, indices = next(iterator)
        return x, self.data[iter_name][lang][splt]['y'][indices]

    def word_shuffle(self, x, l):
        """
        Randomly shuffle input words.
        """
        if self.params.word_shuffle == 0:
            return x, l

        # define noise word scores
        noise = np.random.uniform(0, self.params.word_shuffle, size=(x.shape[0] - 1, x.shape[1]))
        noise[0] = -1  # do not move start sentence symbol

        assert self.params.word_shuffle > 1
        # paddle's clone method? Try to_variable()
        # x2 = x.clone()
        x2 = x.copy()
        for i in range(l.shape[0]):
            # generate a random permutation
            scores = np.arange(l[i] - 1) + noise[:l[i] - 1, i]
            permutation = scores.argsort()
            # shuffle words
            # can paddle's tensor support index slicing?
            # x2[:l[i] - 1, i].copy_(x2[:l[i] - 1, i][to_variable(permutation)])
            x2[:l[i] - 1, i] = x2[:l[i] - 1, i][permutation]
        return x2, l

    def word_dropout(self, x, l):
        """
        Randomly drop input words.
        """
        if self.params.word_dropout == 0:
            return x, l
        assert 0 < self.params.word_dropout < 1

        # define words to drop
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.shape[0]
        keep = np.random.rand(x.shape[0] - 1, x.shape[1]) >= self.params.word_dropout
        keep[0] = 1  # do not drop the start sentence symbol

        sentences = []
        lengths = []
        for i in range(l.shape[0]):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly drop words from the input
            new_s = [w for j, w in enumerate(words) if keep[j, i]]
            # we need to have at least one word in the sentence (more than the start / end sentence symbols)
            if len(new_s) == 1:
                new_s.append(words[np.random.randint(1, len(words))])
            new_s.append(eos)
            assert len(new_s) >= 3 and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
            lengths.append(len(new_s))
        # re-construct input:
        # l2 = torch.LongTensor(lengths)
        l2 = np.array(lengths, dtype=np.int64)
        # x2 = torch.LongTensor(l2.max(), l2.size(0)).fill_(self.params.pad_index)
        x2 = np.full(shape=(l2.max(), l2.shape[0]), dtype=np.int64, fill_value=self.params.pad_index)
        for i in range(l2.shape[0]):
            # x2[:l2[i], i].copy_(torch.LongTensor(sentences[i]))
            x2[:l2[i], i] = sentences[i]
        return x2, l2

    def word_blank(self, x, l):
        """
        Randomly blank input words.
        """
        if self.params.word_blank == 0:
            return x, l
        assert 0 < self.params.word_blank < 1

        # define words to blank
        eos = self.params.eos_index
        assert (x[0] == eos).sum() == l.shape[0]
        keep = np.random.rand(x.shape[0] - 1, x.shape[1]) >= self.params.word_blank
        keep[0] = 1  # do not blank the start sentence symbol

        sentences = []
        for i in range(l.shape[0]):
            assert x[l[i] - 1, i] == eos
            words = x[:l[i] - 1, i].tolist()
            # randomly blank words from the input
            new_s = [w if keep[j, i] else self.params.mask_index for j, w in enumerate(words)]
            new_s.append(eos)
            assert len(new_s) == l[i] and new_s[0] == eos and new_s[-1] == eos
            sentences.append(new_s)
        # re-construct input
        # x2 = torch.LongTensor(l.max(), l.size(0)).fill_(self.params.pad_index)
        x2 = np.full(shape=(l.max(), l.shape[0]), dtype=np.int64, fill_value=self.params.pad_index)
        for i in range(l.shape[0]):
            # x2[:l[i], i].copy_(torch.LongTensor(sentences[i]))
            x2[:l[i], i] = sentences[i]
        return x2, l

    def add_noise(self, words, lengths):
        """
        Add noise to the encoder input.
        Assume words and lengths are both numpy array
        Note: returned data is numpy
        """
        words, lengths = self.word_shuffle(words, lengths)
        words, lengths = self.word_dropout(words, lengths)
        words, lengths = self.word_blank(words, lengths)
        return words, lengths

    def mask_out(self, x, lengths):
        """
        Decide of random words to mask out, and what target they get assigned.
        """
        params = self.params
        slen, bs = x.shape

        # define target words to predict
        if params.sample_alpha == 0:
            pred_mask = np.random.rand(slen, bs) <= params.word_pred
            pred_mask = pred_mask.astype(np.uint8)
        else:
            x_prob = params.mask_scores[x.flatten()]
            n_tgt = math.ceil(params.word_pred * slen * bs)
            tgt_ids = np.random.choice(len(x_prob), n_tgt, replace=False, p=x_prob / x_prob.sum())
            pred_mask = np.zeros(slen * bs, dtype=np.unit8)
            pred_mask[tgt_ids] = 1
            pred_mask = pred_mask.reshape((slen, bs))

        # do not predict padding
        pred_mask[x == params.pad_index] = 0
        pred_mask[0] = 0  # TODO: remove

        # generate possible targets / update x input
        _x_real = x[pred_mask]
        # _x_rand = _x_real.clone().random_(params.n_words)
        _x_rand = np.random.randint(params.n_words, size=_x_real.shape)
        _x_mask = np.full_like(_x_real, fill_value=params.mask_index)
        # probs = torch.multinomial(params.pred_probs, len(_x_real), replacement=True)
        probs = np.random.choice(len(params.pred_probs), len(_x_real), p=params.pred_probs)
        _x = _x_mask * (probs == 0).astype(np.int64) + _x_real * (probs == 1).astype(np.int64) + \
             _x_rand * (probs == 2).astype(np.int64)
        # x = x.masked_scatter(pred_mask, _x)
        x = _x[pred_mask]

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.shape == (slen, bs)
        assert pred_mask.shape == (slen, bs)

        return x, _x_real, pred_mask

    def generate_batch(self, lang1, lang2, name):
        """
        Prepare a batch (for causal or non-causal mode).
        All baches are in numpy format
        """
        params = self.params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None

        if lang2 is None:
            x, lengths = self.get_batch(name, lang1, stream=True)  # why stream needed?
            positions = None
            # langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            langs = np.full_like(x, fill_value=lang1_id) if params.n_langs > 1 else None
        elif lang1 == lang2:  # encoder-decoder to lang1 itself
            (x1, len1) = self.get_batch(name, lang1)  # why no stream needed?
            (x2, len2) = (x1, len1)
            # injecting noise to input for self-loop translation
            (x1, len1) = self.add_noise(x1, len1)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id,
                                                          params.pad_index, params.eos_index, reset_positions=False)
        else:
            (x1, len1), (x2, len2) = self.get_batch(name, lang1, lang2)
            x, lengths, positions, langs = concat_batches(x1, len1, lang1_id, x2, len2, lang2_id,
                                                          params.pad_index, params.eos_index, reset_positions=True)

        return x, lengths, positions, langs, (None, None) if lang2 is None else (len1, len2)

    def save_checkpoint(self, name, include_optimizers=True):
        """
        Save the model / checkpoints.
        """
        # if not self.params.is_master:
        #     return

        path = os.path.join(self.params.dump_path, '%s_paddle' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'n_total_iter': self.n_total_iter,
            'best_metrics': self.best_metrics,
            'best_stopping_criterion': self.best_stopping_criterion,
        }

        for name in self.MODEL_NAMES:
            logger.warning(f"Saving {name} parameters ...")
            data[name] = getattr(self, name).state_dict()

        if include_optimizers:
            for name in self.optimizers.keys():
                logger.warning(f"Saving {name} optimizer ...")
                data[f'{name}_optimizer'] = self.optimizers[name].state_dict()

        # data['dico_id2word'] = self.data['dico'].id2word
        # data['dico_word2id'] = self.data['dico'].word2id
        # data['dico_counts'] = self.data['dico'].counts
        # data['params'] = {k: v for k, v in self.params.__dict__.items()}

        # torch.save(data, path)
        fluid.save_dygraph(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, 'checkpoint_paddle')
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == '':
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = fluid.load_dygraph(checkpoint_path)

        # reload model parameters
        for name in self.MODEL_NAMES:
            # paddle style load_dict
            getattr(self, name).set_dict(data[name])

        # reload optimizers
        for name in self.optimizers.keys():
            logger.warning(f"Reloading checkpoint optimizer {name}.")
            # paddle style load optimizer
            self.optimizers[name].set_dict(data[f'{name}_optimizer'])

        # reload main metrics
        self.epoch = data['epoch'] + 1
        self.n_total_iter = data['n_total_iter']
        self.best_metrics = data['best_metrics']
        self.best_stopping_criterion = data['best_stopping_criterion']
        logger.warning(f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ...")

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save_checkpoint('periodic-%i' % self.epoch, include_optimizers=False)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        # if not self.params.is_master:
        #     return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning("Metric \"%s\" not found in scores!" % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info('New best score for %s: %.6f' % (metric, scores[metric]))
                self.save_checkpoint('best-%s' % metric, include_optimizers=False)

    def end_epoch(self, scores):
        """
        End the epoch.
        """
        # stop if the stopping criterion has not improved after a certain number of epochs
        # self.params.is_master or not self.stopping_criterion[0].endswith('_mt_bleu')
        if self.stopping_criterion is not None:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info("New best validation score: %f" % self.best_stopping_criterion)
                self.decrease_counts = 0
            else:
                logger.info("Not a better validation score (%i / %i)."
                            % (self.decrease_counts, self.decrease_counts_max))
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info("Stopping criterion has been below its best value for more "
                            "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                # if self.params.multi_gpu and 'SLURM_JOB_ID' in os.environ:
                #     os.system('scancel ' + os.environ['SLURM_JOB_ID'])
                exit()
        # disable it to save space
        # self.save_checkpoint('checkpoint_paddle', include_optimizers=True)
        self.epoch += 1


class EncDecTrainer(Trainer):
    """
    classification data:
    data['clf'][lang][splt]['x']
    data['clf'][lang][splt]['y']

    monolingual data:
    data['mono'][lang][splt] or
    data['mono_stream'][lang][splt]

    parallel data if any:
    data['para'][(src, tgt)][splt]
    """

    def __init__(self, encoder, decoder, discriminator, classifier, data, params):
        #  def __init__(self, encoder, decoder, discriminator, data, params):
        self.MODEL_NAMES = ['encoder', 'decoder', 'discriminator', 'classifier']
        # self.MODEL_NAMES = ['encoder', 'decoder', 'discriminator']

        # model / data / params
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.classifier = classifier
        self.data = data
        self.params = params

        super(EncDecTrainer, self).__init__(data, params)

    def classifier_step(self, mode='train_all'):
        """
        1) Train the classifier on the latent space if mode is train.
        2) Given current enc-dec, compute feedback loss on training data when training enc-dec.
        """
        assert mode in ["train_all", "train_clf", "eval"]
        # 1) Finetune on classifier weights only : 'train_clf'
        # 2) Finetune on classifier and encoder-decoder: 'train_all'
        # 3) Evaluate step, output prediction outcome and loss: "eval"
        if mode == 'train_all':
            self.encoder.train()
            self.decoder.train()
            self.classifier.train()
        elif mode == 'train_clf':
            self.encoder.eval()
            self.decoder.eval()
            self.classifier.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.classifier.eval()

        # train on en data train split only
        if self.params.n_langs == 0:
            raise Exception("No input langs!")
        if 'en' not in self.params.langs:
            raise Exception("English 'en' is not provided in --lgs")
        # training on English only for classifier or computing feedback loss of enc-dec
        lang1, lang1_id = 'en', self.params.lang2id['en']
        lang2 = [x for x in self.params.langs if x != 'en'][0]
        lang2_id = self.params.lang2id[lang2]
        # batch / encode for en.
        (x1, len1), y1 = self.get_batch_xy('clf', lang1, 'train')
        # langs1 = x1.clone().fill_(lang1_id)
        # langs1 = to_variable(np.full_like(x1, fill_value=lang1_id))
        langs1 = fill_constant(shape=x1.shape, dtype="int64", value=lang1_id)
        langs1.stop_gradient = True
        # print(x1, len1, y1)
        x1, len1, y1 = to_cuda(x1, len1, y1)
        if mode == 'train_all':
            enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            # backtranslation
            if self.params.clf_mtv:
                x2, len2 = self.decoder.generate(transpose(enc1, perm=[1, 0, 2]), len1, lang2_id,
                                             max_len=int(1.1 * reduce_max(len1).numpy()[0] + 5))
                # langs2 = x2.clone().fill_(lang2_id)
                # langs2 = to_variable(np.full_like(x2, fill_value=lang2_id))
                langs2 = fill_constant(shape=x2.shape, dtype="int64", value=lang2_id)
                langs2.stop_gradient = True
                # encode generated sentence
                enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
        else:  # train_clf or in eval mode, disable tracking autograd below.

            enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            # backtranslation
            if self.params.clf_mtv:
                x2, len2 = self.decoder.generate(transpose(enc1, perm=[1, 0, 2]), len1, lang2_id,
                                             max_len=int(1.1 * reduce_max(len1).numpy()[0] + 5))
                # langs2 = to_variable(np.full_like(x2, fill_value=lang2_id))
                langs2 = fill_constant(shape=x2.shape, dtype="int64", value=lang2_id)
                langs2.stop_gradient = True
                # encode generated sentence
                enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
        # logits of source lang w/o softmax normalized
        output = self.classifier(enc1, len1)
        # compute cross entropy loss
        # print(output, y1)
        loss = reduce_mean(softmax_with_cross_entropy(output, unsqueeze(y1, axes=-1)))
        # print("classifier step loss:", loss.numpy())
        # del enc1
        if self.params.clf_mtv:
            # Use backtranslation to generate source lang's back-translation
            # F.kl_div(F.log_softmax(logits_source_lang, dim=-1), target_lang_prob, reduction='sum/mean')
            tgt_output = self.classifier(enc2, len2)
            kl_div = kldiv_loss(log(softmax(output) + 1e-10), softmax(tgt_output))
            loss += self.params.lambda_div * kl_div
            del enc2

        if mode != "eval":
            # training classifier, backpropogate loss
            self.stats['clf_loss'].append(loss.numpy()[0])
            # print("Optimizing classifier step loss:")
            self.optimize(loss)
            # print("Finished classifier step")
            self.classifier.clear_gradients()
            self.n_sentences += self.params.batch_size

        self.stats['processed_s'] += len1.shape[0]
        self.stats['processed_w'] += reduce_sum(len1 - 1).numpy()[0]
        return loss

    def discriminator_step(self):
        """
        Train the discriminator on the latent space.
        """
        #  train the full model
        # self.encoder.eval()
        # self.decoder.eval()
        # self.discriminator.train()

        # IF not training mt_step or bt_step, have to tune enc-dec here
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.train()
        # train on monolingual data only
        if self.params.n_langs == 0:
            raise Exception("No data to train the discriminator!")

        # batch / encode
        encoded = []
        for lang_id, lang in enumerate(self.params.langs):
            x1, len1 = self.get_batch('dis', lang)
            # (x1, len1), _ = self.get_batch_xy('clf', lang, 'train')
            # langs1 = x1.clone().fill_(lang_id)
            langs1 = fill_constant(shape=x1.shape, dtype="int64", value=lang_id)
            x1, len1 = to_cuda(x1, len1)
            # logger.info("len1 is %i" % len(len1))
            # get transformer last hidden layer
            # tensor = self.model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)

            encoded_h = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            del langs1
            # logger.info("encoder output dim ", encoded_h.size())
            encoded.append(encoded_h)

        # discriminator data setup
        dis_inputs = [reshape(x, shape=[-1, x.shape[-1]]) for x in encoded]
        # for each token, there is a label, hence keep all the transformer's output
        ntokens = [dis_input.shape[0] for dis_input in dis_inputs]
        # why do we need to contact 0 here?
        # change list of torch tensors to torch tensor
        # encoded = torch.cat(dis_inputs, 0)
        encoded = concat(dis_inputs, 0)
        predictions = self.discriminator(encoded.detach())
        # logger.info('dis pred is %s' % predictions[0:5,:])

        del encoded, encoded_h
        # loss
        # dis_target = torch.cat([torch.zeros(sz).fill_(i) for i, sz in enumerate(ntokens)])

        # dis_target = np.concatenate([np.zeros(sz, dtype="int64").fill(i) for i, sz in enumerate(ntokens)])
        # y = to_variable(dis_target)
        y = concat([zeros([sz], dtype="int64")+i for i, sz in enumerate(ntokens)])
        y.stop_gradient = True
        # y = to_variable(dis_target)
        # logger.info('dis y dimmension is %i' % len(y))
        # print("discriminator pred:", predictions)
        # print("y:", y)
        loss = reduce_mean(softmax_with_cross_entropy(predictions, unsqueeze(y, axes=-1)))
        del y
        # print("discriminator step loss:", loss.numpy())
        # logger.info('dis loss is %s' % loss)
        self.stats['dis_loss'].append(loss.numpy()[0])

        # optimizer
        self.optimize(loss)
        self.discriminator.clear_gradients()
        self.n_sentences += self.params.batch_size
        self.stats['processed_s'] += len1.shape[0]
        self.stats['processed_w'] += reduce_sum(len1 - 1).numpy()[0]

    def mt_step(self, lang1, lang2, lambda_coeff):
        """
        Machine translation step.
        Can also be used for denoising auto-encoding.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        params = self.params
        self.encoder.train()
        self.decoder.train()
        if self.discriminator:
            self.discriminator.eval()
        if self.classifier:
            self.classifier.eval()

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        batch_flag = 'ae'

        # generate batch
        if lang1 == lang2:  # auto-encoder
            # get a batch from unlabeled monolingual data
            if random() > params.sample_clf_prob and lang1 == 'en':
                (x1, len1), y1 = self.get_batch_xy('clf', lang1, 'train')
                y1 = to_variable(y1)
                y1.stop_gradient = True
                batch_flag = 'clf'
            else:
                (x1, len1) = self.get_batch('ae', lang1)
            (x2, len2) = (x1, len1)
            (x1, len1) = self.add_noise(x1, len1)
        else:
            (x1, len1), (x2, len2) = self.get_batch('mt', lang1, lang2)
        # langs1 = x1.clone().fill_(lang1_id)
        # langs1 = np.full_like(x1, fill_value=lang1_id)
        langs1 = fill_constant(shape=x1.shape, dtype="int64", value=lang1_id)
        langs1.stop_gradient = True
        # langs2 = x2.clone().fill_(lang2_id)
        # langs2 = np.full_like(x2, fill_value=lang2_id)
        langs2 = fill_constant(shape=x2.shape, dtype="int64", value=lang2_id)
        langs2.stop_gradient = True
        # target words to predict
        # alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        alen = np.arange(len2.max(), dtype=np.int64)
        # alen = fluid.layers.range(0, len2.max(), step=1, dtype="int64")
        pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
        # pred_mask = expand(unsqueeze(alen, axes=0), expand_times=[len2.shape[0]-1, 1]) < unsqueeze(len2[:-1] - 1, axes=1)
        # print("After pred_mask")
        # y = x2[1:].masked_select(pred_mask[:-1])
        y = x2[1:][pred_mask[:-1]]
        # y = masked_select(x2[1:], pred_mask)
        assert y.shape[0] == (len2 - 1).sum()
        x1, len1, x2, len2, y, pred_mask = to_cuda(x1, len1, x2, len2, y, pred_mask)

        # encode source sentence
        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        enc1 = transpose(enc1, perm=[1, 0, 2])

        # decode target sentence
        dec2 = self.decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

        # loss for a batch of unlabeled monolingual data
        _, loss = self.decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=False)
        loss = lambda_coeff * loss
        del langs1, langs2, pred_mask
        # discriminator feedback loss

        if params.lambda_dis > 0:
            predictions = self.discriminator(reshape(enc1, shape=[-1, enc1.shape[-1]]))
            # fake_y = torch.LongTensor(predictions.size(0)).random_(1, params.n_langs)
            fake_y = to_variable(np.random.randint(1, params.n_langs, size=predictions.shape[0]))
            fake_y = (fake_y + lang1_id) % params.n_langs
            fake_y.stop_gradient = True
            dis_loss = reduce_mean(softmax_with_cross_entropy(predictions, unsqueeze(fake_y, axes=-1)))
            loss += params.lambda_dis * dis_loss

        if lang1 == lang2 and params.lambda_clf > 0 and batch_flag == 'clf' and lang1 == 'en':
            # calculate clf loss on a batch of labeled English data with current encoder-decoder
            output = self.classifier(transpose(enc1, [1, 0, 2]), len1)
            # compute cross entropy loss
            assert output.shape[0] == y1.shape[0]
            # clf_loss = F.cross_entropy(output, y1)
            clf_loss = reduce_mean(softmax_with_cross_entropy(output, unsqueeze(y1, axes=-1)))
            if not (loss != loss).detach().numpy().any():  # not NaN loss
                loss += params.lambda_clf * clf_loss

        del enc1
        # Classifier feedback loss is computed on English only for now
        # Try update on English only or universal update since encoder-decoder is shared among langs
        # if lang1 == lang2 and params.lambda_clf > 0:  # and lang1 == 'en'
        #     # calculate clf loss on a batch of labeled English data with current encoder-decoder
        #     clf_loss = self.classifier_step(mode='eval')
        #     loss += params.lambda_clf * clf_loss

        # logging overall loss:
        # if lang1 == 'en':
        self.stats[('AE-DIS-CLF-%s' % lang1) if lang1 == lang2 else ('MT-%s-%s' % (lang1, lang2))].append(
            loss.numpy()[0])
        # else:
        #    self.stats[('AE-DIS-%s' % lang1) if lang1 == lang2 else ('MT-%s-%s' % (lang1, lang2))].append(loss.item())
        # optimize the overall loss
        self.optimize(loss)
        self.encoder.clear_gradients()
        self.decoder.clear_gradients()
        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len2.shape[0]
        self.stats['processed_w'] += reduce_sum(len2 - 1).numpy()[0]

    def bt_step(self, lang1, lang2, lang3, lambda_coeff):
        """
        Back-translation step for machine translation.
        """
        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return
        assert lang1 == lang3 and lang1 != lang2 and lang2 is not None
        params = self.params

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]
        batch_flag = 'bt'

        # generate source batch
        if lang1 == 'en' and random() > params.sample_clf_prob:
            (x1, len1), y1 = self.get_batch_xy('clf', lang1, 'train')
            y1 = to_variable(y1)
            y1.stop_gradient = True
            batch_flag = 'clf'
        else:
            x1, len1 = self.get_batch('bt', lang1)
        langs1 = fill_constant(shape=x1.shape, dtype="int64", value=lang1_id)
        langs1.stop_gradient = True

        # cuda
        x1, len1 = to_cuda(x1, len1)
        dis_loss = 0.0
        # generate a translation: no_grad() no paddle mapping func.

        # print("Before encoder")
        # evaluation mode
        self.encoder.eval()
        self.decoder.eval()
        if self.discriminator:
            self.discriminator.eval()
        if self.classifier:
            self.classifier.eval()
        # encode source sentence and translate it
        enc1 = self.encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
        # print("After encoder \n Before decoder")
        enc1 = transpose(enc1, perm=[1, 0, 2])
        x2, len2 = self.decoder.generate(enc1, len1, lang2_id, max_len=int(1.1 * reduce_max(len1).numpy()[0] + 5))
        # print("After decoder \nSetting up discriminator")
        # langs2 = x2.clone().fill_(lang2_id)
        # langs2 = to_variable(np.full_like(x2, fill_value=lang2_id))
        langs2 = fill_constant(shape=x2.shape, dtype="int64", value=lang2_id)
        langs2.stop_gradient = True
        # discriminator feedback loss
        if params.lambda_dis > 0:
            predictions = self.discriminator(reshape(enc1, shape=[-1, enc1.shape[-1]]))
            fake_y = to_variable(np.random.randint(1, params.n_langs, size=predictions.shape[0]))
            fake_y = (fake_y + lang1_id) % params.n_langs
            fake_y.stop_gradient = True
            dis_loss = reduce_mean(softmax_with_cross_entropy(predictions, unsqueeze(fake_y, axes=-1)))
            del fake_y

        # Classifier feedback loss is computed on English only
        if params.lambda_clf > 0 and lang1 == 'en' and batch_flag == 'clf':
            # calculate clf loss on a batch of labeled English data with current encoder-decoder
            output = self.classifier(transpose(enc1, [1, 0, 2]), len1)
            # compute cross entropy loss
            assert output.shape[0] == y1.shape[0]
            # clf_loss = F.cross_entropy(output, y1)
            clf_loss = reduce_mean(softmax_with_cross_entropy(output, unsqueeze(y1, axes=-1)))

        del enc1
        # print("Setting encoder, decoder train status")
        # training mode
        self.encoder.train()
        self.decoder.train()

        # print("Before encoder enc2")
        # encode generated sentence
        enc2 = self.encoder('fwd', x=x2, lengths=len2, langs=langs2, causal=False)
        enc2 = transpose(enc2, perm=[1, 0, 2])
        # print("After encoder enc2")
        # words to predict
        # alen = torch.arange(len1.max(), dtype=torch.long, device=len1.device)
        alen = np.arange(reduce_max(len1).numpy()[0], dtype=np.int64)
        pred_mask = alen[:, None] < len1.numpy()[None] - 1  # do not predict anything given the last target word
        # y1 = x1[1:].masked_select(pred_mask[:-1])
        # y1 = masked_select(x1[1:], pred_mask[:-1])
        y1 = x1.numpy()[1:][pred_mask[:-1]]
        y1, pred_mask = to_cuda(y1, pred_mask)
        # decode original sentence
        # print("Before decoder dec3")
        dec3 = self.decoder('fwd', x=x1, lengths=len1, langs=langs1, causal=True, src_enc=enc2, src_len=len2)
        # print("After decoder dec3")
        del enc2
        # loss
        _, loss = self.decoder('predict', tensor=dec3, pred_mask=pred_mask, y=y1, get_scores=False)

        del dec3, langs1, langs2
        if params.lambda_dis > 0:
            loss += params.lambda_dis * dis_loss

        if params.lambda_clf > 0 and batch_flag == 'clf' and not (clf_loss != clf_loss).detach().numpy().any():
            loss += params.lambda_clf * clf_loss

        # if lang1 == 'en':
        self.stats[('BT-DIS-CLF-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.numpy()[0])
        # else:
        #    self.stats[('BT-DIS-%s-%s-%s' % (lang1, lang2, lang3))].append(loss.item())

        # optimize
        # print("decoder:", loss.numpy()[0])
        self.optimize(loss)
        self.encoder.clear_gradients()
        self.decoder.clear_gradients()
        # del dis_loss, clf_loss, loss
        # number of processed sentences / words
        self.n_sentences += params.batch_size
        self.stats['processed_s'] += len1.shape[0]
        self.stats['processed_w'] += reduce_sum(len1 - 1).numpy()[0]
        return loss
