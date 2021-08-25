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

from collections import OrderedDict

import copy
import json
import os
import time
import torch
import torch.nn.functional as F
from logging import getLogger
from torch import nn

from ..data.dataset import Dataset
from ..data.loader import load_binarized, set_dico_parameters
from ..optim import get_optimizer
from ..utils import truncate, to_cuda

logger = getLogger()


class CLF:

    def __init__(self, embedder, scores, params):
        """
        Initialize XNLI trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        self.params = params
        self.scores = scores
        self.decrease_counts_max = 5
        self.decrease_counts = 0

    def get_iterator(self, splt, lang):
        """
        Get a monolingual data iterator.
        """
        assert lang != 'en' and splt == 'test' or splt in ['valid', 'train'] and lang == 'en'
        return self.data[lang][splt]['x'].get_iterator(
            shuffle=(splt == 'train'),
            group_by_size=self.params.group_by_size,
            return_indices=True
        )

    def save_checkpoint(self, name):
        """
        Save the encoder and classifier weights / checkpoints.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            'epoch': self.epoch,
            'best_metrics': self.scores
        }

        logger.warning("Saving model parameters ...")
        data['model'] = self.encoder.model.state_dict()
        data['classifier'] = self.proj
        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        # print(self.encoder.pretrain_params)
        data['params'] = self.encoder.pretrain_params.update({k: v for k, v in self.params.__dict__.items()})

        torch.save(data, path)

    def compare_dict(self, dict1, dict2):
        not_shown, mismatch = 0, 0
        for i in range(len(dict1)):
            if dict1.id2word[i] != dict2[i]:
                mismatch += 1
            if dict1.id2word[i] not in dict2:
                not_shown += 1
        print("Total mismatch %d, not shown %d" % (mismatch, not_shown))

    def run(self):
        """
        Run XNLI training / evaluation.
        """
        params = self.params

        # load data
        self.data = self.load_data()
        # check if loaded classification data set is using the same dict as pretrained model
        if not self.data['dico'] == self._embedder.dico:
            self.compare_dict(self.data['dico'], self._embedder.dico)
            raise Exception(("Dictionary in evaluation data (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']),
                                                                                    len(self._embedder.dico)))

        # embedder
        self.encoder = copy.deepcopy(self._embedder)
        self.encoder.cuda()

        # projection layer: CHANGE 3 to your number of classes output
        self.proj = nn.Sequential(*[
            nn.Dropout(params.dropout),
            nn.Linear(self.encoder.out_dim, params.clf_output_dim)
        ]).cuda()

        # optimizers: use different optimizers to tune embedding layer and projection layer
        self.optimizer_e = get_optimizer(list(self.encoder.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.proj.parameters(), params.optimizer_p)
        best_acc = 0
        eval_metric = "CLF_valid_en_acc"
        # train and evaluate the model
        for epoch in range(params.n_epochs):
            # update epoch
            self.epoch = epoch

            # training
            logger.info("CLF - Training epoch %i ..." % epoch)
            self.train()

            # evaluation
            logger.info("CLF - Evaluating epoch %i ..." % epoch)
            with torch.no_grad():
                scores = self.eval()
                if scores[eval_metric] > best_acc:
                    logger.info('New best score for %s: %.6f' % (eval_metric, scores[eval_metric]))
                    self.save_checkpoint('best-%s' % eval_metric)
                    self.decrease_counts = 0
                    best_acc = scores[eval_metric]
                else:
                    logger.info("Not a better validation score (%i / %i)."
                                % (self.decrease_counts, self.decrease_counts_max))
                    self.decrease_counts += 1
                if self.decrease_counts > self.decrease_counts_max:
                    logger.info("Stopping criterion has been below its best value for more "
                                "than %i epochs. Ending the experiment..." % self.decrease_counts_max)
                    exit()
                self.scores.update(scores)

    def train(self):
        """
        Finetune for one epoch on the English training set.
        """
        params = self.params
        self.encoder.train()
        self.proj.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        t = time.time()

        iterator = self.get_iterator('train', 'en')
        lang, lang_id = 'en', params.lang2id['en']
        while True:
            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break
            (sent1, len1), idx = batch
            x, lengths = truncate(sent1, len1, params.max_len, params.eos_index)
            lang_ids = x.clone().fill_(lang_id)

            y = self.data['en']['train']['y'][idx]
            bs = len(len1)

            # cuda
            x, y, lengths, lang_ids = to_cuda(x, y, lengths, lang_ids)

            # loss
            output = self.proj(self.encoder.get_embeddings(x, lengths, langs=lang_ids))
            loss = F.cross_entropy(output, y)

            # backward / optimization
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            losses.append(loss.item())

            # log
            if ns % (100 * bs) < bs:
                logger.info("CLF - Epoch %i - Train iter %7i - %.1f words/s - Loss: %.4f" % (
                    self.epoch, ns, nw / (time.time() - t), sum(losses) / len(losses)))
                nw, t = 0, time.time()
                losses = []

            # epoch size
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def eval(self):
        """
        Evaluate on XNLI validation and test sets, for all languages.
        """
        params = self.params
        langs = ['en', params.target_lang]
        self.encoder.eval()
        self.proj.eval()

        scores = OrderedDict({'epoch': self.epoch})

        for splt in ['valid', 'test']:

            for lang in langs:
                if lang == 'en' and splt == 'test' or lang != 'en' and splt == 'valid':
                    continue
                lang_id = params.lang2id[lang if lang != 'jp' else 'ja']
                valid = 0
                total = 0

                for batch in self.get_iterator(splt, lang):
                    # batch
                    (sent1, len1), idx = batch
                    # set max length to 256, avoid position embedding overflow and save time.
                    x, lengths = truncate(sent1, len1, 256, params.eos_index)
                    lang_ids = x.clone().fill_(lang_id)

                    y = self.data[lang][splt]['y'][idx]

                    # cuda
                    x, y, lengths, lang_ids = to_cuda(x, y, lengths, lang_ids)

                    # forward
                    output = self.proj(self.encoder.get_embeddings(x, lengths, langs=lang_ids))
                    predictions = output.data.max(1)[1]

                    # update statistics
                    valid += predictions.eq(y).sum().item()
                    total += len(len1)

                # compute accuracy
                acc = 100.0 * valid / total
                scores['CLF_%s_%s_acc' % (splt, lang)] = acc
                logger.info("CLF - %s - %s - Epoch %i - Acc: %.1f%%" % (splt, lang, self.epoch, acc))

        logger.info("__log__:%s" % json.dumps(scores))
        return scores

    def load_data(self):
        """
        Load XNLI cross-lingual classification data.
        """
        params = self.params
        catg = params.data_category
        langs = ['en', params.target_lang]
        data = {lang: {splt: {} for splt in (['train', 'valid'] if lang == 'en' else ['test'])} for lang in langs}
        clf_dataset_path = {
            lang: {
                splt: {
                    'x': os.path.join(params.data_path, '%s_%s_%s_x.bpe.pth' % (splt, lang, catg)),
                    'y': os.path.join(params.data_path, '%s_%s_%s_y.txt' % (splt, lang, catg)),
                } for splt in (['train', 'valid'] if lang == 'en' else ['test'])
            } for lang in langs
        }
        for splt in ['train', 'valid', 'test']:
            for lang in langs:
                if lang == 'en' and splt in ['train', 'valid'] or lang != 'en' and splt == 'test':
                    # load data and dictionary
                    data1 = load_binarized(clf_dataset_path[lang][splt]['x'], params)
                    data['dico'] = data.get('dico', data1['dico'])
                    # set dictionary parameters
                    set_dico_parameters(params, data, data1['dico'])
                    # create dataset
                    data[lang][splt]['x'] = Dataset(data1['sentences'], data1['positions'], params)
                    # load labels
                    with open(clf_dataset_path[lang][splt]['y'], 'r') as f:
                        labels = [int(l) for l in f]
                    data[lang][splt]['y'] = torch.LongTensor(labels)
                    assert len(data[lang][splt]['x']) == len(data[lang][splt]['y'])

        return data
