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
from logging import getLogger

import numpy as np
from paddle.fluid.dygraph import to_variable

from ..utils import to_cuda

logger = getLogger()


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params

    def get_iterator(self, data_set, lang1, lang2=None, stream=False):
        """
        Create a new iterator for a dataset.
        """
        assert data_set in ['valid', 'test']
        assert lang1 in self.params.langs
        assert lang2 is None or lang2 in self.params.langs
        assert stream is False or lang2 is None

        n_sentences = -1
        subsample = 1

        if lang2 is None:
            if stream:
                iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                iterator = self.data['mono'][lang1][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences,
                )
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                shuffle=False,
                group_by_size=True,
                n_sentences=n_sentences
            )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def run_all_evals(self, trainer, epoch):
        """
        Run all evaluations.
        """
        params = self.params
        scores = OrderedDict({'epoch': epoch})

        for data_set in ['valid', 'test']:
            # classification task evaluation, disable all other evaluations to save memory
            for lang in params.langs:
                if lang == 'en' and data_set == 'valid' or lang != 'en' and data_set == 'test':
                    self.evaluate_clf(scores, data_set, lang)

        return scores


class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super(EncDecEvaluator, self).__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder
        self.classifier = trainer.classifier

    def evaluate_clf(self, scores, splt, lang):
        """
        :param scores: result storage
        :param splt: data split name
        :param lang: language
        :return:
        """
        params = self.params
        data = self.data
        assert lang == 'en' and splt == 'valid' or lang != 'en' and splt == 'test'
        self.encoder.eval()
        self.classifier.eval()

        encoder = self.encoder
        classifier = self.classifier
        # Trim too long sentences in validation and test data to avoid emb overflow in transfomer's
        data['clf'][lang][splt]['x'].trim_long_sentences(params.max_positions)
        iterator = data['clf'][lang][splt]['x'].get_iterator(
            shuffle=False, group_by_size=self.params.group_by_size, return_indices=True)
        lang_id = params.lang2id[lang]
        valid, total = 0, 0
        for batch in iterator:
            (x, lengths), idx = batch
            y = data['clf'][lang][splt]['y'][idx]
            langs = to_variable(np.full_like(x, fill_value=lang_id))
            # cuda
            x, lengths = to_cuda(x, lengths)
            encx = encoder('fwd', x=x, lengths=lengths, langs=langs, causal=False)

            # forward
            output = classifier(encx, lengths)
            predictions = output.numpy().argmax(axis=1)
            # print(predictions)

            # update statistics
            valid += (predictions == y).sum()
            total += len(y)

        # compute accuracy
        acc = 100.0 * valid / total
        scores['%s_%s_clf_acc' % (splt, lang)] = acc
        # logger.info("CLF - %s - %s - Epoch %i - Acc: %.1f%%" % (splt, lang, self.epoch, acc))
