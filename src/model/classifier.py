#  Copyright (c) 2020-present, Baidu, Inc.
#  All rights reserved.
#  #
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#  #
#  Acknowledgement: The code is modified based on Facebook AI's XLM.

from logging import getLogger

import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, LayerList, BatchNorm

logger = getLogger()


class Classifier(fluid.dygraph.Layer):
    CLA_ATTR = ['input_dim', 'clf_layers', 'clf_hidden_dim', 'clf_output_dim', 'clf_dropout', 'clf_batch_norm']

    def __init__(self, params):
        """
        Classifier initialization.
        """
        super(Classifier, self).__init__()
        # self.input_dim = params.hidden_dim if params.attention else params.enc_dim
        self.input_dim = params.emb_dim
        self.clf_layers = params.clf_layers
        self.clf_hidden_dim = params.clf_hidden_dim
        self.clf_output_dim = params.clf_output_dim
        self.clf_dropout = params.clf_dropout
        self.clf_batch_norm = params.clf_batch_norm

        assert self.clf_layers >= 0, 'Invalid layer numbers'
        self.net = LayerList()
        self.bm = LayerList()
        for i in range(self.clf_layers):
            if i == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.clf_hidden_dim
            self.net.append(Linear(input_dim, self.clf_hidden_dim, act="leaky_relu"))
            if self.clf_batch_norm:
                self.bm.append(BatchNorm(self.clf_hidden_dim))

        last_input_dim = self.clf_hidden_dim if self.clf_layers > 0 else self.input_dim
        # Note: our output layer is not softmax-normalized, but just logits
        self.net.append(Linear(last_input_dim, self.clf_output_dim, bias_attr=True))
        # self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, inputs, lengths):
        """
        :param inputs: (srlen, bs, emb_dim)
        :param lengths: (bs)
        :return:
        """
        assert self.input_dim == inputs.shape[2]
        # project the single-vector sentence representation (first column of last layer)
        # output = self.net(inputs[0])
        x = inputs[0]
        for i in range(self.clf_layers):
            x = self.net[i](x)
            if self.clf_batch_norm:
                x = self.bm[i](x)
            x = fluid.layers.dropout(x, dropout_prob=self.clf_dropout)
        x = self.net[self.clf_layers](x)
        return x
