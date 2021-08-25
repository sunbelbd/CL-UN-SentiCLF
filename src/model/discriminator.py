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

import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, LayerList


# def Linear(in_features, out_features, bias=True):
#     m = paddle.fluid.dygraph.Linear(in_features, out_features, bias_attr=bias)
#     return m


class Discriminator(fluid.dygraph.Layer):
    DIS_ATTR = ['input_dim', 'dis_layers', 'dis_hidden_dim', 'dis_dropout']

    def __init__(self, params):
        """
        Discriminator initialization.
        """
        super(Discriminator, self).__init__()

        self.n_langs = params.n_langs
        # self.input_dim = params.hidden_dim if params.attention else params.enc_dim
        self.input_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hidden_dim = params.dis_hidden_dim
        self.dis_dropout = params.dis_dropout

        self.layers = LayerList()
        for i in range(self.dis_layers + 1):
            if i == 0:
                input_dim = self.input_dim
                # input_dim *= (2 if params.attention and not params.dis_input_proj else 1)
            else:
                input_dim = self.dis_hidden_dim
            output_dim = self.dis_hidden_dim if i < self.dis_layers else self.n_langs
            if i < self.dis_layers:
                self.layers.append(Linear(input_dim, output_dim, bias_attr=True, act='leaky_relu'))
            else:
                self.layers.append(Linear(input_dim, output_dim, bias_attr=True))
        # self.layers = Sequential(layers)  ##* unzip the list

    def forward(self, x):
        # return self.layers(input)
        for i in range(self.dis_layers + 1):
            x = self.layers[i](x)
            if i < self.dis_layers:
                x = fluid.layers.dropout(x, dropout_prob=self.dis_dropout)
        return x

    def load_state_dict(self, state_dict, keys):
        """
        Copy pretrained transformer parameters in state_dict to current transformer object.
        Note: match different keys between pytorch and paddle
        :param state_dict: pytorch state_dict obj
        :return:
        """
        # keys = state_dict.keys()
        for k in keys:
            if "torch" not in str(type(state_dict[k])):
                continue
            if ("lin" in k or "pred_layer" in k or "layers" in k) and "weight" in k:
                state_dict[k] = state_dict[k].data.cpu().numpy().transpose()
            else:
                state_dict[k] = state_dict[k].data.cpu().numpy()
        self.set_dict(state_dict)
