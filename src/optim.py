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


import re

import paddle.fluid as fluid


def get_optimizer(s, parameter_list):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}
        optim_params['lr'] = 0.00001
    # fluid.optimizer.AdamOptimizer
    if method == 'adam':
        optim_fn = fluid.optimizer.AdamOptimizer
    elif method == 'sgd':
        optim_fn = fluid.optimizer.SGD
    # fluid.optimizer.SGD
    else:
        raise Exception("We only support sgd and adam now. Feel free to add yours!")
    assert 'lr' in optim_params

    return optim_fn(learning_rate=optim_params['lr'], parameter_list=parameter_list)
