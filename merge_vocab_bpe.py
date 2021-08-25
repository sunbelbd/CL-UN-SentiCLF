#  Copyright (c) 2020-present, Baidu, Inc.
#  All rights reserved.
#  #
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#  #
#  Acknowledgement: The code is modified based on Facebook AI's XLM.

import sys

from src.data.dictionary import Dictionary

target_bpe = "pretrain/pretrain_xlm_17/codes_xnli_17"
target_vocab = "pretrain/pretrain_xlm_17/vocab_xnli_17"

if __name__ == '__main__':
    src_bpe = sys.argv[2]
    src_vocab = sys.argv[1]
    Dictionary.merge_vocab(src_vocab, target_vocab, src_bpe, target_bpe)
