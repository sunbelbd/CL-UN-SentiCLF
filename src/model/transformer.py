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
import itertools
import math
from logging import getLogger

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import LayerNorm, to_variable, LayerList
from paddle.fluid.layers import reduce_max, reduce_sum, transpose, cast, squeeze, topk, \
    reshape, cross_entropy, concat, matmul, expand_as, dropout, unsqueeze, fill_constant, expand, \
    softmax_with_cross_entropy, scatter, where

from src.utils import masked_select

N_MAX_POSITIONS = 512  # maximum input sequence length

DECODER_ONLY_PARAMS = [
    'layer_norm15.%i.weight', 'layer_norm15.%i.bias',
    'encoder_attn.%i.q_lin.weight', 'encoder_attn.%i.q_lin.bias',
    'encoder_attn.%i.k_lin.weight', 'encoder_attn.%i.k_lin.bias',
    'encoder_attn.%i.v_lin.weight', 'encoder_attn.%i.v_lin.bias',
    'encoder_attn.%i.out_lin.weight', 'encoder_attn.%i.out_lin.bias'
]

TRANSFORMER_LAYER_PARAMS = [
    'attentions.%i.q_lin.weight', 'attentions.%i.q_lin.bias',
    'attentions.%i.k_lin.weight', 'attentions.%i.k_lin.bias',
    'attentions.%i.v_lin.weight', 'attentions.%i.v_lin.bias',
    'attentions.%i.out_lin.weight', 'attentions.%i.out_lin.bias',
    'layer_norm1.%i.weight', 'layer_norm1.%i.bias',
    'ffns.%i.lin1.weight', 'ffns.%i.lin1.bias',
    'ffns.%i.lin2.weight', 'ffns.%i.lin2.bias',
    'layer_norm2.%i.weight', 'layer_norm2.%i.bias'
]

logger = getLogger()


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    """
    Change to paddle flud embedding
    :param num_embeddings:
    :param embedding_dim:
    :param padding_idx:
    :return:
    """
    m = fluid.dygraph.Embedding(size=[num_embeddings, embedding_dim], padding_idx=padding_idx,
                                param_attr=fluid.ParamAttr(
                                    initializer=fluid.initializer.Normal(0., embedding_dim ** -0.5)))
    # if padding_idx is not None:
    #     # set padding vector to 0
    #     m.weight[padding_idx] *= 0.0
    return m


def Linear(in_features, out_features, bias=True):
    m = fluid.dygraph.Linear(in_features, out_features, bias_attr=bias)
    # nn.init.normal_(m.weight, mean=0, std=1)
    # nn.init.xavier_uniform_(m.weight)
    # nn.init.constant_(m.bias, 0.)
    return m


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    # out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    # out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out_tmp = out.numpy()
    out_tmp[:, 0::2] = np.sin(position_enc[:, 0::2])
    out_tmp[:, 1::2] = np.cos(position_enc[:, 1::2])
    # paddle also has detach method
    out = to_variable(out_tmp)
    out = out.detach()
    # out.requires_grad = False
    out.stop_gradient = True


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return 0.5 * x * (1.0 + fluid.layers.erf(x / math.sqrt(2.0)))


def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert reduce_max(lengths) <= slen
    bs = lengths.shape[0]
    alen = fluid.layers.range(0, slen, step=1, dtype='int64')
    # print(alen, unsqueeze(lengths, axes=1))
    mask = expand(unsqueeze(alen, axes=0), expand_times=[lengths.shape[0], 1]) < unsqueeze(lengths, axes=1)

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        # attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
        attn_mask = expand(unsqueeze(alen, axes=[0, 1]), expand_times=[bs, slen, 1]) <= unsqueeze(alen, axes=[0, 2])
    else:
        attn_mask = mask

    # sanity check
    assert mask.shape == [bs, slen]
    assert causal is False or attn_mask.shape == [bs, slen, slen]

    return mask, attn_mask


class PredLayer(fluid.dygraph.Layer):
    """
    Prediction layer (cross_entropy or adaptive_softmax).
    """

    def __init__(self, params):
        super(PredLayer, self).__init__()
        # not supporting adaptive softmax
        self.n_words = params.n_words
        self.pad_index = params.pad_index
        dim = params.emb_dim
        # if params.asm is False:
        self.proj = Linear(dim, params.n_words, bias=True)
        # else:
        #     self.proj = nn.AdaptiveLogSoftmaxWithLoss(
        #         in_features=dim,
        #         n_classes=params.n_words,
        #         cutoffs=params.asm_cutoffs,
        #         div_value=params.asm_div_value,
        #         head_bias=True,  # default is False
        #     )

    def forward(self, x, y, get_scores=False):
        """
        Compute the loss, and optionally the scores.
        """
        assert reduce_sum(cast(y == self.pad_index, "int64")).numpy()[0] == 0

        # if self.asm is False:
        scores = reshape(self.proj(x), [-1, self.n_words])
        loss = softmax_with_cross_entropy(scores, unsqueeze(y, axes=-1))
        # else:
        #     _, loss = self.proj(x, y)
        #     scores = self.proj.log_prob(x) if get_scores else None

        return scores, loss

    def get_scores(self, x):
        """
        Compute scores.
        """
        assert len(x.shape) == 2
        # return self.proj.log_prob(x) if self.asm else self.proj(x)
        return self.proj(x)


class MultiHeadAttention(fluid.dygraph.Layer):
    NEW_ID = itertools.count()

    def __init__(self, n_heads, dim, dropout):
        super(MultiHeadAttention, self).__init__()
        self.layer_id = next(MultiHeadAttention.NEW_ID)
        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.dim % self.n_heads == 0

        self.q_lin = Linear(dim, dim)
        self.k_lin = Linear(dim, dim)
        self.v_lin = Linear(dim, dim)
        self.out_lin = Linear(dim, dim)

    def forward(self, input, mask, kv=None, cache=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        [bs, qlen, dim] = input.shape
        if kv is None:
            klen = qlen if cache is None else cache['slen'] + qlen
        else:
            klen = kv.shape[1]
        assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        mask_reshape = [bs, 1, qlen, klen] if len(mask.shape) == 3 else [bs, 1, 1, klen]

        def shape(x):
            """  projection """
            # return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
            x = reshape(x=x, shape=[bs, -1, self.n_heads, dim_per_head])
            x = transpose(x=x, perm=[0, 2, 1, 3])
            return x

        def unshape(x):
            """  compute context """
            # return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            x = transpose(x=x, perm=[0, 2, 1, 3])
            x = reshape(x=x, shape=[bs, -1, self.n_heads * dim_per_head])
            return x

        q = shape(self.q_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k_lin(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k_lin(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v_lin(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = concat([k_, k], axis=2)  # (bs, n_heads, klen, dim_per_head)
                    v = concat([v_, v], axis=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, qlen, dim_per_head)
        scores = matmul(q, transpose(x=k, perm=[0, 1, 3, 2]))  # (bs, n_heads, qlen, klen)
        # mask = (mask == 0).view(mask_reshape).expand_as(scores)               # (bs, n_heads, qlen, klen)
        mask = expand_as(reshape(cast(mask, "float32"), shape=mask_reshape), scores)  # (bs, n_heads, qlen, klen)
        mask.stop_gradient = True
        # scores.masked_fill_(mask, -float('inf'))                              # (bs, n_heads, qlen, klen)
        scores += fluid.layers.log(mask)

        # weights = F.softmax(scores.float(), dim=-1).type_as(scores)           # (bs, n_heads, qlen, klen)
        weights = fluid.layers.softmax(scores, axis=-1)  # (bs, n_heads, qlen, klen)
        weights = fluid.layers.dropout(weights, dropout_prob=self.dropout,
                                       is_test=False)  # (bs, n_heads, qlen, klen)
        context = matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        return self.out_lin(context)


class TransformerFFN(fluid.dygraph.Layer):

    def __init__(self, in_dim, dim_hidden, out_dim, dropout, gelu_activation):
        super(TransformerFFN, self).__init__()
        self.dropout = dropout
        self.lin1 = Linear(in_dim, dim_hidden)
        self.lin2 = Linear(dim_hidden, out_dim)
        self.act = gelu if gelu_activation else fluid.layers.relu

    def forward(self, input):
        x = self.lin1(input)
        x = self.act(x)
        x = self.lin2(x)
        x = fluid.layers.dropout(x, dropout_prob=self.dropout)
        return x


class TransformerModel(fluid.dygraph.Layer):
    ATTRIBUTES = ['encoder', 'with_output', 'eos_index', 'pad_index', 'n_langs', 'n_words', 'dim', 'n_layers',
                  'n_heads', 'hidden_dim', 'dropout', 'attention_dropout', 'asm', 'asm_cutoffs', 'asm_div_value']

    def __init__(self, params, dico, is_encoder, with_output):
        """
        Transformer model (encoder or decoder).
        """
        super(TransformerModel, self).__init__()

        # encoder / decoder, output layer
        self.is_encoder = is_encoder
        self.is_decoder = not is_encoder
        self.with_output = with_output
        self.share_inout_emb = params.share_inout_emb

        # dictionary / languages
        self.n_langs = params.n_langs
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.dico = dico
        self.id2lang = params.id2lang
        self.lang2id = params.lang2id
        self.use_lang_emb = getattr(params, 'use_lang_emb', True)
        assert len(self.dico) == self.n_words
        assert len(self.id2lang) == len(self.lang2id) == self.n_langs

        # model parameters
        self.dim = params.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.n_heads = params.n_heads  # 8 by default
        self.n_layers = params.n_layers
        if self.is_decoder:
            self.n_layers = params.dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'
        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.dim)
        if params.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.dim, out=self.position_embeddings.weight)
        if params.n_langs > 1 and self.use_lang_emb:
            self.lang_embeddings = Embedding(self.n_langs, self.dim)
        self.embeddings = Embedding(self.n_words, self.dim, padding_idx=self.pad_index)
        self.layer_norm_emb = LayerNorm(self.dim, epsilon=1e-12)

        # transformer layers: LayerList -> sequential, python list -> modulelist?
        # try LayerList for now
        # self.attentions = nn.ModuleList()
        # self.layer_norm1 = nn.ModuleList()
        # self.ffns = nn.ModuleList()
        # self.layer_norm2 = nn.ModuleList()
        self.attentions = LayerList()
        self.layer_norm1 = LayerList()
        self.ffns = LayerList()
        self.layer_norm2 = LayerList()
        if self.is_decoder:
            # self.layer_norm15 = nn.ModuleList()
            # self.encoder_attn = nn.ModuleList()
            self.layer_norm15 = LayerList()
            self.encoder_attn = LayerList()

        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))
            self.layer_norm1.append(LayerNorm(self.dim))
            if self.is_decoder:
                self.layer_norm15.append(LayerNorm(self.dim))
                self.encoder_attn.append(MultiHeadAttention(self.n_heads, self.dim, dropout=self.attention_dropout))

            self.ffns.append(TransformerFFN(self.dim, self.hidden_dim, self.dim, dropout=self.dropout,
                                            gelu_activation=params.gelu_activation))
            self.layer_norm2.append(LayerNorm(self.dim))

        # output layer
        if self.with_output:
            self.pred_layer = PredLayer(params)
            # print("Debug: pred_layer, ", self.pred_layer.proj.weight.shape)
            # self.pred_layer.proj.weight = to_variable(self.embeddings.weight.numpy().transpose())
            # if params.share_inout_emb:
            #     # self.pred_layer.proj.weight = self.embeddings.weight
            #     a = fluid.global_scope().find_var('pred_layer.proj.weight').get_tensor()
            #     a.set(self.embeddings.weight.numpy().transpose())

    def load_state_dict(self, state_dict, keys):
        """
        Copy pretrained transformer parameters in state_dict to current transformer object.
        Note: match different keys between pytorch and paddle
        :param keys:
        :param state_dict: pytorch state_dict obj
        :return:
        """
        # keys = state_dict.keys()
        for k in keys:
            # print(k, type(state_dict[k]))
            if "torch" not in str(type(state_dict[k])):
                continue
            if ("lin" in k or "pred_layer" in k) and "weight" in k:
                if k == "pred_layer.proj.weight" and self.share_inout_emb:
                    state_dict[k] = state_dict["embeddings.weight"].data.cpu().numpy().transpose()
                else:
                    state_dict[k] = state_dict[k].data.cpu().numpy().transpose()
            else:
                state_dict[k] = state_dict[k].data.cpu().numpy()
        self.set_dict(state_dict)

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None,
            positions=None, langs=None, cache=None):
        """
        Inputs:
            `x` LongTensor(slen, bs), containing word indices
            `lengths` LongTensor(bs), containing the length of each sentence
            `causal` Boolean, if True, the attention is only done over previous hidden states
            `positions` LongTensor(slen, bs), containing word positions
            `langs` LongTensor(slen, bs), containing language IDs
        """
        # lengths = (x != self.pad_index).float().sum(dim=1)
        # mask = x != self.pad_index

        # check inputs
        slen, bs = x.shape
        assert lengths.shape[0] == bs
        assert reduce_max(lengths).numpy()[0] <= slen
        x = transpose(x=x, perm=[1, 0])  # batch size as dimension 0
        assert (src_enc is None) == (src_len is None)
        if src_enc is not None:
            assert self.is_decoder
            assert src_enc.shape[0] == bs

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        mask.stop_gradient = True
        attn_mask.stop_gradient = True
        if self.is_decoder and src_enc is not None:
            # fluid.layers.range(0, slen, step=1, dtype='int64')
            # src_mask = torch.arange(src_len.max(), dtype=torch.long, device=lengths.device) < src_len[:, None]
            slen_range = fluid.layers.range(0, reduce_max(src_len), step=1, dtype='int64')
            src_mask = expand(unsqueeze(slen_range, axes=0),
                              expand_times=[src_len.shape[0], 1]) < unsqueeze(src_len, axes=1)
            # positions
        if positions is None:
            # positions = x.new(slen).long()
            # positions = torch.arange(slen, out=positions).unsqueeze(0)
            positions = unsqueeze(fluid.layers.range(0, slen, step=1, dtype='int64'), axes=0)
        else:
            assert positions.shape == [slen, bs]
            positions = transpose(positions, perm=[1, 0])
        positions.stop_gradient = True
        # langs
        if langs is not None:
            assert langs.shape == [slen, bs]
            langs = transpose(langs, perm=[1, 0])

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            # what stupid design: slicing doesn't support bool, uint8 types!!!
            mask = cast(cast(mask, "int64")[:, -_slen:], "bool")
            attn_mask = cast(cast(attn_mask, "int64")[:, -_slen:], "bool")

        # embeddings: subword embeddings + position embeddings
        tensor = self.embeddings(x)
        tensor = tensor + expand_as(self.position_embeddings(positions), target_tensor=tensor)
        if langs is not None and self.use_lang_emb:
            tensor = tensor + self.lang_embeddings(langs)
        tensor = self.layer_norm_emb(tensor)
        tensor = dropout(tensor, dropout_prob=self.dropout)
        # tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        tensor *= unsqueeze(cast(mask, "float32"), axes=-1)

        # transformer layers
        for i in range(self.n_layers):

            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = fluid.layers.dropout(attn, dropout_prob=self.dropout)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            if self.is_decoder and src_enc is not None:
                attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
                attn = fluid.layers.dropout(attn, dropout_prob=self.dropout)
                tensor = tensor + attn
                tensor = self.layer_norm15[i](tensor)

            # FFN

            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= unsqueeze(cast(mask, "float32"), axes=-1)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.shape[1]

        # move back sequence length to dimension 0
        # [slen, bs, hidden_dim]
        # tensor = tensor.transpose(0, 1)
        tensor = transpose(tensor, perm=[1, 0, 2])
        return tensor

    def predict(self, tensor, pred_mask, y, get_scores):
        """
        Given the last hidden state, compute word scores and/or the loss.
            `pred_mask` is a ByteTensor of shape (slen, bs), filled with 1 when
                we need to predict a word
            `y` is a LongTensor of shape (pred_mask.sum(),)
            `get_scores` is a boolean specifying whether we need to return scores
        """
        # masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.dim)
        mask = cast(unsqueeze(cast(pred_mask, "float32"), axes=-1), "bool")
        masked_tensor = masked_select(tensor, mask)
        masked_tensor = reshape(masked_tensor, shape=[-1, self.dim])
        scores, loss = self.pred_layer(masked_tensor, y, get_scores)
        return scores, loss

    def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        `langs`:
            - must be None if the model only supports one language
            - lang_id if only one language is involved (LM)
            - (lang_id1, lang_id2) if two languages are involved (MT)
        """

        # input batch
        bs = src_len.shape[0]
        assert src_enc.shape[0] == bs

        # generated = src_len.new(max_len, bs)  # upcoming output
        # generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
        # Temporal solution: use numpy array all the time unless passing generated to decoder.forward func.
        generated = fill_constant(shape=[max_len, bs], dtype="int64", value=self.pad_index)
        # generated = np.full(shape=(max_len, bs), dtype=np.int64, fill_value=self.pad_index)
        # generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere
        # generated[0] = self.eos_index
        generated = scatter(generated, to_variable(np.array([0], dtype=np.int64)),
                            to_variable(np.array([self.eos_index]*bs, dtype=np.int64)[None, :]))
        generated.stop_gradient = True
        # positions
        # positions = src_len.new(max_len).long()
        # positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
        positions = expand(unsqueeze(fluid.layers.range(0, max_len, step=1, dtype="int64"), axes=1),
                           expand_times=[max_len, bs])
        positions.stop_gradient = True
        # language IDs
        # langs = src_len.new(max_len).long().fill_(tgt_lang_id)
        langs = fill_constant(shape=[max_len], dtype="int64", value=tgt_lang_id)
        langs = expand(unsqueeze(langs, axes=1), expand_times=[max_len, bs])
        langs.stop_gradient = True

        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        # gen_len = src_len.clone().fill_(1)
        gen_len = fill_constant(shape=[bs], dtype="int64", value=1)
        unfinished_sents = fill_constant(shape=[bs], dtype="int64", value=1)

        # cache compute states
        cache = {'slen': 0}
        # print("src_len=", src_len, "\nmax_len=", max_len)
        while cur_len < max_len:
            # compute word scores
            tensor = self.forward(
                'fwd',
                x=generated[:cur_len],
                lengths=gen_len,
                positions=positions[:cur_len],
                langs=langs[:cur_len],
                causal=True,
                src_enc=src_enc,
                src_len=src_len,
                cache=cache
            )
            assert tensor.shape == [1, bs, self.dim], \
                (cur_len, max_len, src_enc.shape, tensor.shape,
                 (1, bs, self.dim))
            tensor = cast(tensor[-1, :, :], dtype=src_enc.dtype)  # (bs, dim)
            scores = self.pred_layer.get_scores(tensor)  # (bs, n_words)

            # select next words: sample or greedy
            # if sample_temperature is None:
            next_words = squeeze(topk(scores, 1)[1], axes=[1])
            # print("next words are:", next_words.numpy())
            # else:
            #    next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
            assert next_words.shape[0] == bs

            # update generations / lengths / finished sentences / current length
            tmp = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            # generated[cur_len] = tmp.numpy()
            generated = scatter(generated, to_variable(np.array([cur_len], dtype=np.int64)), unsqueeze(tmp, 0))
            generated.stop_gradient = True
            gen_len += unfinished_sents
            unfinished_sents *= cast(next_words != self.eos_index, dtype="int64")
            cur_len = cur_len + 1
            # print(cur_len, unfinished_sents)
            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if reduce_max(unfinished_sents).numpy()[0] == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            updates = generated[max_len-1] + unfinished_sents*(generated[0]-generated[max_len-1])
            # print(updates)
            # print(generated.numpy())
            generated = scatter(generated, to_variable(np.array([max_len-1], dtype=np.int64)), unsqueeze(updates, 0))
            # generated[-1][] = self.eos_index

        # sanity check
        # print(generated)
        generated.stop_gradient = True
        # assert (generated.numpy() == self.eos_index).sum() == 2 * bs
        # print(generated.numpy())
        return generated[:cur_len], gen_len

    # def generate(self, src_enc, src_len, tgt_lang_id, max_len=200, sample_temperature=None):
    #     """
    #     Decode a sentence given initial start.
    #     `x`:
    #         - LongTensor(bs, slen)
    #             <EOS> W1 W2 W3 <EOS> <PAD>
    #             <EOS> W1 W2 W3   W4  <EOS>
    #     `lengths`:
    #         - LongTensor(bs) [5, 6]
    #     `positions`:
    #         - False, for regular "arange" positions (LM)
    #         - True, to reset positions from the new generation (MT)
    #     `langs`:
    #         - must be None if the model only supports one language
    #         - lang_id if only one language is involved (LM)
    #         - (lang_id1, lang_id2) if two languages are involved (MT)
    #     """
    #
    #     # input batch
    #     bs = src_len.shape[0]
    #     assert src_enc.shape[0] == bs
    #
    #     # generated = src_len.new(max_len, bs)  # upcoming output
    #     # generated.fill_(self.pad_index)  # fill upcoming ouput with <PAD>
    #     # Temporal solution: use numpy array all the time unless passing generated to decoder.forward func.
    #     # generated = fill_constant(shape=[max_len, bs], dtype="int64", value=self.pad_index)
    #     generated = np.full(shape=(max_len, bs), dtype=np.int64, fill_value=self.pad_index)
    #     # generated[0].fill_(self.eos_index)  # we use <EOS> for <BOS> everywhere
    #     generated[0] = self.eos_index
    #
    #     # positions
    #     # positions = src_len.new(max_len).long()
    #     # positions = torch.arange(max_len, out=positions).unsqueeze(1).expand(max_len, bs)
    #     positions = expand(unsqueeze(fluid.layers.range(0, max_len, step=1, dtype="int64"), axes=1),
    #                        expand_times=[max_len, bs])
    #
    #     # language IDs
    #     # langs = src_len.new(max_len).long().fill_(tgt_lang_id)
    #     langs = fill_constant(shape=[max_len], dtype="int64", value=tgt_lang_id)
    #     langs = expand(unsqueeze(langs, axes=1), expand_times=[max_len, bs])
    #
    #     # current position / max lengths / length of generated sentences / unfinished sentences
    #     cur_len = 1
    #     # gen_len = src_len.clone().fill_(1)
    #     gen_len = fill_constant(shape=[bs], dtype="int64", value=1)
    #     unfinished_sents = fill_constant(shape=[bs], dtype="int64", value=1)
    #
    #     # cache compute states
    #     cache = {'slen': 0}
    #     # print("src_len=", src_len, "\nmax_len=", max_len)
    #     while cur_len < max_len:
    #         # compute word scores
    #         tensor = self.forward(
    #             'fwd',
    #             x=to_variable(generated[:cur_len]),
    #             lengths=gen_len,
    #             positions=positions[:cur_len],
    #             langs=langs[:cur_len],
    #             causal=True,
    #             src_enc=src_enc,
    #             src_len=src_len,
    #             cache=cache
    #         )
    #         assert tensor.shape == [1, bs, self.dim], \
    #             (cur_len, max_len, src_enc.shape, tensor.shape,
    #              (1, bs, self.dim))
    #         tensor = cast(tensor[-1, :, :], dtype=src_enc.dtype)  # (bs, dim)
    #         scores = self.pred_layer.get_scores(tensor)  # (bs, n_words)
    #
    #         # select next words: sample or greedy
    #         # if sample_temperature is None:
    #         next_words = squeeze(topk(scores, 1)[1], axes=[1])
    #         # print("next words are:", next_words.numpy())
    #         # else:
    #         #    next_words = torch.multinomial(F.softmax(scores / sample_temperature, dim=1), 1).squeeze(1)
    #         assert next_words.shape[0] == bs
    #
    #         # update generations / lengths / finished sentences / current length
    #         tmp = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
    #         generated[cur_len] = tmp.numpy()
    #         gen_len += unfinished_sents
    #         unfinished_sents *= cast(next_words != self.eos_index, dtype="int64")
    #         cur_len = cur_len + 1
    #         # print(cur_len, unfinished_sents)
    #         # stop when there is a </s> in each sentence, or if we exceed the maximul length
    #         if reduce_max(unfinished_sents).numpy()[0] == 0:
    #             break
    #
    #     # add <EOS> to unfinished sentences
    #     if cur_len == max_len:
    #         # generated[-1].masked_fill_(unfinished_sents.byte(), self.eos_index)
    #         # partial assignment in paddle dygraph is not working!!! Use numpy to bridge
    #         # generated[-1][cast(unfinished_sents, dtype='unit8')] = self.eos_index
    #         generated[-1][unfinished_sents.numpy().astype(np.bool)] = self.eos_index
    #         # generated[-1] = np.ma.array(generated[-1], mask=unfinished_sents.numpy().astype(np.bool),
    #         #                            fill_value=self.eos_index)
    #
    #     # sanity check
    #     # print(generated)
    #     assert (generated == self.eos_index).sum() == 2 * bs
    #
    #     return to_variable(generated[:cur_len]), gen_len
