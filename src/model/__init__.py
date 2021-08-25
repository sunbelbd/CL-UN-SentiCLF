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

import os
from logging import getLogger

import torch

from src.data.dictionary import Dictionary
from .classifier import Classifier
from .discriminator import Discriminator
from .memory import HashingMemory
from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel, TRANSFORMER_LAYER_PARAMS

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        # reload encoder only
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            # reload encoder, decoder and/or discriminator
            assert len(s) == 2 or len(s) == 3
            assert all([x == '' or os.path.isfile(x) for x in s])


def print_shape(torch_dict, paddle_dict):
    """
    Compare state dictionary from pytorch and paddle.
    :param torch_dict:
    :param paddle_dict:
    :return:
    """
    for k, v in torch_dict.items():
        if k in paddle_dict:
            print("key:", k, ", torch shape:", v.shape, ", paddle shape:", paddle_dict[k].shape)


def build_model(params, dico):
    """
    Build model.
    """
    # build with provided dictionary file: dico and params.
    encoder = TransformerModel(params, dico, is_encoder=True,
                               with_output=False)
    # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
    decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)
    discriminator = Discriminator(params)
    classifier = Classifier(params)

    # reload a pretrained model: if pretrained model's dico is inconsistent with the given dico,
    # use matched embeddings to initialize encoder/decoder.
    if params.reload_model != '':
        pretrain_paths = params.reload_model.split(',')
        enc_path, dec_path = pretrain_paths[:2]
        assert not (enc_path == '' and dec_path == '')

        # reuse partial models and dictionary. More advanced initialization when computing power is low.
        # reload encoder: align encoder's embedding vs pretrained emb with dico and enc_dico
        if enc_path != '':
            logger.info("Reloading encoder from %s ..." % enc_path)
            enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
            enc_reload_dico = Dictionary(enc_reload['dico_id2word'], enc_reload['dico_word2id'],
                                         enc_reload['dico_counts'])
            logger.info("Data dico size %d vs pretrained dico size %d" % (len(dico),
                                                                          len(enc_reload_dico)))
            # # load discriminator weights if specified by parameters
            # # reload discriminator
            # if params.reload_dis and params.lambda_dis > 0 and 'discriminator' in enc_reload:
            #     logger.info("Reloading discriminator from %s ..." % enc_path)
            #     disc_reload = enc_reload['discriminator']
            #     if all([k.startswith('module.') for k in disc_reload.keys()]):
            #         disc_reload = {k[len('module.'):]: v for k, v in disc_reload.items()}
            #     disc_state_dict = discriminator.state_dict()
            #     keys = disc_state_dict.keys()
            #     print("paddle discriminator keys", keys)
            #     print("torch discriminator keys", disc_reload.keys())
            #     print_shape(disc_state_dict, disc_reload)
            #     disc_state_dict.update(disc_reload)
            #     discriminator.load_state_dict(disc_state_dict, keys)
            # reload encoder weights
            enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
            # print("pytorch encoder state_dict:", enc_reload.keys())
            encoder_model_dict = encoder.state_dict()
            keys = encoder_model_dict.keys()
            # print("paddle encoder state_dict:", encoder_model_dict.keys())
            # print_shape(enc_reload, encoder_model_dict)
            print("###########################################")
            if all([k.startswith('module.') for k in enc_reload.keys()]):
                enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}

            # filter out unnecessary keys and embedding layer, which will be merged later
            pretrained_dict = {k: v for k, v in enc_reload.items() if
                               k in encoder_model_dict and k != "embeddings.weight" and "pred_layer" not in k}
            encoder_model_dict.update(pretrained_dict)
            # handle embedding layer initialization and output pred layer if defined
            reloaded_word_ids = [enc_reload_dico.word2id[dico.id2word[i]] for i in range(len(dico))]
            encoder_model_dict["embeddings.weight"] = enc_reload["embeddings.weight"][reloaded_word_ids]
            if "pred_layer.proj.weight" in encoder_model_dict and "pred_layer.proj.weight" in enc_reload:
                # process weight and bias
                encoder_model_dict["pred_layer.proj.weight"] = enc_reload["pred_layer.proj.weight"][
                    reloaded_word_ids]
                encoder_model_dict["pred_layer.proj.bias"] = enc_reload["pred_layer.proj.bias"][
                    reloaded_word_ids]

            # encoder is an object in paddle, implement a paddle version load_state_dict()
            # to copy pretrained model encoder_model_dict to encoder
            encoder.load_state_dict(encoder_model_dict, keys)
            # encoder.set_dict(encoder_model_dict)
            del enc_reload, encoder_model_dict

        # reload decoder
        if dec_path != '':
            logger.info("Reloading decoder from %s ..." % dec_path)
            dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
            dec_reload_dico = Dictionary(dec_reload['dico_id2word'], dec_reload['dico_word2id'],
                                         dec_reload['dico_counts'])
            dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
            # print("pytorch decoder state_dict:", dec_reload.keys())
            dec_model_dict = decoder.state_dict()
            keys = dec_model_dict.keys()
            # print("paddle decoder state_dict:", dec_model_dict.keys())
            if all([k.startswith('module.') for k in dec_reload.keys()]):
                dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}

            for i in range(min(params.dec_layers, params.n_layers)):
                for name in DECODER_ONLY_PARAMS:
                    if name % i not in dec_reload:
                        logger.warning("Parameter %s not found." % (name % i))
                        dec_reload[name % i] = decoder.state_dict()[name % i]
            # filter out unnecessary keys and embedding layer, which will be merged later
            pretrained_dict = {k: v for k, v in dec_reload.items() if
                               k in dec_model_dict and k != "embeddings.weight" and "pred_layer" not in k}
            # print_shape(dec_reload, dec_model_dict)
            dec_model_dict.update(pretrained_dict)
            # handle embedding layer initialization and output pred layer if defined
            reloaded_word_ids = [dec_reload_dico.word2id[dico.id2word[i]] for i in range(len(dico))]
            dec_model_dict["embeddings.weight"] = dec_reload["embeddings.weight"][reloaded_word_ids]
            if "pred_layer.proj.weight" in dec_model_dict and "pred_layer.proj.weight" in dec_reload:
                # process weight and bias
                dec_model_dict["pred_layer.proj.weight"] = dec_reload["pred_layer.proj.weight"][
                    reloaded_word_ids]
                dec_model_dict["pred_layer.proj.bias"] = dec_reload["pred_layer.proj.bias"][
                    reloaded_word_ids]

            # decoder is an object in paddle, implement a paddle version load_state_dict()
            # to copy pretrained model dec_model_dict to encoder
            #
            decoder.load_state_dict(dec_model_dict, keys)
            del dec_reload, dec_model_dict

        # logger.info("Encoder: {}".format(encoder))
        # logger.info("Decoder: {}".format(decoder))
        # logger.info("Discriminator: {}".format(discriminator))
        # logger.info("Classifier: {}".format(classifier))
        # logger.info(
        #     "Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        # logger.info(
        #     "Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))
        # logger.info("Number of parameters (discriminator): %i" % sum(
        #     [p.numel() for p in discriminator.parameters() if p.requires_grad]))
        # logger.info("Number of parameters (classifier): %i" % sum(
        #     [p.numel() for p in classifier.parameters() if p.requires_grad]))

    return encoder, decoder, discriminator, classifier
