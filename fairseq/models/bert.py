# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from transformers.models.bert.modeling_bert import BertModel, BertEncoder
from transformers.models.bert.configuration_bert import BertConfig
from pinyin_data.pinyin_tool import PinyinTool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import (
    FairseqDecoder, FairseqEncoder, FairseqLanguageModel,
    register_model, register_model_architecture,
    FairseqIncrementalDecoder, FairseqModel
)

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax, CharacterTokenEmbedder, MultiheadAttention,
    SimpleSinusoidalPositionalEmbedding, LearnedPositionalEmbedding
)
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class pinyinEmbedding:
    def __init__(self, padding_idx, config):
        py_dict_path = './pinyin_data/zi_py.txt'
        py_vocab_path = './pinyin_data/py_vocab.txt'
        self.pytool = PinyinTool(py_dict_path=py_dict_path, py_vocab_path=py_vocab_path, py_or_sk='py')
        self.py_label_list = {v: k for k, v in self.pytool.vocab.items()}
        f=open('./vocab_bert.txt','r',encoding='utf-8')
        lines=f.readlines()
        self.vocab={}
        self.index={}
        self.tokenid_pyid = {}
        for line in lines:
            line=line.split(' ')
            self.vocab[line[0]]=int(line[1])
            self.index[int(line[1])]=line[0]
        for key in self.vocab:
            self.tokenid_pyid[key] = self.pytool.get_pinyin_id(key)
        self.padding_idx = padding_idx
        embedding_size = 128
        self.out_dim = config.hidden_size
        self.emb = nn.Embedding(419, embedding_size, self.padding_idx)
        self.conv = nn.Conv1d(in_channels= embedding_size, out_channels=self.out_dim, kernel_size=2, stride=1, padding=0)


    def encode_batch(self, inputer):
        output=[]
        for bsz in inputer:
            for idx in bsz:
                output.append(self.tokenid_pyid[self.index[idx.item()]])
        # print(bsz, ids)i

        output = np.array(output)
        output = output.reshape(inputer.size())

        return torch.tensor(output)

    def make_embedding(self, inputer):
        encode = self.encode_batch(inputer)#.to(inputer.device)
        embed_py = self.emb(encode)
        return torch.FloatTensor(embed_py ).cuda()# bsz, seq_len, outchannels

@register_model('bert_1encoder')
class Transformer_nonautoregressive(FairseqModel):
    def __init__(self, encoderx,   decoder):
        super().__init__(encoderx,  decoder)
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder_layers_x', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder_layers_z', type=int, metavar='N',
                            help='num encoder z layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-enc-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--no-dec-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--embedding-only', default=False, action='store_true',
                            help='if set, replaces the encoder with just token embeddings (could be complex e.g. bilm')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--bilm-model-dropout', default=0.1, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the model dropout for bilm')
        parser.add_argument('--bilm-attention-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the attention dropout for bilm')
        parser.add_argument('--bilm-relu-dropout', default=0.0, type=float, metavar='D',
                            help='if using a pretrained bilm encoder, what is the relu dropout for bilm')
        parser.add_argument('--bilm-mask-last-state', action='store_true',
                            help='if set, masks last state in bilm as is done during training')
        parser.add_argument('--bilm-add-bos', action='store_true',
                            help='if set, adds bos to input')
        parser.add_argument('--decoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in decoder')
        parser.add_argument('--encoder-embed-scale', type=float,
                            help='scaling factor for embeddings used in encoder')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        #for ds in task.datasets.values():
        #    ds.target_is_source = True

        # make sure all arguments are present in older models

        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        bert_config = BertConfig.from_pretrained('/home/sunrui/Mask-Predict-main/fairseq/models/bert_config2.json')
        bert = BertModel.from_pretrained('/home/sunrui/bert-base-chinese/pytorch_model.bin', config = bert_config)

        pho_embedding = pinyinEmbedding(tgt_dict.pad() , bert_config)

        #transformer_config = BertConfig.from_pretrained('/home/sunrui/Mask-Predict-main/fairseq/models/transformer_config.json')
        #transformer_layer_1 = BertEncoder(transformer_config)
        encoder_embed_bert = bert
        encoder_embed_pinyin = pho_embedding
        decoder_embed_bert = bert
        decoder_embed_pinyin = pho_embedding
        args.share_decoder_input_output_embed = True


        encoder_x = TransformerEncoder_x(args, src_dict, encoder_embed_bert, encoder_embed_pinyin, bert_config)
        decoder = SelfTransformerDecoder(args, tgt_dict, decoder_embed_bert, decoder_embed_pinyin, bert_config)
        return Transformer_nonautoregressive(encoder_x, decoder)



class SelfTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, embed_bert, embed_pinyin, bert_config,  embed_scale=None, no_encoder_attn=False, left_pad=False,
                 final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.padding_idx = dictionary.pad()
        self.max_source_positions = args.max_source_positions
        self.eos_idx = dictionary.eos()
      
        self.bert_model = embed_bert
        self.embed_pinyin = embed_pinyin

        self.embed_scale = math.sqrt(args.encoder_embed_dim) if embed_scale is None else embed_scale
        self.map_fc = nn.Linear(bert_config.hidden_size+128  , bert_config.vocab_size)   #p
        self.LayerNorm_py = nn.LayerNorm(128)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)


    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """

        #print("encdoer mask", encoder_out['encoder_padding_mask'])
        #print("encoder_out", encoder_out['encoder_out'])
        decoder_padding_mask = prev_output_tokens.ne(self.padding_idx)
        #prev_output_tokens = prev_output_tokens.new(prev_output_tokens.size()).fill_(103)
        #print("prev_output_tokens",prev_output_tokens)
        #print("decoder mask", decoder_padding_mask)
        context_bert_output = self.bert_model(input_ids=prev_output_tokens,
                                              attention_mask=decoder_padding_mask,
                                              encoder_hidden_states = encoder_out['encoder_out'],
                                              encoder_attention_mask = encoder_out['encoder_padding_mask'])[0]
        pho_embeddings = self.embed_pinyin.make_embedding(prev_output_tokens)
        pho_embeddings = self.dropout(self.LayerNorm_py(pho_embeddings))
        input_features = torch.cat((context_bert_output, pho_embeddings), 2)    #batch, seq_len, hiddensize*2
        x = self.map_fc(input_features)
        bsz ,seq, vol = x.size()
        y = x.view(-1, x.size(-1))
        y = F.softmax(x, -1)
        index = torch.max(y, -1)[1].view(bsz, seq)
        return x,x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        #if self.embed_positions is None:
        return self.max_source_positions
        #return min(self.max_target_positions, self.embed_positions.max_positions())
        #return self.max_target_positions
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        pass



class TransformerEncoder_x(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(self, args, dictionary, embed_bert, embed_pinyin,  bert_config, embed_scale=None, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.bert_model = embed_bert
        self.padding_idx = dictionary.pad()
        self.max_source_positions = args.max_source_positions
        self.eos_idx = dictionary.eos()
        self.dic = dictionary
        self.embed_bert = embed_bert
        self.embed_pinyin = embed_pinyin

        self.embed_scale = math.sqrt(args.encoder_embed_dim) if embed_scale is None else embed_scale
        self.map_fc = nn.Linear(bert_config.hidden_size+128 , bert_config.hidden_size)   #p
        self.LayerNorm_py = nn.LayerNorm(128)
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)


    def forward(self, srcx_tokens, srcx_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        encoder_padding_mask = srcx_tokens.ne(self.padding_idx)
        context_bert_output = self.bert_model(input_ids=srcx_tokens,
                                              attention_mask=encoder_padding_mask,
                                              )[0]
        pho_embeddings = self.embed_pinyin.make_embedding(srcx_tokens)
        
        pho_embeddings = self.dropout(self.LayerNorm_py(pho_embeddings))
        input_features = torch.cat((context_bert_output, pho_embeddings), 2)    #batch, seq_len, hiddensize*2
        #last_hidden_state = self.transformer_layer(input_features, encoder_padding_mask)[0]
        last_hidden_state = self.map_fc(input_features)
        return {
            'encoder_out': last_hidden_state,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
      #  if encoder_out['predicted_lengths'] is not None:
      #      encoder_out['predicted_lengths'] = \
      #          encoder_out['predicted_lengths'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        #if self.embed_positions is None:
        return self.max_source_positions
       # return self.max_source_positions
        #return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if utils.item(state_dict.get('encoder.version', torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['encoder.version'] = torch.Tensor([1])
        return state_dict



@register_model_architecture('bert_1encoder', 'bert_1encoder')   #args是一个对象，getattr用来获取对象中的属性值
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', args.encoder_embed_dim * 4)
    args.encoder_layers_x = getattr(args, 'encoder_layers_x', 4)
    args.encoder_layers_z = getattr(args, 'encoder_layers_z', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', args.encoder_embed_dim // 64)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', args.encoder_attention_heads)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_enc_token_positional_embeddings = getattr(args, 'no_enc_token_positional_embeddings', False)
    args.no_dec_token_positional_embeddings = getattr(args, 'no_dec_token_positional_embeddings', False)
    args.embedding_only = getattr(args, 'embedding_only', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.decoder_embed_scale = getattr(args, 'decoder_embed_scale', None)
    args.encoder_embed_scale = getattr(args, 'encoder_embed_scale', None)

    args.bilm_mask_last_state = getattr(args, 'bilm_mask_last_state', False)
    args.bilm_add_bos = getattr(args, 'bilm_add_bos', False)



@register_model_architecture('bert_1encoder', 'bert_1encoder_big')
def bi_transformer_lm_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_architecture(args)


