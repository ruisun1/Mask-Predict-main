# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('mask_predict_sc_bert')
class MaskPredictSCBert(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
    
    def generate(self, model, encoder_x_out, encoder_z_out, tgt_tokens, tgt_dict):      
      #  print("tgt_tokens",tgt_tokens)
        old_tgt = tgt_tokens
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(0)
        seq_lens = seq_len - pad_mask.sum(dim=1)
        #print("tgt_dict.mask()",tgt_dict.mask())
        mask_tokens = tgt_tokens.eq(103)
#        print("mask_tokens",mask_tokens)
        mask_num = mask_tokens.sum(dim=1)
        max_mask_num = max(mask_num)
        fix_tokens = tgt_tokens.ne(103)
        print("initial:",tgt_dict.string(tgt_tokens[0]))
        #iterations = seq_len if self.iterations is None else self.iterations
        new_tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_x_out,encoder_z_out, tgt_tokens)
         
        #print("nar:", tgt_dict.string(tgt_tokens[0]))
        assign_single_value_byte(tgt_tokens, pad_mask, 0) #pad_mask.nonzero=pad()
        assign_single_value_byte(token_probs, pad_mask, 1.0)
        assign_single_value_byte(token_probs, fix_tokens, 1.0)
#        print("Initialization: ", convert_tokens(tgt_dict, tgt_tokens[0]))
#        print("max_mask_num",max_mask_num) 
        for counter in range(0, max_mask_num):

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, mask_num) #[bsz,num_mask]
            assign_multi_value_long(tgt_tokens, mask_tokens,new_tgt_tokens) 
            assign_single_value_long(tgt_tokens, mask_ind, 103)
            assign_multi_value_long(tgt_tokens, fix_tokens, old_tgt)
            assign_single_value_byte(tgt_tokens, pad_mask, 0)
            fix_tokens = tgt_tokens.ne(103)
            
            assign_single_value_byte(token_probs, fix_tokens, 1.0)
            print("Step: ", counter+1)
            print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[1]))
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[2]))
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[3]))
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[4]))
            decoder_out = model.decoder(tgt_tokens, encoder_x_out,encoder_z_out)
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            
           # assign_multi_value_long(token_probs, mask_ind, new_token_probs)  #[bsz,seq_len]
           # assign_single_value_byte(token_probs, pad_mask, 1.0)
           # assign_single_value_byte(token_probs, fix_tokens, 1.0)
            mask_num = self.sub_mask(token_probs, mask_num)

            #assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            #assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            #print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))i
            mask_tokens = mask_ind
        assign_multi_value_long(tgt_tokens,mask_tokens,new_tgt_tokens) 
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs    
    def generate_non_autoregressive(self, model, encoder_x_out, encoder_z_out, tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_x_out, encoder_z_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def select_worst(self, token_probs, num_mask):
        bsz, seq_len = token_probs.size()
        masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
        masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
        return torch.stack(masks, dim=0)
    def sub_mask(self, token_probs, num_mask):
        bsz = token_probs.size(0)
        for i in range(bsz):
            if(num_mask[i]>=0):
                num_mask[i]=num_mask[i]-1
            else:
                num_mask[i]=0
        return num_mask
