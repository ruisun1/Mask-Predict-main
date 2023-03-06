# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import json
from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens


@register_strategy('mask_predict_sc_conf')
class MaskPredictSCConf(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        f = open('/home/sunrui/Mask-Predict-main/fairseq/criterions/Confusion.json')
        dict = json.load(f)
        self.confusion_set = dict
        self.confusion_matrix = [[0 for _ in range(5709)] for _ in range (5709)]
        for i in range(5709):
            self.confusion_matrix[i][i] = 1
        for key in dict:
            for word in dict[key]:
                self.confusion_matrix[int(key)][int(word)] = 1
                self.confusion_matrix[int(word)][int(key)] = 1
        f2 = open('/home/sunrui/Mask-Predict-main/fairseq/criterions/shapeConfusion.json')
        dict = json.load(f2)
        self.confusion_set = dict
        for key in dict:
            for word in dict[key]:
                self.confusion_matrix[int(key)][int(word)] = 1
                self.confusion_matrix[int(word)][int(key)] = 1
        weight = torch.tensor(self.confusion_matrix, dtype = torch.float32)
        self.embedding = torch.nn.Embedding.from_pretrained(weight) 


    def generate(self, model, encoder_x_out, encoder_z_out, tgt_tokens, tgt_dict):      
        print("tgt_tokens",tgt_tokens)
        bsz, seq_len = tgt_tokens.size()
        #print("bsz, seq_len from tgt_tokens", bsz,seq_len)
        #print ("encoderx size, encoderz size, tgt_tokens size",encoder_x_out['encoder_x_out'].size() ,encoder_z_out['encoder_z_out'].size(),tgt_tokens.size())
        #print("tgt_tokens",tgt_tokens)
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)

        iterations = seq_len if self.iterations is None else self.iterations
        
        tgt_tokens, token_probs, probs = self.generate_non_autoregressive(model, encoder_x_out, encoder_z_out, tgt_tokens)
        inputemb = torch.tensor(tgt_tokens.view(-1,1), dtype = torch.long).cpu()
        conf_prob  = self.embedding(inputemb).view(bsz*seq_len, -1).cuda()
        print(conf_prob)
        conf_prob = conf_prob * probs.view(bsz*seq_len, -1)
        conf_prob = conf_prob.view(bsz,seq_len,-1)
        print(conf_prob)
        token_probs, tgt_tokens  = conf_prob.max(dim=-1) 
        #print(tgt_tokens.size(),token_probs.size())
        #print(tgt_tokens[0])
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
        assign_single_value_byte(token_probs, pad_mask, 1.0)
    
        for counter in range(1, iterations):
            num_mask = (seq_lens.float() * (1.0 - (counter / iterations))).long()

            assign_single_value_byte(token_probs, pad_mask, 1.0)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())

           # print("Step: ", counter+1)
           # print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))
            decoder_out = model.decoder(tgt_tokens, encoder_x_out,encoder_z_out)
            
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
           # print(token_probs.size(),mask_ind.size(),new_token_probs.size())
            conf_prob = self.embedding(torch.tensor(new_tgt_tokens.view(-1,1), dtype = torch.long).cpu()).view(bsz*seq_len, -1).cuda()
            
            conf_prob = conf_prob * all_token_probs.view(bsz*seq_len, -1)
            conf_prob = conf_prob.view(bsz,seq_len,-1)
            token_probs, new_tgt_tokens  = conf_prob.max(dim=-1)
            assign_multi_value_long(token_probs, mask_ind, new_token_probs)
            assign_single_value_byte(token_probs, pad_mask, 1.0)

            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
           # print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs  
    def generate_non_autoregressive(self, model, encoder_x_out, encoder_z_out, tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_x_out, encoder_z_out)
        tgt_tokens, token_probs, probs = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs,probs

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
