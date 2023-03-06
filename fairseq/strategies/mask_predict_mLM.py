# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_long, convert_tokens
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertTokenizer,  BasicTokenizer


@register_strategy('mask_predict_mLM')
class MaskPredict_mLM(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations

    def prob_normalized(self, probs):
        new_probs = probs.new(probs.size())
        for i in range(len(probs)):
            minv = probs[i].min()
            maxv = probs[i].max()
            for k in range(len(probs[i])):
         #       print(probs[i][k],minv, maxv)
                new_probs[i][k]= (probs[i][k]-minv)//maxv-minv
        return new_probs


    def generate(self, model, encoder_x_out, encoder_z_out,  tgt_tokens, tgt_dict):
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)

        #iterations = seq_len if self.iterations is None else self.iteration
        new_tgt_tokens, new_token_probs = self.generate_non_autoregressive(model, encoder_x_out,encoder_z_out,  tgt_tokens)
        #print("nat",new_token_probs.size())
        assign_single_value_byte(new_tgt_tokens, pad_mask , tgt_dict.pad())
        #print("new_tgt_tokens", new_tgt_tokens)
        ta_tgt_probs = self.generate_teacher(tgt_dict, new_tgt_tokens, tgt_tokens)
        #print("teacher",ta_tgt_probs.size())
        ta_tgt_probs = self.prob_normalized(ta_tgt_probs)
        new_token_probs = self.prob_normalized(new_token_probs)

        gamma = 0.5
        token_probs = gamma*ta_tgt_probs + (1-gamma)*new_token_probs
        assign_single_value_byte(token_probs, pad_mask, 1.0)

       # print("token_probs", token_probs)
        num_mask = (seq_lens.float() /3 ).long()
        mask_ind = self.select_worst(token_probs, num_mask) #复合概率最低的
        assign_single_value_long(new_tgt_tokens, mask_ind, tgt_dict.mask())
#        assign_single_value_long(new_tgt_tokens, pad_mask, tgt_dict.pad())

        mask_tokens = new_tgt_tokens.eq(tgt_dict.mask())
        mask_num = mask_tokens.sum(dim=1)
        max_mask_num = max(mask_num)
        fix_tokens = new_tgt_tokens.ne(tgt_dict.mask())

        tgt_tokens , token_probs = new_tgt_tokens, token_probs

        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad()) #pad_mask.nonzero=pad()
        assign_single_value_byte(token_probs, pad_mask, 1.0)    #pad， 概率设置为1
        assign_single_value_byte(token_probs, fix_tokens, 1.0)   #没加mask的 概率设置为1
        #print("Initialization: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        
        for counter in range( max_mask_num):
            #print("Step: ", counter+1)
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))
 
            num_mask = self.sub_mask(token_probs, num_mask)
            decoder_out = model.decoder(tgt_tokens, encoder_x_out,encoder_z_out)
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            assign_single_value_byte(new_tgt_tokens, pad_mask, tgt_dict.pad())
            ta_tgt_probs = self.generate_teacher(tgt_dict, new_tgt_tokens, tgt_tokens)
            new_token_probs = gamma*ta_tgt_probs + (1-gamma)*new_token_probs

            assign_multi_value_long(token_probs, mask_ind, new_token_probs)  #[bsz,seq_len]
            assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
            #assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            
#            num_mask = self.sub_mask(token_probs, num_mask)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
 #           assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            fix_tokens = tgt_tokens.ne(tgt_dict.mask())
            assign_single_value_byte(token_probs, pad_mask, 1.0)
            assign_single_value_byte(token_probs, fix_tokens, 1.0)


            #print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        decoder_out = model.decoder(tgt_tokens, encoder_x_out,encoder_z_out)
        new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out) 
        assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
        assign_multi_value_long(token_probs, mask_ind, new_token_probs)
        lprobs = token_probs.log().sum(-1)
        return tgt_tokens, lprobs
    
    def generate_non_autoregressive(self, model, encoder_x_out,encoder_z_out , tgt_tokens):
        decoder_out = model.decoder(tgt_tokens, encoder_x_out, encoder_z_out)
        tgt_tokens, token_probs, _ = generate_step_with_prob(decoder_out)
        return tgt_tokens, token_probs

    def check_spelling_error(self, tgt_dict,  sentence,  basic_tokenizer, bert_tokenizer, bert_model, tgt_tokens):
        base_tokens = ["[PAD]"]*len(sentence)
        longs = len(sentence)
        mask_index = []
        for i in range(len(tgt_tokens)):
            if tgt_tokens[i]==tgt_dict.mask():
                mask_index.append(i+1)

        if (len(mask_index)==0):
            return torch.tensor( [1]*longs)
        sentence = tgt_dict.string(sentence)
        sentence  = sentence.split(' ')
        #print("sentence", sentence)
        for i in range(len(sentence)):
            base_tokens[i] = sentence[i]
        #print("base_tokens", base_tokens)
        base_tokens = ["[CLS]"] + base_tokens + ["[SEP]"]
        tokens = [x if x in bert_tokenizer.vocab else "[UNK]" for x in base_tokens]
       # print(tokens)
        ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        #print("bert",ids)
        mask_ids = bert_tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        input_tokens , original_tokens = [] ,[]
        for i in mask_index:
            input_tokens.append(ids[:])
            original_tokens.append(ids[:])
            input_tokens[-1][i] = mask_ids
        batch_size = 2048 // len(tokens)
        bert_predictions = None
        #print("inputtokens",len(input_tokens))
        for i in range(0,len(input_tokens), batch_size):
            batch_tokens = input_tokens[i:i+batch_size]
            predictions = bert_model(torch.tensor(batch_tokens).cuda()).cpu()
            bert_predictions = predictions if bert_predictions is None else torch.cat([bert_predictions,predictions],dim=0)

        bert_predictions = bert_predictions.gather(index=torch.tensor(original_tokens).unsqueeze(-1), dim =-1)
        probs = [1]*longs
        for i , mask in enumerate(mask_index):
            probs[mask-1] = bert_predictions[i, mask]
        return torch.tensor(probs)


    def generate_sentence_p(self, tgt_dict, sentence, basic_tokenizer,bert_tokenizer, bert_model,tgt_tokens):
        probs = self.check_spelling_error(tgt_dict, sentence , basic_tokenizer, bert_tokenizer, bert_model, tgt_tokens)
        return probs

    def generate_teacher(self, tgt_dict, nat_input, tgt_tokens):
        output = nat_input.new(nat_input.size())
        bert_model = BertForMaskedLM.from_pretrained("/home/sunrui/bert-base-chinese/").cuda()
        bert_model.eval()

        bert_tokenizer = BertTokenizer.from_pretrained("/home/sunrui/Mask-Predict-main/vocab_bert_ori.txt")
        basic_tokenizer = BasicTokenizer(do_lower_case=False)
        bsz,_=nat_input.size()
        for sen  in range(bsz):
            output[sen]=(self.generate_sentence_p(tgt_dict, nat_input[sen], basic_tokenizer, bert_tokenizer, bert_model,tgt_tokens[sen]))
        return output

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

