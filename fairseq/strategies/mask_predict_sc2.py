# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from fairseq.data import data_utils
from . import DecodingStrategy, register_strategy
from .strategy_utils import generate_step_with_prob, assign_single_value_long, assign_single_value_byte, assign_multi_value_byte,  assign_multi_value_long, convert_tokens
from spell_check import check_batch
from pytorch_pretrained_bert import BertModel, BertForMaskedLM, BertTokenizer,  BasicTokenizer
@register_strategy('mask_predicti_sc2')
class MaskPredictSC2(DecodingStrategy):
    
    def __init__(self, args):
        super().__init__()
        self.iterations = args.decoding_iterations
        self.bert_model = BertForMaskedLM.from_pretrained("/home/sunrui/bert-base-chinese/").cuda()
        self.bert_model.eval()

        self.bert_tokenizer = BertTokenizer.from_pretrained("/home/sunrui/Mask-Predict-main/vocab_bert_ori.txt")
        self.basic_tokenizer = BasicTokenizer(do_lower_case=False)
        f = open('/home/sunrui/Mask-Predict-main/output_codataz_filter/dict.cor.txt','r',encoding = 'utf-8')

        lines = f.readlines()
        self.wordindex={}
        self.indexword={}
        for line in lines:
            line = line[:-1]
            line = line.split(' ')
            self.wordindex[line[0]]=int(line[1])
            self.indexword[int(line[1])]=line[0]
    def fin_min(self, probs):
        minv = 100000
        for i in probs:
            if i!=1 and i <minv :
                minv = i
        return minv
    def fin_max(self, probs):
        maxv = -1
        for i in probs:
            if i!=1 and i>maxv :
                maxv = i
        return maxv

    def prob_normalized(self, probs):
        new_probs = []
        for i in range(len(probs)):
            minv = self.fin_min(probs[i])
            maxv = self.fin_max(probs[i])
            for k in range(len(probs[i])):
                probs[i][k] = torch.tensor(probs[i][k])
                minv = torch.tensor(minv)
                maxv = torch.tensor(maxv)
                new_probs.append( ((probs[i][k]-minv).item())/((maxv-minv).item()+0.0000000001))
        import numpy
        new_probs = torch.FloatTensor((numpy.array(new_probs).reshape(probs.size(0),probs.size(1)))).cuda()
        return (new_probs)
    def string(self, tensor,  bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.
        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            return self.indexword[i.item()]
                #return self.word[i]

        sent = ' '.join(token_string(i) for i in tensor if i != 1)
       # return data_utils.process_bpe_symbol(sent, bpe_symbol) 
        return sent[:-5]
    def generate(self, model, encoder_x_out, encoder_z_out, tgt_tokens, tgt_dict):      
        #print("tgt_tokens",tgt_tokens)
        bsz, seq_len = tgt_tokens.size()
        pad_mask = tgt_tokens.eq(tgt_dict.pad())
        seq_lens = seq_len - pad_mask.sum(dim=1)
        #print("tgt_dict.mask()",tgt_dict.mask())
    
        #print("initial:",tgt_dict.string(tgt_tokens[0]))
        #iterations = seq_len if self.iterations is None else self.iterations
        masker_tokens = tgt_tokens.new(tgt_tokens.size()).fill_(tgt_dict.mask())
        new_tgt_tokens, token_probs = self.generate_non_autoregressive(model, encoder_x_out,encoder_z_out, masker_tokens)
    
        new_tgt_tokens_words = tgt_dict.string(new_tgt_tokens).split('\n')
        new_tgt_tokens_align = new_tgt_tokens.new(new_tgt_tokens.size())
  #      print(new_tgt_tokens_words)
        for i in range(len(new_tgt_tokens_words)):
            if(tgt_dict.encode_line(new_tgt_tokens_words[i]).size() == new_tgt_tokens[i].size()):
                new_tgt_tokens_align[i] = tgt_dict.encode_line(new_tgt_tokens_words[i])
            else:
                new_tgt_tokens_align[i] = new_tgt_tokens[i]
        #print("new",new_tgt_tokens_align)
        #print("tgt",tgt_tokens)
        different = (new_tgt_tokens_align != tgt_tokens)
        #print(different)
        
        assign_single_value_byte(new_tgt_tokens, different, tgt_dict.mask())
        max_mask_num = max(new_tgt_tokens.eq(tgt_dict.mask()).sum(dim=1))
        fix_tokens = new_tgt_tokens.ne(tgt_dict.mask())
        #print("nar:", tgt_dict.string(tgt_tokens[0]))
        tgt_tokens = new_tgt_tokens
        assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad()) #pad_mask.nonzero=pad()
       # assign_single_value_byte(token_probs, pad_mask, 1.0)
       # assign_single_value_byte(token_probs, fix_tokens, 1.0)
#        print("Initialization: ", convert_tokens(tgt_dict, tgt_tokens[0]))
#        print("max_mask_num",max_mask_num) 
        num_mask = different.sum(dim=-1)
        mask_ind  = different
        #print("different",mask_ind)
        #print("initializatioon:", convert_tokens(tgt_dict, tgt_tokens[7]))
        for counter in range( max_mask_num):
            #print("Step: ", counter+1)
            #print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[0]))

            #num_mask = self.sub_mask(token_probs, num_mask)
            decoder_out = model.decoder(tgt_tokens, encoder_x_out,encoder_z_out)
            new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
            assign_single_value_byte(new_tgt_tokens, pad_mask, tgt_dict.pad())
            ta_tgt_probs = self.generate_teacher(tgt_dict, new_tgt_tokens, tgt_tokens)
            ta_tgt_probs = self.prob_normalized(ta_tgt_probs)
            new_token_probs = self.prob_normalized(new_token_probs)
            #print("normalized ta",ta_tgt_probs)
            #print("noormalizer nat",new_token_probs)
            gamma = 0.5
            new_token_probs = gamma*ta_tgt_probs + (1-gamma)*new_token_probs
            
            #print("cons", new_token_probs)

            fix_tokens = tgt_tokens.ne(tgt_dict.mask())
        
            if counter == 0:
                assign_multi_value_byte(token_probs, mask_ind, new_token_probs) 
                #[bsz,seq_len]
                assign_multi_value_byte(tgt_tokens, mask_ind, new_tgt_tokens)
            elif (counter == max_mask_num -1) or( counter == max_mask_num -2) :
                assign_multi_value_long(token_probs, mask_ind, new_token_probs)
                suggest_tokens = check_batch( tgt_dict.string(new_tgt_tokens).split('\n') , tgt_dict, self.basic_tokenizer, self.bert_model, self.bert_tokenizer)
                assign_multi_value_long(tgt_tokens, mask_ind , suggest_tokens)

            else:
                assign_multi_value_long(token_probs, mask_ind, new_token_probs)
                assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
    
         #   print("filled", convert_tokens(tgt_dict, tgt_tokens[7]))
            assign_single_value_byte(token_probs, fix_tokens, 100.0)
            assign_single_value_byte(token_probs, pad_mask, 100.0)
            #assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
            num_mask = self.sub_mask(token_probs, num_mask)
#            num_mask = self.sub_mask(token_probs, num_mask)
            mask_ind = self.select_worst(token_probs, num_mask)
            assign_single_value_long(tgt_tokens, mask_ind, tgt_dict.mask())
 #           assign_single_value_byte(tgt_tokens, pad_mask, tgt_dict.pad())
          #  print("Step: ", counter+1)
          #  print("Masking: ", convert_tokens(tgt_dict, tgt_tokens[7]))
           # assign_single_value_byte(token_probs, pad_mask, 1.0)
           # assign_single_value_byte(token_probs, fix_tokens, 1.0)
            

            #print("Prediction: ", convert_tokens(tgt_dict, tgt_tokens[0]))
        decoder_out = model.decoder(tgt_tokens, encoder_x_out,encoder_z_out)
        new_tgt_tokens, new_token_probs, all_token_probs = generate_step_with_prob(decoder_out)
        assign_multi_value_long(tgt_tokens, mask_ind, new_tgt_tokens)
        assign_multi_value_long(token_probs, mask_ind, new_token_probs)
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
                num_mask[i]=num_mask[i]-2
            else:
                num_mask[i]=0
        return num_mask
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
        tokens = [x if x in self.bert_tokenizer.vocab else "[UNK]" for x in base_tokens]
       # print(tokens)
        ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        #print("bert",ids)
        mask_ids = self.bert_tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        input_tokens , original_tokens = [] ,[]
        for i in mask_index:
            input_tokens.append(ids[:])
            original_tokens.append(ids[:])
            input_tokens[-1][i] = mask_ids
        batch_size = 20480 // len(tokens)
        bert_predictions = None
        #print("inputtokens",len(input_tokens))
        for i in range(0,len(input_tokens), batch_size):
            batch_tokens = input_tokens[i:i+batch_size]
            predictions = self.bert_model(torch.tensor(batch_tokens).cuda()).cpu()
            bert_predictions = predictions if bert_predictions is None else torch.cat([bert_predictions,predictions],dim=0)

        bert_predictions = bert_predictions.gather(index=torch.tensor(original_tokens).unsqueeze(-1), dim =-1)
        probs = [1]*longs
        for i , mask in enumerate(mask_index):
           # print("prob",bert_predictions[i,mask])
            probs[mask-1] = bert_predictions[i, mask]
#        print(torch.tensor(probs))
        return torch.tensor(probs)

        
    def generate_sentence_p(self, tgt_dict, sentence, basic_tokenizer,bert_tokenizer, bert_model,tgt_tokens):
        probs = self.check_spelling_error(tgt_dict, sentence , self.basic_tokenizer, self.bert_tokenizer, self.bert_model, tgt_tokens)
        return probs

    def generate_teacher(self, tgt_dict, nat_input, tgt_tokens):
        output = nat_input.new(nat_input.size())
        bsz,_=nat_input.size()
        for sen  in range(bsz):
            output[sen]=(self.generate_sentence_p(tgt_dict, nat_input[sen], self.basic_tokenizer, self.bert_tokenizer, self.bert_model,tgt_tokens[sen]))
        return output
