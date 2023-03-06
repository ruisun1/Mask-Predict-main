# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion
import  json

@register_criterion('confusion_loss3')
class LabelSmoothedLengthCrossEntropyCriterionCFL3(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        f = open('/home/sunrui/Mask-Predict-main/fairseq/criterions/soundConfusion_bert.json')
        dict = json.load(f)
        self.confusion_set = dict
        self.confusion_matrix = [[0 for _ in range(21128)] for _ in range (21128)]
        for key in dict:
            for word in dict[key]:
                self.confusion_matrix[int(key)][int(word)] = 1
                self.confusion_matrix[int(word)][int(key)] = 1
        f2 = open('/home/sunrui/Mask-Predict-main/fairseq/criterions/shapeConfusion_bert.json')
        dict = json.load(f2)
        self.confusion_set = dict
        for key in dict:
            for word in dict[key]:
                self.confusion_matrix[int(key)][int(word)] = 1
                self.confusion_matrix[int(word)][int(key)] = 1
        weight = torch.tensor(self.confusion_matrix, dtype = torch.float32)
        self.embedding = torch.nn.Embedding.from_pretrained(weight)
        #print("try",self.embedding(torch.tensor([6],dtype = torch.long))) 
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input']) # ** 传入字典
        '''    
        batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
        
        return from encoder:
              return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'predicted_lengths': predicted_lengths, # B x L
        }
    }'''
        loss, nll_loss, length_loss, ntokens = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = ntokens #TODO why not merge ntokens and sample_size? what is the difference?
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'length_loss': utils.item(length_loss.data) if reduce else length_loss.data,
            'ntokens': ntokens,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
    def make_confusion(self, word):
        if word not in self.confusion_set:
            return -1
        return self.confusion_set[word]

    def compute_loss(self, model, net_output, sample, reduce=True):
        #torch.autograd.set_detect_anomaly(True)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)   # fairseq_model.py bsz,seq_len,vocab_len
        lprobs_nll = lprobs.clone()
        lprobs_nll = lprobs.view(-1, lprobs_nll.size(-1)) # bsz*seq_len, vocab_len

        target = model.get_targets(sample, net_output) #bsz, seq_len
        target_nll = target.clone().view(-1,1)  #bsz*seqlen, 1
        non_pad_mask = target_nll.ne(self.padding_idx)


        loss_nll = -lprobs_nll.gather(dim=-1, index = target_nll)[non_pad_mask].sum()
        inputemb = torch.tensor(target_nll, dtype = torch.long)
        confusion_line = self.embedding(inputemb)
       # confusion_line = torch.tensor(self.confusion_matrix[target_nll[0]])
       # for word in target_nll[1:]:
       #     confusion_line = torch.cat((confusion_line, torch.tensor(self.confusion_matrix[word])), dim = 0)
        confusion_line = confusion_line.view(lprobs.size(0)*lprobs.size(1),-1).cuda() #bsz*seq_len, vocab_len
        confusion_probs = - torch.sum(lprobs_nll * confusion_line, dim = -1).unsqueeze(1)[non_pad_mask]
        l_probs = -lprobs_nll.gather(dim = -1, index= target_nll)[non_pad_mask]
        
        confusion_probs = confusion_probs + 0.1
        confusion_num = torch.sum(confusion_line, dim = -1).unsqueeze(1)[non_pad_mask]
        #print('target',target_nll[0])
        #for k in range(len(confusion_line[0])):
        #    if(confusion_line[0][k] == 1):
        #        print(k)

       # print("l_probs",l_probs)
       # print("cf num", confusion_num)
       # print("confusioon_probs", confusion_probs)
        #loss_cf = torch.sum(l_probs/(confusion_probs + l_probs))
        loss_cf = torch.sum(l_probs*(confusion_num)/confusion_probs)
        #print(loss_cf, loss_nll)
        loss = -loss_cf + loss_nll
        #print(l_probs, confusion_num, confusion_probs)
       # loss = torch.sum(l_probs*(confusion_num+1) - confusion_probs)
       # print(loss)
        #print(lo
        return loss, loss , loss_nll , non_pad_mask.sum().data.item()

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'length_loss': sum(log.get('length_loss', 0) for log in logging_outputs) / nsentences / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
