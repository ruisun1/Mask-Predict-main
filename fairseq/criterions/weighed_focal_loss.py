# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('weighted_focal_cross_entropy')
class LabelSmoothedLengthCrossEntropyCriterionWF(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

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

        weighted_loss, focal_loss, ntokens = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = ntokens #TODO why not merge ntokens and sample_size? what is the difference?
        logging_output = {
            'loss': utils.item(weighted_loss.data) if reduce else weighted_loss.data,
            'nll_loss': utils.item(focal_loss.data) if reduce else focal_loss.data,
            'ntokens': ntokens,
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return weighted_loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)   # fairseq_model.py
        probs = model.get_normalized_probs(net_output, log_probs=False)
        print("lprobs, probs")
        print(lprobs == probs.log())
        # get_noralized_probs: fairseq_model.py if(has attr(self,'decoder'), return decoder.get_normalized_probs
        # =net_output[0]
        #lprobs=[batch,seq_len,prob]
        dec_source = sample['net_input']['prev_output_tokens'].view(-1,1)
        non_mask_locs = dec_source.ne(self.mask_idx) #[bsz*seq_len,1]
        mask_locs = dec_source.eq(self.mask_idx)
        lprobs = lprobs.view(-1, lprobs.size(-1))# [batch*seqlen,prob]
        probs = probs.view(-1, probs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1) # return sample['target'],[batch*seq_len,1]
        non_pad_mask = target.ne(self.padding_idx)  #.ne 返回每个位置是否相等的比较，不等为true #[batch*seq_len, 1]
        mask_locs = mask_locs * non_pad_mask
        non_mask_locs = non_mask_locs * non_pad_mask

        gamma= 0.5
        nll_loss = -lprobs.gather(dim=-1, index=target)
        focal_loss = torch.mul(torch.pow((1 - probs.gather(dim=-1, index=target)), gamma), nll_loss)
        #print("focal loss",focal_loss.size())
        #print(non_mask_locs.size(), mask_locs.size())
        weighted_loss = focal_loss[non_mask_locs].sum() + 1.5*focal_loss[mask_locs].sum()
        #smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        #length_loss = -length_lprobs.gather(dim=-1, index=length_target)
        #if reduce:
        #    weighted_loss = weighted_loss.sum()
            #smooth_loss = smooth_loss.sum()
            #length_loss = length_loss.sum(

        #focal_loss = focal_loss[non_pad_mask].sum()
        focal_loss = focal_loss.sum()
        return focal_loss, focal_loss,  non_pad_mask.sum().data.item()

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

