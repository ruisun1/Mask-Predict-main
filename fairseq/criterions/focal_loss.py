# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch.nn as nn
import torch

@register_criterion('focal_loss')
class LabelSmoothedLengthCrossEntropyCriterionF(FairseqCriterion):

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

        import torch
        s_prob = torch.nn.functional.softmax(net_output[0].view(-1, net_output[0].size(2)),dim = 1) #bsz*seq, vocab_len
        target = model.get_targets(sample, net_output).view(-1, 1) # return sample['target'],[batch*seq_len,1]
        nll_loss = - s_prob.log().gather(dim=-1, index=target)
        gamma = 1
        s_prob = s_prob.clamp(min=0.0001,max=1)
        focal_loss = torch.mul(torch.pow((1 - s_prob.gather(dim=-1, index=target)), gamma), nll_loss)
        non_pad_mask = target.ne(self.padding_idx)
        focal_loss = focal_loss[non_pad_mask]
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


