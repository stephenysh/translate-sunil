import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from model.nn_models.wordrep import WordRep
import torch.nn.utils.rnn as R

from model.nn_models.crflayer import MyCRF



class NERTagger(nn.Module):
    def __init__(self,params,DEVICE):
        super(NERTagger, self).__init__()


        self.word_rep_layer = WordRep(params, DEVICE)

        in_d = params.dim_w + params.n_char_filters

        params.word_total_dim = in_d

        self.crf_layer = MyCRF(2*params.dim_rnn, 73, DEVICE).to(DEVICE)




        self.encoder_rnn_layer = getattr(nn, 'GRU')(input_size=in_d,
                                              hidden_size=params.dim_rnn,
                                              num_layers=params.rnn_layers,
                                              bidirectional=False if params.direction_no == 1 else True,
                                              dropout=params.rnn_dropout,
                                              batch_first=True)
        self.device = DEVICE

        self.nnDropout = nn.Dropout(params.nn_dropout)



    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)  #TODO: the output is sorted?, dont changed

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        return output, h



    def forward(self, Word_ints, Char_ids, Label_y, Sent_length):



        X = self.word_rep_layer(Word_ints, Char_ids)

        X = self.nnDropout(X)

        o, h = self._run_rnn_packed(self.encoder_rnn_layer, X, Sent_length)  # batch_first=True
        o = o.contiguous()

        o = self.nnDropout(o)


        crf_loss = self.crf_layer.crf_neglog_loss(o, Label_y, Sent_length)


        return crf_loss



    def predict_crf(self, Word_ints, Char_ids, Sent_length):

        X = self.word_rep_layer(Word_ints, Char_ids)

        X = self.nnDropout(X)

        o, h = self._run_rnn_packed(self.encoder_rnn_layer, X, Sent_length)  # batch_first=True
        o = o.contiguous()

        o = self.nnDropout(o)

        scores, paths = self.crf_layer.viterbi_decode(o, Sent_length)

        return paths




