import torch
import torch.nn as nn
import torch.nn.init as I
from torch.autograd import Variable


def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))


class MyCRF(nn.Module):
    def __init__(self, f_size, vocab_size, DEVICE):
        super(MyCRF, self).__init__()

        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels))

        self.entity_type_direct_fc = torch.nn.Linear(f_size, self.n_labels)

        self.device = DEVICE

    def reset_parameters(self):
        I.normal(self.transitions.data, 0, 1)

    def forward_score(self, logits, lens):

        """
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        alpha = Variable(alpha)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, features, lens):

        logits = self.entity_type_direct_fc(features)

        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        vit = Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[self.stop_idx].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        # idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        # for argmax in reversed(pointers):
        #     idx_exp = idx.unsqueeze(-1)
        #     idx = torch.gather(argmax, 1, idx_exp)
        #     idx = idx.squeeze(-1)
        #
        #     paths.insert(0, idx.unsqueeze(1))

        for i in reversed(range(len(pointers))):
            argmax = pointers[i, :, :]
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2).to(self.device)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), trn.size(0))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1, max_len=seq_len + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score

    def _seq_score(self, logits, y, lens):
        batch_size, seq_len = y.size()

        y_exp = y.unsqueeze(-1)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        mask = sequence_mask(lens, max_len=seq_len).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

    def gold_score(self, y, lens, logits):

        transition_score = self.transition_score(y, lens)
        seq_score = self._seq_score(logits, y, lens)

        score = transition_score + seq_score

        return score

    def crf_neglog_loss(self, features, labels, lens):

        logits = self.entity_type_direct_fc(features)

        norm_score = self.forward_score(logits, lens)

        sequence_score = self.gold_score(labels, lens, logits)

        negloglik = norm_score - sequence_score

        negloglik = negloglik.mean()

        return negloglik


def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    if lens.data.is_cuda:
        # ranges = ranges.cuda()
        ranges = ranges.to(lens.device)

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask
