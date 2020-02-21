

import re
import numpy as np


import scipy.stats


def get_intersection(AG,BSys):
    N = 0
    for b in BSys:
        if b in AG:
            N = N+1
    return N



def str_index_to_word_index(strString):

    str_w_dic = {}

    wordN = 1

    for i, s in enumerate(strString):
        if s == ',':
            wordN = wordN +1
            continue
        else:
            str_w_dic[i] = wordN

    return str_w_dic

def get_metric_number_with_RE(seqG, segSys, lens):
    #RE_PATTENS = '1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20'
    RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32|33[,34]*,35|36|37[,38]*,39|40|41[,42]*,43|44|45[,46]*,47|48|49[,50]*,51|52|53[,54]*,55|56|57[,58]*,59|60|61[,62]*,63|64|65[,66]*,67|68|69[,70]*,71|72'

    batch_size = len(seqG)

    batch_matrix_number = []

    for i in range(batch_size):

        sys_list_triple = []
        g_list_triple = []


        cur_L = lens[i]

        #GPU
        cur_g = ','.join(map(str,seqG[i][0:cur_L].tolist()))
        cur_sys = ','.join(map(str,segSys[i][0:cur_L].tolist()))

        #CPU
        # cur_g = ','.join(map(str, seqG[i][0:cur_L]))
        # cur_sys = ','.join(map(str, segSys[i][0:cur_L]))

        cur_g_str_to_w = str_index_to_word_index(cur_g)
        cur_sys_str_to_w = str_index_to_word_index((cur_sys))

        #Bug: len(cur_g) != len(cur_sys)

        miter = re.finditer(RE_PATTENS, cur_g)
        for m in miter:
            g_list_triple.append((m.group(), cur_g_str_to_w[m.start()]))

        miter = re.finditer(RE_PATTENS, cur_sys)
        for m in miter:
            sys_list_triple.append((m.group(), cur_sys_str_to_w[m.start()]))



        G = len(g_list_triple)
        Re = len(sys_list_triple)
        Correct = get_intersection(g_list_triple, sys_list_triple)

        batch_matrix_number.append([G, Re, Correct])

    return batch_matrix_number



def get_ner_PRF_score_from_listmatrix(listmatrix):

    np_matrix = np.array(listmatrix)

    G = np.sum(np_matrix[:,0])

    if np.sum(np_matrix[:,1]) !=0:
        SegPrecsion = np.sum(np_matrix[:,2]) / np.sum(np_matrix[:,1])
    else:
        SegPrecsion = 0  # if no return, precison = 0 or =1


    SegRecall = np.sum(np_matrix[:,2]) / G
    if (SegPrecsion+SegRecall)!=0:
        SegF1 = 2* SegPrecsion *SegRecall /(SegPrecsion+SegRecall)
    else:
        SegF1 = 0

    # print('Ground:', G)
    # print('Total Return:',np.sum(np_matrix[:,1]))
    # print('Correct:', np.sum(np_matrix[:,2]))



    return SegPrecsion,SegRecall,SegF1





def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    # h 1.96 * np.std(dev_F_scores) / np.sqrt(len(dev_F_scores))
    return m, h
