
import numpy as np
import random
import torch

from model.utils.map_word_char_to_ints import convert_to_ids


def tag_to_number(tagArray):
    TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-WORK_OF_ART': 5, 'I-WORK_OF_ART': 6,
                     'E-WORK_OF_ART': 7, 'S-WORK_OF_ART': 8, 'B-LOC': 9, 'I-LOC': 10, 'E-LOC': 11, 'S-LOC': 12,
                     'B-CARDINAL': 13, 'I-CARDINAL': 14, 'E-CARDINAL': 15, 'S-CARDINAL': 16, 'B-EVENT': 17,
                     'I-EVENT': 18, 'E-EVENT': 19, 'S-EVENT': 20, 'B-NORP': 21, 'I-NORP': 22, 'E-NORP': 23,
                     'S-NORP': 24, 'B-GPE': 25, 'I-GPE': 26, 'E-GPE': 27, 'S-GPE': 28, 'B-DATE': 29, 'I-DATE': 30,
                     'E-DATE': 31, 'S-DATE': 32, 'B-PERSON': 33, 'I-PERSON': 34, 'E-PERSON': 35, 'S-PERSON': 36,
                     'B-FAC': 37, 'I-FAC': 38, 'E-FAC': 39, 'S-FAC': 40, 'B-QUANTITY': 41, 'I-QUANTITY': 42,
                     'E-QUANTITY': 43, 'S-QUANTITY': 44, 'B-ORDINAL': 45, 'I-ORDINAL': 46, 'E-ORDINAL': 47,
                     'S-ORDINAL': 48, 'B-TIME': 49, 'I-TIME': 50, 'E-TIME': 51, 'S-TIME': 52, 'B-PRODUCT': 53,
                     'I-PRODUCT': 54, 'E-PRODUCT': 55, 'S-PRODUCT': 56, 'B-PERCENT': 57, 'I-PERCENT': 58,
                     'E-PERCENT': 59, 'S-PERCENT': 60, 'B-MONEY': 61, 'I-MONEY': 62, 'E-MONEY': 63, 'S-MONEY': 64,
                     'B-LAW': 65, 'I-LAW': 66, 'E-LAW': 67, 'S-LAW': 68, 'B-LANGUAGE': 69, 'I-LANGUAGE': 70,
                     'E-LANGUAGE': 71, 'S-LANGUAGE': 72}


    #RE_PATTENS = r'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20|21[,22]*,23|24|25[,26]*,27|28|29[,30]*,31|32|33[,34]*,35|36|37[,38]*,39|40|41[,42]*,43|44|45[,46]*,47|48|49[,50]*,51|52|53[,54]*,55|56|57[,58]*,59|60|61[,62]*,63|64|65[,66]*,67|68|69[,70]*,71|72'

    mapped_array = []
    for line in tagArray:
        # mapped_list = map(all_mapping_dic.get,line.split(' '))
        # mapped_array.append(mapped_list)

        mapped_array.append([TAG_TO_NUNBER[k] for k in line.split(' ')])

    mapped_array = np.array(mapped_array)

    return mapped_array







def double_check_results(s_w_ids, s_c_ids, dataDict):

    #check the first one

    word_L = s_w_ids[0]
    char_L = s_c_ids[0]

    sent_from_w = [dataDict.index2word[w] for w in word_L]
    print('sentence from words:', ' '.join(sent_from_w))

    sent_from_char = ''
    for cl in char_L:
        for c in cl:
            sent_from_char = sent_from_char + dataDict.index2char[c]
        sent_from_char = sent_from_char + ' '
    print('sentence from characters:', sent_from_char)







def sample_a_batch_from_text(bsize, dataDict, sentsArray, tagsArray,sampleFlag, device):




    if sampleFlag == True:
        select_index = np.array(random.sample(range(len(sentsArray)), bsize))
    else:
        select_index = np.array(range(len(sentsArray)))








    batch_word_ids, batch_char_ids = convert_to_ids(sentsArray[select_index], dataDict, True, True)

    batch_tag_ids = tag_to_number(tagsArray[select_index])


    sen_L = [len(l) for l in batch_tag_ids]
    maxL_word = max(sen_L)

    char_L = [len(l) for s in batch_char_ids for l in s ]
    maxL_char = max(char_L)
    if maxL_char <= 5:
       maxL_char = 5

    batch_length = np.array(sen_L)

    #TODO: pad Y

    pad_y = []



    for t in batch_tag_ids:
        pad_y.append(np.pad(t, (0, maxL_word - len(t)), 'constant', constant_values=(0, 0)))  # TODO: becareful, CRF will use the index

    pad_y = np.array(pad_y)



    #TODO: pad X
    pad_word_ids = []
    for aa in batch_word_ids:
        pad_word_ids.append(np.pad(aa, (0, maxL_word - len(aa)), 'constant'))
    pad_word_ids = np.array(pad_word_ids)

    pad_char_ids = []
    for ddd in batch_char_ids:
        for i in range(maxL_word):
            if i < len(ddd):
                dd = ddd[i]
            else:
                dd = np.zeros(1, dtype=np.int32)
            pad_char_ids.append(np.pad(dd, (0, maxL_char - len(dd)), 'constant'))

    pad_char_ids = np.array(pad_char_ids)
    pad_char_ids = pad_char_ids.reshape(pad_word_ids.shape[0], pad_word_ids.shape[1], maxL_char)

    #double_check_results(pad_word_ids, pad_char_ids, dataDict)

    torch_batch_y = torch.torch.from_numpy(pad_y.astype(np.int64)).to(device)

    torch_batch_pad_word_ids= torch.torch.from_numpy(pad_word_ids.astype(np.int64)).to(device)

    torch_batch_pad_char_ids = torch.torch.from_numpy(pad_char_ids.astype(np.int64)).to(device)

    torch_batch_length = torch.from_numpy(np.array(batch_length).astype(np.int64)).to(device)



    return torch_batch_pad_word_ids, torch_batch_pad_char_ids, torch_batch_y, torch_batch_length





