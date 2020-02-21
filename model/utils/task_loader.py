
import copy
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from flair.data import Sentence

from model.utils.map_word_char_to_ints import convert_to_ids


def compute_gain(ori_dic, add_list,input_shot):

    isGain = None


    if len(ori_dic) == 0:
        isGain = True
    else:
        temp_ori_dic = copy.deepcopy(ori_dic)

        for t in add_list:
            temp_ori_dic[t] = temp_ori_dic[t] + 1

        if len(temp_ori_dic.keys()) > len(ori_dic.keys()):
            isGain = True
        else:
            A1 = np.array(list(temp_ori_dic.values()))   #[3,4,5,6,5]
            A0 = np.array(list(ori_dic.values()))  #[2,3,5,5,4]

            temp_gain = (A1 - A0) * A1

            count_gain_n = ((temp_gain <= input_shot) & (temp_gain> 0)).sum()

            if count_gain_n > 0:
                isGain = True
            else:
                isGain = False

    return isGain








def tags_to_numbers(allArray, tagList):
    all_mapping_dic ={}
    all_mapping_dic['O'] = 0

    patten_list = []

    i = 1
    for t in tagList:
        all_mapping_dic['B-' + t] = i
        all_mapping_dic['I-' + t] = i + 1
        all_mapping_dic['E-' + t] = i + 2
        all_mapping_dic['S-' + t] = i + 3

        patten_list.append(str(i) + '[,' + str(i + 1) + ']*,' + str(i + 2))
        patten_list.append(str(i + 3))

        i = i + 4

    #RE_PATTENS = '|'.join(patten_list)
    #5 ways:
    #'1[,2]*,3|4|5[,6]*,7|8|9[,10]*,11|12|13[,14]*,15|16|17[,18]*,19|20'


    mapped_array = []
    for line in allArray:
        # mapped_list = map(all_mapping_dic.get,line.split(' '))
        # mapped_array.append(mapped_list)

        mapped_array.append( [all_mapping_dic[k] for k in line.split(' ')] )


    mapped_array = np.array(mapped_array)


    return mapped_array




def sample_task_from_text(N_ways, support_k, query_k, sentences, tags):


    support_index = []
    query_index = []


    random_index = list(range(0,len(sentences)))
    random.shuffle(random_index)


    support_task_tags_dic ={}


    stop_support = False
    support_task_tags_dic = defaultdict(lambda: 0, support_task_tags_dic)



    # sample instances for support
    for loop_index in random_index:
        cur_tag = tags[loop_index]

        cur_t_list = np.array(cur_tag.split(' '))
        non_O_tags = cur_t_list[cur_t_list != 'O']

        cur_sent_tag_list = [t[2:] for t in non_O_tags]

        union_tags = set(support_task_tags_dic.keys()).union(set(cur_sent_tag_list))

        # sample instances for support
        if len(union_tags) <= N_ways and stop_support == False:

            isGain = compute_gain(support_task_tags_dic, cur_sent_tag_list, support_k)
            if isGain == True:
                support_index.append(loop_index)
                for t in cur_sent_tag_list:
                    support_task_tags_dic[t] = support_task_tags_dic[t] + 1

                if (np.array(list(support_task_tags_dic.values())) > support_k-1).sum() == N_ways:  #TODO: all ways are done
                    stop_support = True
                    break




    # sample instances for query
    remain_index = np.setdiff1d(random_index, support_index)

    query_task_tags_dic = {}
    query_task_tags_dic = defaultdict(lambda: 0, query_task_tags_dic)
    for k in support_task_tags_dic.keys():
        query_task_tags_dic[k] = 0

    stop_query = False


    for loop_index in remain_index:
        cur_tag = tags[loop_index]

        cur_t_list = np.array(cur_tag.split(' '))
        non_O_tags = cur_t_list[cur_t_list != 'O']

        cur_sent_tag_list = [t[2:] for t in non_O_tags]

        union_tags = set(query_task_tags_dic.keys()).union(set(cur_sent_tag_list))

        # sample instances for support
        if len(union_tags) <= N_ways and stop_query == False:

            isGain = compute_gain(query_task_tags_dic, cur_sent_tag_list, query_k)
            if isGain == True:
                query_index.append(loop_index)
                for t in cur_sent_tag_list:
                    query_task_tags_dic[t] = query_task_tags_dic[t] + 1

                if (np.array(list(query_task_tags_dic.values())) > query_k - 1).sum() == N_ways: #TODO: all ways are done
                    stop_query = True
                    break

    task_return = {}

    task_return['support_sentences'] = sentences[support_index]
    task_return['support_string_tags'] = tags[support_index]



    task_return['query_sentences'] = sentences[query_index]
    task_return['query_string_tags'] = tags[query_index]
    task_return['task_types'] = list(support_task_tags_dic.keys())

    task_return['support_int_tags'] = tags_to_numbers(task_return['support_string_tags'], task_return['task_types'])
    task_return['query_int_tags'] = tags_to_numbers(task_return['query_string_tags'], task_return['task_types'])

    # print(task_return)
    # print('support_index',len(support_index))
    # print('query_index',len(query_index))
    # print('support_task_tags_dic',support_task_tags_dic)
    # print('query_task_tags_dic',query_task_tags_dic)


    if len(task_return['query_int_tags'])==0:
        task_return['query_sentences'] = task_return['support_sentences']
        task_return['query_string_tags'] = task_return['support_string_tags']
        task_return['query_int_tags'] =  task_return['support_int_tags']




    return task_return




def task_to_model_input(task, dataDict, wordLower, charLower):


    support_word_ids, support_char_ids = convert_to_ids(task['support_sentences'], dataDict, wordLower, charLower)

    query_word_ids, query_char_ids = convert_to_ids(task['query_sentences'], dataDict, wordLower, charLower)





    support_sentences = []
    query_sentences = []


    support_char_lengths = []
    for i, s in enumerate(task['support_sentences']):
        support_sentences.append(Sentence(s, use_tokenizer=False))
        support_char_lengths.extend([len(xx) for xx in support_char_ids[i]])
    maxchar_L_support = np.max(support_char_lengths)


    query_cha_lengths = []
    for i, s in enumerate(task['query_sentences']):
        query_sentences.append(Sentence(s, use_tokenizer=False))
        query_cha_lengths.extend([len(xx) for xx in query_char_ids[i]])
    maxchar_L_query = np.max(query_cha_lengths)


    support_y = []
    query_y = []


    s_L = [len(l) for l in task['support_int_tags']]
    maxL_sup = max(s_L)
    for t in task['support_int_tags']:
        support_y.append(np.pad(t, (0, maxL_sup - len(t)), 'constant', constant_values=(0, 0)))  #TODO: becareful, CRF will use the index

    q_L = [len(l) for l in task['query_int_tags']]
    maxL_query = max(q_L)
    for t in task['query_int_tags']:
        query_y.append(np.pad(t, (0, maxL_query - len(t)), 'constant',constant_values=(0, 0)))


    support_y = np.array(support_y)
    query_y = np.array(query_y)


    torch_support_y = torch.from_numpy(support_y.astype(np.int64))
    torch_support_length = torch.from_numpy(np.array(s_L).astype(np.int64))


    torch_query_y = torch.from_numpy(query_y.astype(np.int64))
    torch_query_length = torch.from_numpy(np.array(q_L).astype(np.int64))



    #covert to Ids

    # TODO: padding support words, characters
    pad_support_word_ids = []
    for aa in support_word_ids:
        pad_support_word_ids.append(np.pad(aa, (0, maxL_sup - len(aa)), 'constant'))

    pad_support_word_ids = np.array(pad_support_word_ids)

    pad_support_char_ids = []
    for ddd in support_char_ids:
        for i in range(maxL_sup):
            if i < len(ddd):
                dd = ddd[i]
            else:
                dd = np.zeros(1, dtype=np.int32)
            pad_support_char_ids.append(np.pad(dd, (0, maxchar_L_support - len(dd)), 'constant'))

    pad_support_char_ids = np.array(pad_support_char_ids)
    pad_support_char_ids = pad_support_char_ids.reshape(pad_support_word_ids.shape[0], pad_support_word_ids.shape[1], maxchar_L_support)

    # TODO: padding query words, characters
    pad_query_word_ids = []
    for aa in query_word_ids:
        pad_query_word_ids.append(np.pad(aa, (0, maxL_query - len(aa)), 'constant'))

    pad_query_word_ids = np.array(pad_query_word_ids)

    pad_query_char_ids = []
    for ddd in query_char_ids:
        for i in range(maxL_query):
            if i < len(ddd):
                dd = ddd[i]
            else:
                dd = np.zeros(1, dtype=np.int32)
            pad_query_char_ids.append(np.pad(dd, (0, maxchar_L_query - len(dd)), 'constant'))

    pad_query_char_ids = np.array(pad_query_char_ids)
    pad_query_char_ids = pad_query_char_ids.reshape(pad_query_word_ids.shape[0], pad_query_word_ids.shape[1], maxchar_L_query)

    #double_check_results(pad_support_word_ids, pad_support_char_ids, pad_query_word_ids, pad_query_char_ids, dataDict)



    torch_pad_support_word_ids = torch.torch.from_numpy(pad_support_word_ids.astype(np.int64))
    torch_pad_support_char_ids = torch.torch.from_numpy(pad_support_char_ids.astype(np.int64))

    torch_pad_query_word_ids = torch.torch.from_numpy(pad_query_word_ids.astype(np.int64))
    torch_pad_query_char_ids = torch.torch.from_numpy(pad_query_char_ids.astype(np.int64))





    return support_sentences,torch_pad_support_word_ids, torch_pad_support_char_ids, torch_support_y, torch_support_length, \
           query_sentences, torch_pad_query_word_ids, torch_query_y, torch_pad_query_char_ids, torch_query_length





def double_check_results(s_w_ids, s_c_ids, q_w_ids, q_c_ids, dataDict):

    #check the first one

    word_L = s_w_ids[0]
    char_L = s_c_ids[0]

    sent_from_w = [dataDict.index2word[w] for w in word_L]
    print('support sentence from words:', ' '.join(sent_from_w))

    sent_from_char = ''
    for cl in char_L:
        for c in cl:
            sent_from_char = sent_from_char + dataDict.index2char[c]
        sent_from_char = sent_from_char + ' '
    print('support sentence from characters:', sent_from_char)



    word_L = q_w_ids[0]
    char_L = q_c_ids[0]

    sent_from_w = [dataDict.index2word[w] for w in word_L]
    print('query sentence from words:', ' '.join(sent_from_w))

    sent_from_char = ''
    for cl in char_L:
        for c in cl:
            sent_from_char = sent_from_char + dataDict.index2char[c]
        sent_from_char = sent_from_char + ' '
    print('query sentence from characters:', sent_from_char)





if __name__ == '__main__':



    random.seed(50)


    in_p = '/home/jl1/MyIIAIData/FewShotNERData/input/intra_domain_cross_type/fgner'

    sent_array = []
    with open(os.path.join(in_p, 'final_train_sentences.txt'), 'r', encoding='utf-8') as lines:
        for line in lines:
            sent_array.append(line.strip('\n'))

    sent_array = np.array(sent_array)



    tag_array = []
    with open(os.path.join(in_p, 'final_train_sentence_tags.txt'), 'r', encoding='utf-8') as lines:
        for line in lines:
            tag_array.append(line.strip('\n'))

    tag_array = np.array(tag_array)

    w_c_class_path = '/home/jl1/MyIIAIData/FewShotNERData/input/intra_domain_cross_type/fgner/Word_Cha_Class_2833_109.pickle'
    Word_Cha_Class = pickle.load(open(w_c_class_path, "rb"))

    for kk in range(50):
        cur_task = sample_task_from_text(N_ways=5, support_k=5, query_k=5, sentences= sent_array, tags=tag_array)

        support_sentences,torch_support_y, torch_support_length, query_sentences, torch_query_y, torch_query_length = task_to_model_input(cur_task, Word_Cha_Class, wordLower=True, charLower= False)
