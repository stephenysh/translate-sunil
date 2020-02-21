
import pickle
import re

import numpy as np

from model.utils.generate_vocabulary_for_dataset import KnownWordCharacter


def IsNumber(word):

    p = r'[\d+].*\d$'

    RegFlag = re.match(p,word)

    if RegFlag == None:
        return False
    else:
        return True


def get_character_mapping(WCclass,word):
    WL = []
    for c in word:
        if c in WCclass.char2index:
            WL.append(WCclass.char2index[c])
        else:
            WL.append(1)

    return np.array(WL)



def convert_to_ids(sentArray, dataClass, wlowered, clowered):
    All_word_mapping = []
    All_char_mapping = []


    for cur_s in sentArray:
        cur_words = cur_s.split(' ')

        sent_word_map_list = []
        sent_char_map_list = []
        for cur_w in cur_words:

            # step 1: number preprocessing
            if wlowered == True:
                pre_cur_w = cur_w.lower()
            else:
                pre_cur_w = cur_w

            if IsNumber(cur_w) == True:  # It is a number
                sent_word_map_list.append(1)  # "RE_DIGITS":1

            else:  # It is not a number
                if pre_cur_w not in dataClass.word2index:
                    sent_word_map_list.append(2)  # "UNKNOWN":2
                else:
                    sent_word_map_list.append(dataClass.word2index[pre_cur_w])  # like [1,2,3,4,5,6,7]

            # Step 2: get mapping for characters
            if clowered == True:
                sent_char_map_list.append(
                    get_character_mapping(dataClass, cur_w.lower()))  # like [array[1,2,3], array[1,2,3,4,5]]
            else:
                sent_char_map_list.append(
                    get_character_mapping(dataClass, cur_w))  # like [array[1,2,3], array[1,2,3,4,5]]



        All_word_mapping.append(np.array(sent_word_map_list))
        All_char_mapping.append(sent_char_map_list)

    All_word_mapping = np.array(All_word_mapping)
    All_char_mapping = np.array(All_char_mapping)


    return All_word_mapping, All_char_mapping



if __name__ == '__main__':



    w_c_class_path = '/home/jl1/MyIIAIData/FewShotNERData/input/intra_domain_cross_type/fgner/Word_Cha_Class_2833_109.pickle'
    Word_Cha_Class =  pickle.load(open(w_c_class_path, "rb"))
