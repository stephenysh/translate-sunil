


import numpy as np
import os
import pickle
import pandas as pd
import re
import random

import copy



class KnownWordCharacter:
    def __init__(self,pretrained_word_dict, wlowered, clowered):

        self.index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}
        self.word2index = {"RE_DIGITS": 1, "UNKNOWN": 2, "PADDING": 0}


        self.word2count = {"RE_DIGITS": 1, "UNKNOWN": 1, "PADDING": 1}
        self.ori_n_words = 3  # Count SOS and EOS
        self.ori_word2index = {"RE_DIGITS": 1, "UNKNOWN": 2, "PADDING": 0}
        self.ori_index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}


        self.char2index = {'P_':0, 'U_':1}
        self.index2char = {0: 'P_', 1:'U_'}
        self.char2count = {'P_': 1, 'U_': 1}

        self.n_chars = 2
        self.wlowered = wlowered
        self.clowered = clowered

        self.pretrained_word_dict = pickle.load(open(pretrained_word_dict, "rb"))

    def addSentence(self, sentence):
        for word in sentence.strip('\n').strip('\r').split(' '):
            self.addWord(word)

    def addWord(self, word):
        if self.wlowered == True:
            processed_word = word.lower()
        else:
            processed_word = word

        if (processed_word not in self.ori_word2index) and (processed_word in self.pretrained_word_dict):
            self.ori_word2index[processed_word] = self.ori_n_words
            self.ori_index2word[self.ori_n_words] = processed_word
            self.ori_n_words += 1

            self.word2count[processed_word] = 1

        elif processed_word in self.ori_word2index and processed_word in self.pretrained_word_dict:
            self.word2count[processed_word] += 1
        else:
            self.word2count['UNKNOWN'] += 1



        # each word should be parsed into characters
        if self.clowered == True:
            pre_char_word = word.lower()
        else:
            pre_char_word = word


        for cur_char in pre_char_word:
            self.addChar(cur_char)


    def addChar(self, character):
        if character not in self.char2index:
            self.char2index[character] = self.n_chars
            self.index2char[self.n_chars] = character
            self.n_chars += 1
            self.char2count[character] = 1
        else:

            self.char2count[character] += 1


    def filtering(self, minWFreq, minChFreq):
        loopN = 3
        for i in range(3, len(self.ori_index2word)):
            cur_w  = self.ori_index2word[i]
            cur_F = self.word2count[cur_w]

            if cur_F >= minWFreq:
                self.index2word[loopN] = cur_w
                self.word2index[cur_w] = loopN
                loopN += 1


        org_index2char = copy.deepcopy(self.index2char)

        self.char2index = {'P_': 0, 'U_': 1}
        self.index2char = {0: 'P_', 1: 'U_'}

        loopN = 2
        for i in range(2, len(org_index2char)):
            cur_char = org_index2char[i]
            cur_charF = self.char2count[cur_char]

            if cur_charF >= minChFreq:
                self.index2char[loopN] = cur_char
                self.char2index[cur_char] = loopN
                loopN += 1



def generate_vocabulary_save_embeddings(lowF, txtfile, w_dict, v_matrix, save_p):

    sent_array = []
    with open(txtfile, 'r', encoding='utf-8') as lines:
        for line in lines:
            sent_array.append(line.strip('\n'))



    Word_Cha_Class = KnownWordCharacter(w_dict, wlowered = True, clowered = True)

    for cur_s in sent_array:
        Word_Cha_Class.addSentence(cur_s)



    print('No Unique Characters:',Word_Cha_Class.n_chars)
    print('No Unique Known Words:',Word_Cha_Class.ori_n_words)


    Word_Cha_Class.filtering(minWFreq = lowF, minChFreq =1)

    print('No unique known words after filtering', len(Word_Cha_Class.index2word))


    final_w_size = len(Word_Cha_Class.index2word)
    final_cha_size = len(Word_Cha_Class.index2char)

    # save vocabulary
    with open(os.path.join(save_p, 'Word_Cha_Class_%d_%d.pickle'%(final_w_size, final_cha_size)), 'wb') as f:
        pickle.dump(Word_Cha_Class, f)


    # save vectors


    GLOVE_VECTORS = pickle.load(open(v_matrix, 'rb'))
    GLOVE_WORDS = pickle.load(open(w_dict, "rb"))

    S_W_Matrix = []

    # First initial zeros  "RE_DIGITS":1,"UNKNOWN":2,"PADDING":0
    S_W_Matrix.append(np.zeros(300))  # "PADDING":0
    S_W_Matrix.append(GLOVE_VECTORS[GLOVE_WORDS['0'], :])  # "RE_DIGITS":1

    sample_index = random.sample(range(GLOVE_VECTORS.shape[0]), 1000)
    unknown_vector = np.mean(GLOVE_VECTORS[sample_index, :], axis=0)
    print('unknown_vector:', unknown_vector.shape)

    S_W_Matrix.append(unknown_vector)

    for i in range(3, len(Word_Cha_Class.index2word)):
        print(i)
        cur_w = Word_Cha_Class.index2word[i]

        cur_v = GLOVE_VECTORS[GLOVE_WORDS[cur_w], :]

        S_W_Matrix.append(cur_v)

    S_W_Matrix = np.array(S_W_Matrix)

    print(S_W_Matrix.shape)
    print('word size:', len(Word_Cha_Class.index2word))
    print('char size:', len(Word_Cha_Class.index2char))

    with open(os.path.join(save_p, 'save_initializaton_embeddings_%d_%d.pickle'%(final_w_size, final_cha_size)), 'wb') as f:
        pickle.dump(S_W_Matrix, f)






if __name__ == '__main__':


    input_text = '/home/jl1/MyIIAIData/FewShotNERData/input/cross_domain_cross_type/ontonotes2fgner/xx.txt'
    input_embedding_Word_dict = '/home/jl1/MyIIAIData/glove/glove_words.pickle'
    input_embedding_Vector_matrix = '/home/jl1/MyIIAIData/glove/glove_vectors.pickle'
    save_path = '/home/jl1/MyIIAIData/FewShotNERData/input/cross_domain_cross_type/xxx'
    #Word_Cha_Class = KnownWordCharacter(w_dict, wlowered=True, clowered=False)
    generate_vocabulary_save_embeddings(3,input_text, input_embedding_Word_dict, input_embedding_Vector_matrix, save_path)