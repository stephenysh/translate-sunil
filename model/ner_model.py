# from NERModule import NERModel
import pickle

import nltk
import numpy as np
import torch

from model.nn_models.ner_model import NERTagger
from model.utils.pickle_desrialiser import CustomUnpickler
from model.utils.sample_input_for_model import sample_a_batch_from_text


class NERModel:
    def __init__(self, run_on_cpu=False):
        word_cha_class_path = 'data/ner_data/Word_Cha_Class_20173_87.pickle'
        params_path = str('data/ner_data/a_params.pickle')
        model_path = str('data/ner_data/best_model_on_dev.torchsave')

        word_class_unpickler = CustomUnpickler(open(word_cha_class_path, "rb"))
        self.Word_Cha_Class = word_class_unpickler.load()
        params = pickle.load(open(params_path, 'rb'))
        params.text_path = str('data/ner_data')

        # prediction on CPU
        if run_on_cpu == True:
            self.DEVICE = torch.device("cpu")
            loaded_dic = torch.load(model_path, map_location=lambda storage, loc: storage)

        else:
            # prediction on GPU
            gpu_no = 0
            self.DEVICE = torch.device("cuda:%d" % gpu_no)
            loaded_dic = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpu_no))

        self.loaded_model = NERTagger(params, self.DEVICE).to(self.DEVICE)
        self.loaded_model.load_state_dict(loaded_dic['model'])

        self.TAG_TO_NUNBER = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'E-ORG': 3, 'S-ORG': 4, 'B-WORK_OF_ART': 5,
                              'I-WORK_OF_ART': 6,
                              'E-WORK_OF_ART': 7, 'S-WORK_OF_ART': 8, 'B-LOC': 9, 'I-LOC': 10, 'E-LOC': 11, 'S-LOC': 12,
                              'B-CARDINAL': 13, 'I-CARDINAL': 14, 'E-CARDINAL': 15, 'S-CARDINAL': 16, 'B-EVENT': 17,
                              'I-EVENT': 18, 'E-EVENT': 19, 'S-EVENT': 20, 'B-NORP': 21, 'I-NORP': 22, 'E-NORP': 23,
                              'S-NORP': 24, 'B-GPE': 25, 'I-GPE': 26, 'E-GPE': 27, 'S-GPE': 28, 'B-DATE': 29,
                              'I-DATE': 30,
                              'E-DATE': 31, 'S-DATE': 32, 'B-PERSON': 33, 'I-PERSON': 34, 'E-PERSON': 35,
                              'S-PERSON': 36,
                              'B-FAC': 37, 'I-FAC': 38, 'E-FAC': 39, 'S-FAC': 40, 'B-QUANTITY': 41, 'I-QUANTITY': 42,
                              'E-QUANTITY': 43, 'S-QUANTITY': 44, 'B-ORDINAL': 45, 'I-ORDINAL': 46, 'E-ORDINAL': 47,
                              'S-ORDINAL': 48, 'B-TIME': 49, 'I-TIME': 50, 'E-TIME': 51, 'S-TIME': 52, 'B-PRODUCT': 53,
                              'I-PRODUCT': 54, 'E-PRODUCT': 55, 'S-PRODUCT': 56, 'B-PERCENT': 57, 'I-PERCENT': 58,
                              'E-PERCENT': 59, 'S-PERCENT': 60, 'B-MONEY': 61, 'I-MONEY': 62, 'E-MONEY': 63,
                              'S-MONEY': 64,
                              'B-LAW': 65, 'I-LAW': 66, 'E-LAW': 67, 'S-LAW': 68, 'B-LANGUAGE': 69, 'I-LANGUAGE': 70,
                              'E-LANGUAGE': 71, 'S-LANGUAGE': 72}

        self.need_entity_tag = ['B-GPE', 'S-GPE', 'B-ORG', 'S-ORG', 'B-WORK_OF_ART', 'S-WORK_OF_ART', 'B-LOC', 'S-LOC',
                                'B-NORP', 'S-NORP', 'B-PERSON', 'S-PERSON', 'E-PERSON', 'B-FAC', 'S-FAC', 'B-PRODUCT',
                                'S-PRODUCT', 'B-LANGUAGE', 'S-LANGUAGE']
        self.NUMBER_TO_TAG = dict(zip(self.TAG_TO_NUNBER.values(), self.TAG_TO_NUNBER.keys()))

    def entity_capitalization(self, inputText):

        tokenized_terms = nltk.word_tokenize(inputText.lower())
        checkSentArray = np.array([' '.join(tokenized_terms)])
        checkTagArray = np.array([' '.join(['O'] * len(tokenized_terms))])

        torch_batch_pad_word_ids, torch_batch_pad_char_ids, torch_batch_y, torch_batch_length = sample_a_batch_from_text(
            None, self.Word_Cha_Class, checkSentArray, checkTagArray, False, self.DEVICE)
        paths = self.loaded_model.predict_crf(torch_batch_pad_word_ids, torch_batch_pad_char_ids, torch_batch_length)
        tag_list = list(map(self.NUMBER_TO_TAG.get, paths[0].tolist()))
        text = ' '.join(
            [to.capitalize() if tag in self.need_entity_tag else to for to, tag in zip(tokenized_terms, tag_list)])

        return text
