from functools import partial

from preprocess import process_on_sentence_obj
from preprocess.moses import do_moses
from preprocess.pre_processor import PreProcessor


class MosesProcessor(PreProcessor):
    def process(self, obj, is_split, model_id):
        if model_id == 'en2ar':
            return process_on_sentence_obj(obj, func=(partial(do_moses, lang=model_id)))
        else:
            return obj
