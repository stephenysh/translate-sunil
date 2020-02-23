from functools import partial

from preprocess import process_on_sentence_obj
from preprocess.morfessor import do_morfessor
from preprocess.pre_processor import PreProcessor


class MorfessorProcessor(PreProcessor):
    def process(self, obj, is_split, model_id):
        return process_on_sentence_obj(obj, func=(partial(do_morfessor, lang=model_id)))
