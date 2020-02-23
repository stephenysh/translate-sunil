from preprocess.pre_processor import PreProcessor
from sentence_util.sentence import Sentence


class SentObjProcessor(PreProcessor):
    def process(self, obj, is_split, model_id):
        return Sentence(obj, is_split)
