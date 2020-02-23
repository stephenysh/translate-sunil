from model.ner_model import NERModel
from post_processor.post_processor import PostProcessor


class NerProcessor(PostProcessor):
    ner_model = NERModel()

    def process(self, pred, source, model_id):
        if model_id == 'ar2en':
            return self.ner_model.entity_capitalization(pred)
        else:
            return pred
