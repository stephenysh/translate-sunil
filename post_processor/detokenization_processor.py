import re

from post_processor.post_processor import PostProcessor


class DetokenizationProcessor(PostProcessor):

    def process(self, pred, source, model_id):
        return re.sub(r'(@@ )|(@@ ?$)', '', pred.strip())
