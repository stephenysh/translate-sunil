from post_processor.post_processor import PostProcessor
import re


class AbbrevProcessor(PostProcessor):
    abbrev = re.compile(r"(\b([a-z]\.){2,})")

    def process(self, pred, source, model_id):
        return re.sub(self.abbrev, self.__to_upper, pred)

    def __to_upper(self, m):
        word = m.group(0)
        return word.upper()