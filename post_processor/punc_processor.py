from post_processor.post_processor import PostProcessor
import re


def add_slash_to_each_char(string: str) -> str:
    return ''.join([rf'\{item}' for item in list(string)])

class PuncProcessor(PostProcessor):
    puncs_should_not_have_before_space_with_slash = add_slash_to_each_char('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    puncs_should_not_have_after_space_with_slash = add_slash_to_each_char('&*+-/<=>@[\\]^_`{|}~')
    puncs_should_continue_with_big_letter = add_slash_to_each_char('!.?')

    p1 = re.compile(rf'\s+[{puncs_should_not_have_before_space_with_slash}]')
    p2 = re.compile(rf'[{puncs_should_not_have_after_space_with_slash}]\s+')
    p3 = re.compile(rf'[{puncs_should_continue_with_big_letter}] [a-z]')

    def process(self, pred, source, model_id):
        out = pred
        out = re.sub(self.p1, lambda match: match.group().strip(), out)
        out = re.sub(self.p2, lambda match: match.group().strip(), out)
        out = re.sub(self.p3, lambda match: match.group().upper(), out)
        return out
