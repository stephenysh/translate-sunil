import re
from collections import deque

from sentence_util import WordType
from sentence_util.regular_filter import RegularFilter
from sentence_util.file_filter import FileFilter


class SingleSentence(object):
    def __init__(self, sentence, type):
        self.sentence = sentence
        self.type = type


class Sentence(object):
    __url_filer = RegularFilter(
        re=re.compile(r"\b(https?://(?:[\w]+)(?:\.[\w\-]+)+)(?::\d*)?(?:/[^/ ]*)*\b", flags=re.IGNORECASE),
        type=WordType.url)
    __email_filter = RegularFilter(
        re=re.compile(r'[\.\w]+@\w+?(?:\.\w+)+', flags=re.IGNORECASE),
        type=WordType.email
    )
    __emoji_filter = RegularFilter(
        re=re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      "]+", flags=re.UNICODE),
        type=WordType.emoji
    )
    __acronymn_filter = FileFilter(file_path='conf/wikipedia-acronyms-simple.json', type=WordType.acronymn)
    __filter_list = [
        __url_filer,
        __email_filter,
        __emoji_filter,
        __acronymn_filter
    ]

    def __init__(self, sentence):
        self.sentence = sentence
        self.word_count = 0
        self.split_sentences = []
        self.special_words = []
        self.__detect_special_char()
        self.__split_sentence()

    def get_sentence_list(self, type=WordType.word):
        sentences = []
        for sentence_obj in self.split_sentences:
            if sentence_obj.type == type:
                sentences.append(sentence_obj.sentence)
        return sentences

    def get_translation(self, translate_list):
        deque_list = deque(translate_list)
        trans_list = []
        for sent_obj in self.split_sentences:
            if sent_obj.type == WordType.word:
                trans_list.append(deque_list.popleft())
            else:
                trans_list.append(sent_obj.sentence)
        return " ".join(trans_list)

    def __detect_special_char(self):
        sp_words = []
        for re_filter in self.__filter_list:
            for m in re.finditer(re_filter.re, self.sentence):
                sp_words.append((m.group(0), m.start(), m.end(), re_filter.type))
        self.special_words = sorted(sp_words, key=lambda z: z[1])

    def __split_sentence(self):
        index = 0
        for sp_word in self.special_words:
            word_part = self.sentence[index:sp_word[1]].strip()
            if len(word_part) > 0:
                self.split_sentences.append(SingleSentence(sentence=word_part, type=WordType.word))
                self.word_count += 1

            sp_word_part = self.sentence[sp_word[1]:sp_word[2]].strip()
            self.split_sentences.append(SingleSentence(sentence=sp_word_part, type=sp_word[3]))
            index = sp_word[2]
        last_part = self.sentence[index:].strip()
        if len(last_part) > 0:
            self.split_sentences.append(SingleSentence(sentence=last_part, type=WordType.word))
            self.word_count += 1


if __name__ == '__main__':
    msg = "Please visit us at http://www.google.com for more information. https://www.aaa.com"
    sent_obj = Sentence(msg)
    print(sent_obj.get_translation(['11 ', ' 22 ']))
