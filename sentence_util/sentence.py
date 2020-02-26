import re
from collections import deque

from nltk.tokenize import sent_tokenize

from sentence_util import WordType
from sentence_util.file_filter import FileFilter
from sentence_util.regular_filter import RegularFilter


class SingleSentence(object):
    def __init__(self, sentence, type):
        self.sentence = sentence
        self.type = type


class Sentence(object):
    __url_filer = RegularFilter(
        re=re.compile(r"\b(https?://(?:[\w]+)(?:\.[\w\-]+)+)(?::\d*)?(?:/[^/ ]*)*(\?[^ \?]*)?\b", flags=re.IGNORECASE),
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
    __seperator_filter = RegularFilter(
        re=re.compile(r"[\f\r\n]+"),
        type=WordType.seperator
    )
    __acronymn_filter = FileFilter(file_path='conf/wikipedia-acronyms-simple.json', type=WordType.acronymn)
    __filter_list = [
        __url_filer,
        __email_filter,
        __emoji_filter,
        __acronymn_filter,
        __seperator_filter
    ]

    MAX_WORD = 30

    def __init__(self, sentence, is_split):
        self.sentence = sentence
        self.is_split = is_split

        self.word_count = 0
        self.split_sentences = []
        self.tokenized_list = []
        self.special_words = []
        self.__detect_special_char()
        self.__split_sentence()

    def get_sentence_list(self, type=WordType.word):
        sentences = []
        for sentence_obj in self.split_sentences:
            if sentence_obj.type == type:
                sentences.append(sentence_obj.sentence)
        return sentences

    def __first_letter_cap(self, pred):
        line = pred.strip()
        if len(line) == 0:
            return line
        return line[0].upper() + line[1:]

    def get_translation(self, translate_list):
        deque_list = deque(translate_list)
        trans_list = []
        for sent_obj in self.split_sentences:
            if sent_obj.type == WordType.word:
                trans_list.append(deque_list.popleft())
            else:
                trans_list.append(sent_obj.sentence)
        return self.__first_letter_cap(" ".join(trans_list))

    def __detect_special_char(self):
        sp_words = []
        for re_filter in self.__filter_list:
            for m in re.finditer(re_filter.re, self.sentence):
                sp_words.append((m.group(0), m.start(), m.end(), re_filter.type))
        self.special_words = sorted(sp_words, key=lambda z: z[1])
        self.__merge_overlap()

    def __merge_overlap(self):
        temp_result = []
        i = 0
        while i < len(self.special_words):
            word_a = self.special_words[i]
            if i == len(self.special_words) - 1:
                temp_result.append(word_a)
            flag_add_to_list = False
            for j in range(i + 1, len(self.special_words)):
                word_b = self.special_words[j]
                if word_b[1] <= word_a[2] < word_b[2]:
                    if not flag_add_to_list:
                        flag_add_to_list = True
                        temp_result.append((self.sentence[word_a[1]:word_b[2]], word_a[1], word_b[2], WordType.mix))
                    i += 1
                elif word_a[2] >= word_b[2]:
                    if not flag_add_to_list:
                        flag_add_to_list = True
                        temp_result.append(word_a)
                    i += 1
                else:
                    if not flag_add_to_list:
                        temp_result.append(word_a)
                    break
            i += 1
        self.special_words = sorted(temp_result, key=lambda z: z[1])

    def __split_sentence(self):
        index = 0
        for sp_word in self.special_words:
            word_part = self.sentence[index:sp_word[1]].strip()
            if len(word_part) > 0:
                self.__handle_is_split(word_part)

            sp_word_part = sp_word[0]
            self.split_sentences.append(SingleSentence(sentence=sp_word_part, type=sp_word[3]))
            index = sp_word[2]
        last_part = self.sentence[index:].strip()
        if len(last_part) > 0:
            self.__handle_is_split(last_part)

    def __handle_is_split(self, word_part):
        if not self.is_split:
            self.__add_split_sent(word_part)
        else:
            if len(word_part.split(" ")) <= self.MAX_WORD:
                self.__add_split_sent(word_part)
            else:
                sent_list = sent_tokenize(word_part)
                for sent in sent_list:
                    self.__handle_sent_over_len(sent)

    def __handle_sent_over_len(self, sent):
        if len(sent.split(" ")) > self.MAX_WORD:
            word_list = sent.split(" ")
            index = 0
            while len(word_list[index:]) > self.MAX_WORD:
                self.__add_split_sent(" ".join(word_list[index:index + self.MAX_WORD]))
                index += self.MAX_WORD
            self.__add_split_sent(" ".join(word_list[index:]))
        else:
            self.__add_split_sent(sent)

    def __add_split_sent(self, sent):
        self.split_sentences.append(SingleSentence(sentence=sent, type=WordType.word))
        self.tokenized_list.append(sent)
        self.word_count += 1


if __name__ == '__main__':
    msg = "Please visit us at http://www.google.com for more information. https://www.aaa.com"
    sent_obj = Sentence(msg)
    print(sent_obj.get_translation(['11 ', ' 22 ']))
