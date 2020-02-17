from sentence_util.regular_filter import RegularFilter
import re


class FileFilter(RegularFilter):
    def __init__(self, file_path, type):
        self.file_path = file_path
        self.type = type
        self.re = None

        self.__load_file()

    def __load_file(self):
        words = []
        with open(self.file_path, 'r') as f:
            while True:
                text = f.readline()
                if text == '':
                    break
                words.append(text.strip())
        words = set(words)
        self.re = re.compile('|'.join([rf'\b{word}\b' for word in words]))
