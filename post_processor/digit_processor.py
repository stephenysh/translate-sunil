import re
from collections import namedtuple

from post_processor.post_processor import PostProcessor


class DigitProcessor(PostProcessor):
    convert_to_ar = False

    ar_nums_file = 'data/ar_nums.txt'
    with open(ar_nums_file, 'r') as f:
        ar_txt = f.readlines()
    ar_dict = {key.strip(): index for index, key in enumerate(ar_txt)}
    group = namedtuple('Group', 'pos str type')
    zeros = re.compile(r'0+')
    eastern_to_western = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8",
                          "٩": "9"}
    en_digits_re = re.compile(r'([0-9])+')
    ar_digits_re = re.compile(r'\b(?:%s)+\b' % '|'.join(list(eastern_to_western.keys())))
    ar_txt_re = re.compile(r'\b(?:%s)\b' % '|'.join(list(ar_dict.keys())))
    zeros = re.compile(r'0+')

    def process(self, pred, source, model_id):
        replace_pred = pred
        pred_zeros = re.findall(self.zeros, replace_pred)
        if model_id == 'en2ar':
            found_digits = self.__find_en_digits(source)
        elif model_id == 'ar2en':
            found_digits = self.__find_en_digits(source) + self.__find_ar_digits(source) + self.__find_ar_text_digits(
                source)

        found_digits.sort(key=lambda x: x.pos)  # In place

        # The zeros dont match the found digits
        if len(pred_zeros) != len(found_digits):
            return pred

        # Todo, look for ways to improve swapped groups
        found_digits_lens = [len(dig.str) for dig in found_digits]
        zeros_lens = [len(zer) for zer in pred_zeros]
        if found_digits_lens != zeros_lens:
            for i in range(len(zeros_lens)):
                if found_digits_lens[i] == zeros_lens[i]:
                    continue
                # min is so we dont access outside the range
                # condition is checking if the swap is viable
                elif found_digits_lens[min(i + 1, len(zeros_lens) - 1)] == zeros_lens[i] and \
                        found_digits_lens[i] == zeros_lens[min(i + 1, len(zeros_lens) - 1)]:
                    print('swap groups')
                    found_digits_lens[i], found_digits_lens[i + 1] = found_digits_lens[i + 1], found_digits_lens[i]
                    found_digits[i], found_digits[i + 1] = found_digits[i + 1], found_digits[i]

        for group_num, (dig, zer) in enumerate(zip(found_digits, pred_zeros)):
            # We expect the number of digits to match except in the case of arabic text digits
            # if len(dig.str) != len(zer) and dig.type != 'ar_text_digits':
            # if line_num:
            #     tqdm.write(f'Error for match line {line_num} group {group_num}: {dig} {zer}')
            # else:
            #     tqdm.write(f'Error for group {group_num}: {dig} {zer}')

            # Since these are replaced in order, we match the first occurence of zeros found in the target
            replace_pred = replace_pred.replace(str(zer), str(dig.str), 1)

        return replace_pred

    def __find_en_digits(self, src):
        en_digits = []
        for m in re.finditer(self.en_digits_re, src):
            found_digits = m.group()
            if self.convert_to_ar:
                found_digits = [self.eastern_to_western[dig] for dig in found_digits]
            en_digits.append(self.group(m.start(), found_digits, 'en_digits'))
        return en_digits

    def __find_ar_digits(self, src):
        ar_digits = []
        for m in re.finditer(self.ar_digits_re, src):
            en_converted_from_ar = ''.join([self.eastern_to_western[digit] for digit in m.group()])
            ar_digits.append(self.group(m.start(), en_converted_from_ar, 'ar_digits'))
        return ar_digits

    def __find_ar_text_digits(self, src):
        ar_txt_digits = []
        for m in re.finditer(self.ar_txt_re, src):
            ar_txt_digits.append(self.group(m.start(), m.group(), 'ar_text_digits'))
        return ar_txt_digits
