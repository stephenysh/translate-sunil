import re
import subprocess
from pathlib import Path
from preprocess.moses.moses_tokenizer import MosesTokenizer

current_path = Path(__file__).parent.resolve()

perl_file = str(current_path / 'moses_tokenizer.pl')
pp_file = str(current_path / 'protected_patterns')

lang_convert_dict = {
    'en2ar': 'en',
    'ar2en': 'ar'
    }

def do_moses_use_perl(input_texts:list, lang='en2ar') -> list:
    '''
    do moses preprocess use perl
    :param lang: language type, choose in 'en2ar'->'en', 'ar2en'->'ar'
    :param input_texts: list of input texts
    :return: list of processed text
    '''
    assert lang in lang_convert_dict.keys()
    lang = lang_convert_dict[lang]

    cmd = f'perl {perl_file} -no-escape -l {lang} -protected {pp_file} {len(input_texts)}'
    for text in input_texts:
        cmd += f" '{text}'" # space and text inside quotes

    output = subprocess.run(cmd, capture_output=True, shell=True).stdout # get byte output, don not print output

    output = output.decode('utf-8').split('\n')
    output = [item for item in output if item is not '']

    return output

def do_moses(input_texts:list, lang='en2ar') -> list:
    '''
        do moses preprocess
        :param lang: language type, choose in 'en2ar'->'en', 'ar2en'->'ar'
        :param input_texts: list of input texts
        :return: list of processed text
        '''
    assert lang in lang_convert_dict.keys()
    lang = lang_convert_dict[lang]

    tokenize = MosesTokenizer(lang)

    output = []
    for input in input_texts:
        out = tokenize(input)
        output.append(out)

    tokenize.close()

    return output

def do_moses_single_sentence(single_text, lang='en2ar'):
    '''
    do moses preprocess
    :param lang: language type, choose in 'en2ar'->'en', 'ar2en'->'ar'
    :param single_text: list of input line
    :return: processed line
    '''

    return do_moses([single_text], lang)[0]
