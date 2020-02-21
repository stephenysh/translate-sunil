import os
from pathlib import Path

current_path = Path(__file__).parent.resolve()
os.environ['POLYGLOT_DATA_PATH'] = str(current_path)
from polyglot import text

lang_convert_dict = {
    'en2ar': 'en',
    'ar2en': 'ar'
}

def process_single_text(line, lang):
    # Common slow down
    if not line.strip():
        return ''

    seg = text.Text(line.strip())
    seg.language = lang
    t = []

    # Join with '@@' for subwords and ' ' for words
    for word in seg.words:
        for i, morph in enumerate(word.morphemes):
            l = len(word.morphemes)
            if l and i < l - 1 and l > 1:
                t.append(f'{morph}@@')
            else:
                t.append(morph)
    return_str = ' '.join(t)
    if lang == 'en':
        return_str = return_str.lower()
    return return_str

def do_morfessor(text_list: list, lang='en2ar') -> list:
    '''
    Operates on a line
    Uses morfessor tokenization and returns the output using the '@@' subword token separator
    and the ' ' word token separator
    :param line: list of input texts
    :param lang: language type, choose in 'en2ar'->'en', 'ar2en'->'ar'
    :return: list of processed texts
    '''

    assert lang in lang_convert_dict.keys()
    lang = lang_convert_dict[lang]

    return [process_single_text(line, lang) for line in text_list]

def do_morfessor_single_sentence(single_line, lang='en2ar'):
    '''
    Operates on a line
    Uses morfessor tokenization and returns the output using the '@@' subword token separator
    and the ' ' word token separator
    :param line: input line
    :param lang: language type, choose in 'en2ar'->'en', 'ar2en'->'ar'
    :return: processed line
    '''

    return do_morfessor([single_line], lang)[0]

