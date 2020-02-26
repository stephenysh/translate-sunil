import re
import subprocess
from pathlib import Path
from preprocess.moses.moses_tokenizer import MosesTokenizer
import threading
import concurrent.futures

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

def event_thread(func, stop_event):
    result = func()
    stop_event.set()
    return result

def do_moses(input_texts:list, lang='en2ar') -> list:
    '''
        do moses preprocess
        use toolwrapper to get stdout and stderr
        but one of the return maybe hangup
        so need to use two thread with event to wait until at least one result has got returned
        also need to get return value from thread, so use concurrent.future
        note that because we dont want to wait for all thread finished, so should call future.shutdown(wait=False)

        :param lang: language type, choose in 'en2ar'->'en', 'ar2en'->'ar'
        :param input_texts: list of input texts
        :return: list of processed text
        '''
    assert lang in lang_convert_dict.keys()
    lang = lang_convert_dict[lang]

    tokenize = MosesTokenizer(lang)

    output = []
    for input in input_texts:
        tokenize(input)

        out_stop_event = threading.Event()
        err_stop_event = threading.Event()
        # spawn the threads
        executor = concurrent.futures.ThreadPoolExecutor()

        future_out = executor.submit(event_thread, tokenize.read_stdout, out_stop_event)
        future_err = executor.submit(event_thread, tokenize.read_stderr, err_stop_event)

        while not out_stop_event.is_set() and not err_stop_event.is_set():
            pass

        out = ''
        err = ''
        if out_stop_event.is_set():
            out = future_out.result()

        if err_stop_event.is_set():
            err = future_err.result()

        executor.shutdown(wait=False)

        if out == '' and err != '':
            raise RuntimeError(err)
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
