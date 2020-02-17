import codecs
import os
import re
import sys
import threading
import traceback
from collections import deque

import onmt.opts
import torch
from onmt.translate.translator import build_translator
from onmt.utils.alignment import to_word_align
from onmt.utils.logging import init_logger
from onmt.utils.misc import set_random_seed
from onmt.utils.parse import ArgumentParser

from sentence_util.sentence import Sentence
from util import Timer


class ServerModelError(Exception):
    pass


def critical(func):
    """Decorator for critical section (mutually exclusive code)"""

    def wrapper(server_model, *args, **kwargs):
        if sys.version_info[0] == 3:
            if not server_model.running_lock.acquire(True, 120):
                raise ServerModelError("Model %s running lock timeout"
                                       % server_model.model_id)
        else:
            # semaphore doesn't have a timeout arg in Python 2.7
            server_model.running_lock.acquire(True)
        try:
            o = func(server_model, *args, **kwargs)
        except Exception as e:
            raise
        finally:
            server_model.running_lock.release()
        return o

    return wrapper


class ServerModel(object):
    """Wrap a model with server functionality.
    Args:
        opt (dict): Options for the Translator
        model_id (str): Model ID
        preprocess_opt (list): Options for preprocess processus or None
                               (extend for CJK)
        tokenizer_opt (dict): Options for the tokenizer or None
        postprocess_opt (list): Options for postprocess processus or None
                                (extend for CJK)
        load (bool): whether to load the model during :func:`__init__()`
        timeout (int): Seconds before running :func:`do_timeout()`
            Negative values means no timeout
        on_timeout (str): Options are ["to_cpu", "unload"]. Set what to do on
            timeout (see :func:`do_timeout()`.)
        model_root (str): Path to the model directory
            it must contain the model and tokenizer file
    """
    abbrev = re.compile(r"(\b([a-z]\.){2,})")

    eastern_to_western = {"٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6", "٧": "7", "٨": "8",
                          "٩": "9"}

    en_digits = re.compile(r'[0-9]+')
    ar_digits = re.compile(r'\b(?:%s)+\b' % '|'.join(list(eastern_to_western.keys())))
    zeros = re.compile(r'0+')

    def digit_mapping(self, tgt_line, src_line):
        src_digits = []
        for m in re.finditer(self.en_digits, src_line):
            src_digits.append((m.group(0), m.start(), m.end(), 'en'))
        for m in re.finditer(self.ar_digits, src_line):
            src_digits.append((m.group(0), m.start(), m.end(), 'ar'))

        # sort by start position
        src_digits = sorted(src_digits, key=lambda z: z[1])
        for (i, src) in enumerate(src_digits):
            if src[3] == 'ar':
                en_number = ''.join(self.eastern_to_western[k] for k in src[0])
                src_digits[i] = (en_number, src[1], src[2], src[3])

        tgt_zeros = []
        for m in re.finditer(self.zeros, tgt_line):
            tgt_zeros.append((m.group(0), m.start(), m.end()))

        if len(src_digits) != len(tgt_zeros):
            self.logger.info('cannot replace, mismatch in digit occurrences')
            return tgt_line

        for src, tgt in zip(src_digits, tgt_zeros):
            tgt_line = tgt_line.replace(tgt[0], src[0], 1)
        if len(src_digits) > 0:
            self.logger.info('digits correctly replaced')
        return tgt_line

    def trans_to_object(self, message):
        return Sentence(message)

    def encode(self, sent_obj, model_id):
        sent_list = sent_obj.get_sentence_list()
        tokenized_list = []
        for sent in sent_list:
            tokenized_list.append(' '.join(self.encoders[model_id].encode(sent)))
        sent_obj.tokenized_list = tokenized_list
        return sent_obj

    def encode_en2ar(self, sent_obj):
        return self.encode(sent_obj, 'en2ar')

    def encode_ar2en(self, sent_obj):
        return self.encode(sent_obj, 'ar2en')

    def decode_en2ar(self, message):
        return ''.join(self.decoders['en2ar'].decode(message.split()))

    def decode_ar2en(self, message):
        return ''.join(self.decoders['ar2en'].decode(message.split()))

    def first_letter_capitalize(self, message):
        message = message.strip()
        if len(message) == 0:
            return message
        return message[0].upper() + message[1:]

    def to_upper(self, m):
        word = m.group(0)
        return word.upper()

    def abbreviation_capitalize(self, line):
        return re.sub(self.abbrev, self.to_upper, line)

    def __init__(self, opt, model_id, preprocess_opt=None, tokenizer_opt=None,
                 postprocess_opt=None, load=False, timeout=-1,
                 on_timeout="to_cpu", model_root="./"):
        self.model_root = model_root
        self.opt = self.parse_opt(opt)

        self.model_id = model_id
        self.preprocess_opt = preprocess_opt
        self.tokenizer_opt = tokenizer_opt
        self.postprocess_opt = postprocess_opt
        self.timeout = timeout
        self.on_timeout = on_timeout

        self.unload_timer = None
        self.user_opt = opt
        self.tokenizer = None

        if len(self.opt.log_file) > 0:
            log_file = os.path.join(model_root, self.opt.log_file)
        else:
            log_file = None
        self.logger = init_logger(log_file=log_file,
                                  log_file_level=self.opt.log_file_level)

        self.loading_lock = threading.Event()
        self.loading_lock.set()
        self.running_lock = threading.Semaphore(value=1)

        from bpemb import BPEmb
        arbp = BPEmb(lang="ar", dim=50, vs=50000, cache_dir="bpemb_cache")
        enbp = BPEmb(lang="en", dim=50, vs=50000, cache_dir="bpemb_cache")
        self.encoders = {"en2ar": enbp, "ar2en": arbp}
        self.decoders = {"en2ar": arbp, "ar2en": enbp}
        self.encode_fn = {"en2ar": self.encode_en2ar, "ar2en": self.encode_ar2en}
        self.decode_fn = {"en2ar": self.decode_en2ar, "ar2en": self.decode_ar2en}

        set_random_seed(self.opt.seed, self.opt.cuda)

        self.logger.info("Loading preprocessors and post processors")
        self.preprocessor = [self.trans_to_object, self.encode_fn[model_id]]
        self.postprocessor = [
            {'func': self.decode_fn[model_id], 'need_source': False},
            {'func': self.digit_mapping, 'need_source': True},
            {'func': self.first_letter_capitalize, 'need_source': False},
            {'func': self.abbreviation_capitalize, 'need_source': False}
        ]

        if load:
            self.load()

    def parse_opt(self, opt):
        """Parse the option set passed by the user using `onmt.opts`

       Args:
           opt (dict): Options passed by the user

       Returns:
           opt (argparse.Namespace): full set of options for the Translator
        """

        prec_argv = sys.argv
        sys.argv = sys.argv[:1]
        parser = ArgumentParser()
        onmt.opts.translate_opts(parser)

        models = opt['models']
        if not isinstance(models, (list, tuple)):
            models = [models]
        opt['models'] = [os.path.join(self.model_root, model)
                         for model in models]
        opt['src'] = "dummy_src"

        for (k, v) in opt.items():
            if k == 'models':
                sys.argv += ['-model']
                sys.argv += [str(model) for model in v]
            elif type(v) == bool:  # only true bool should be parsed
                if v is True:
                    sys.argv += ['-%s' % k]
            else:
                sys.argv += ['-%s' % k, str(v)]

        opt = parser.parse_args()
        opt.alignment_heads = 8
        ArgumentParser.validate_translate_opts(opt)
        opt.cuda = opt.gpu > -1

        sys.argv = prec_argv
        return opt

    @property
    def loaded(self):
        return hasattr(self, 'translator')

    def load(self):
        self.loading_lock.clear()

        timer = Timer()
        self.logger.info("Loading model %s" % self.model_id)
        timer.start()

        try:
            # opt = DefaultOpt(self.user_opt['models'], 'src-test.txt', 'temp.txt') # should read model paths from json, not fixed
            opt = self.opt
            self.translator = build_translator(opt,
                                               report_score=False,
                                               out_file=codecs.open(
                                                   os.devnull, "w", "utf-8"))

        except RuntimeError as e:
            raise ServerModelError("Runtime Error: %s" % str(e))

        timer.tick("model_loading")
        self.load_time = timer.tick()
        self.reset_unload_timer()
        self.loading_lock.set()

    @critical
    def run(self, inputs):
        """Translate `inputs` using this model

        Args:
            inputs (List[dict[str, str]]): [{"src": "..."},{"src": ...}]

        Returns:
            result (list): translations
            times (dict): containing times
        """

        self.stop_unload_timer()

        timer = Timer()
        timer.start()

        self.logger.info("Running translation using %s" % self.model_id)

        if not self.loading_lock.is_set():
            self.logger.info(
                "Model #%s is being loaded by another thread, waiting"
                % self.model_id)
            if not self.loading_lock.wait(timeout=30):
                raise ServerModelError("Model %s loading timeout"
                                       % self.model_id)

        else:
            if not self.loaded:
                self.load()
                timer.tick(name="load")
            elif self.opt.cuda:
                self.to_gpu()
                timer.tick(name="to_gpu")

        texts = []
        head_spaces = []
        tail_spaces = []
        source_text = []
        sentence_objs = []
        for i, inp in enumerate(inputs):
            src = inp['src']
            if src.strip() == "":
                head_spaces.append(src)
                texts.append("")
                tail_spaces.append("")
            else:
                whitespaces_before, whitespaces_after = "", ""
                match_before = re.search(r'^\s+', src)
                match_after = re.search(r'\s+$', src)
                if match_before is not None:
                    whitespaces_before = match_before.group(0)
                if match_after is not None:
                    whitespaces_after = match_after.group(0)
                head_spaces.append(whitespaces_before)
                sent_obj = self.maybe_preprocess(src.strip())
                sentence_objs.append(sent_obj)
                source_text.append(src.strip())
                tok = self.maybe_tokenize(sent_obj.tokenized_list)
                texts.extend(tok)
                tail_spaces.append(whitespaces_after)

        empty_indices = [i for i, x in enumerate(texts) if x == ""]
        texts_to_translate = [x for x in texts if x != ""]
        source_lines = [x for x in source_text if x != ""]

        scores = []
        predictions = []
        if len(texts_to_translate) > 0:
            try:
                # scores, predictions = self.translator.translate(
                #     texts_to_translate,
                #     batch_size=len(texts_to_translate)
                #     if self.opt.batch_size == 0
                #     else self.opt.batch_size)
                scores, predictions = self.translator.translate(texts_to_translate, None, '', 1, 'sent', False, False)
            except (RuntimeError, Exception) as e:
                err = "Error: %s" % str(e)
                self.logger.error(err)
                self.logger.error("repr(text_to_translate): "
                                  + repr(texts_to_translate))
                self.logger.error("model: #%s" % self.model_id)
                self.logger.error("model opt: " + str(self.opt.__dict__))
                self.logger.error(traceback.format_exc())

                raise ServerModelError(err)

        timer.tick(name="translation")
        self.logger.info("""Using model [%s], input num [%d], translation time: [%f]""" % (
            self.model_id, len(texts), timer.times['translation']))
        self.reset_unload_timer()

        # NOTE: translator returns lists of `n_best` list
        def flatten_list(_list):
            return sum(_list, [])

        # tiled_texts = [t for t in texts_to_translate
        #               for _ in range(self.opt.n_best)]
        results = flatten_list(predictions)
        scores = [score_tensor.item()
                  for score_tensor in flatten_list(scores)]

        # results = [self.maybe_detokenize_with_align(result, src)
        #           for result, src in zip(results, tiled_texts)]

        # aligns = [align for _, align in results]
        final_result = self.__get_final_result(results, sentence_objs)
        final_result = [self.maybe_postprocess(target, source) for target, source in zip(final_result, source_lines)]

        # build back results with empty texts
        for i in empty_indices:
            j = i * self.opt.n_best
            results = results[:j] + [""] * self.opt.n_best + results[j:]
            aligns = aligns[:j] + [None] * self.opt.n_best + aligns[j:]
            scores = scores[:j] + [0] * self.opt.n_best + scores[j:]

        head_spaces = [h for h in head_spaces for i in range(self.opt.n_best)]
        tail_spaces = [h for h in tail_spaces for i in range(self.opt.n_best)]
        final_result = ["".join(items)
                        for items in zip(head_spaces, final_result, tail_spaces)]

        self.logger.info("Translation Results: %d", len(final_result))
        return final_result

    def __get_final_result(self, results, sentence_objs):
        final_result = []
        deque_result = deque(results)
        for sent_obj in sentence_objs:
            trans_list = []
            for i in range(sent_obj.word_count):
                trans_list.append(deque_result.popleft())
            final_result.append(sent_obj.get_translation(trans_list))
        return final_result

    def do_timeout(self):
        """Timeout function that frees GPU memory.

        Moves the model to CPU or unloads it; depending on
        attr`self.on_timemout` value
        """

        if self.on_timeout == "unload":
            self.logger.info("Timeout: unloading model %s" % self.model_id)
            self.unload()
        if self.on_timeout == "to_cpu":
            self.logger.info("Timeout: sending model %s to CPU"
                             % self.model_id)
            self.to_cpu()

    @critical
    def unload(self):
        self.logger.info("Unloading model %s" % self.model_id)
        del self.translator
        if self.opt.cuda:
            torch.cuda.empty_cache()
        self.stop_unload_timer()
        self.unload_timer = None

    def stop_unload_timer(self):
        if self.unload_timer is not None:
            self.unload_timer.cancel()

    def reset_unload_timer(self):
        if self.timeout < 0:
            return

        self.stop_unload_timer()
        self.unload_timer = threading.Timer(self.timeout, self.do_timeout)
        self.unload_timer.start()

    def to_dict(self):
        hide_opt = ["models", "src"]
        d = {"model_id": self.model_id,
             "opt": {k: self.user_opt[k] for k in self.user_opt.keys()
                     if k not in hide_opt},
             "models": self.user_opt["models"],
             "loaded": self.loaded,
             "timeout": self.timeout,
             }
        if self.tokenizer_opt is not None:
            d["tokenizer"] = self.tokenizer_opt
        return d

    @critical
    def to_cpu(self):
        """Move the model to CPU and clear CUDA cache."""
        self.translator.model.cpu()
        if self.opt.cuda:
            torch.cuda.empty_cache()

    def to_gpu(self):
        """Move the model to GPU."""
        torch.cuda.set_device(self.opt.gpu)
        self.translator.model.cuda()

    def maybe_preprocess(self, sequence):
        """Preprocess the sequence (or not)

        """
        return self.preprocess(sequence)

    def preprocess(self, sequence):
        """Preprocess a single sequence.

        Args:
            sequence (str): The sequence to preprocess.

        Returns:
            sequence (str): The preprocessed sequence.
        """
        if self.preprocessor is None:
            raise ValueError("No preprocessor loaded")
        for function in self.preprocessor:
            sequence = function(sequence)
        return sequence

    def maybe_tokenize(self, sequence):
        """Tokenize the sequence (or not).

        Same args/returns as `tokenize`
        """

        if self.tokenizer_opt is not None:
            return self.tokenize(sequence)
        return sequence

    def tokenize(self, sequence):
        """Tokenize a single sequence.

        Args:
            sequence (str): The sequence to tokenize.

        Returns:
            tok (str): The tokenized sequence.
        """

        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded")

        if self.tokenizer_opt["type"] == "sentencepiece":
            tok = self.tokenizer.EncodeAsPieces(sequence)
            tok = " ".join(tok)
        elif self.tokenizer_opt["type"] == "pyonmttok":
            tok, _ = self.tokenizer.tokenize(sequence)
            tok = " ".join(tok)
        return tok

    @property
    def tokenizer_marker(self):
        marker = None
        tokenizer_type = self.tokenizer_opt.get('type', None)
        if tokenizer_type == "pyonmttok":
            params = self.tokenizer_opt.get('params', None)
            if params is not None:
                if params.get("joiner_annotate", None) is not None:
                    marker = 'joiner'
                elif params.get("spacer_annotate", None) is not None:
                    marker = 'spacer'
        elif tokenizer_type == "sentencepiece":
            marker = 'spacer'
        return marker

    def maybe_detokenize_with_align(self, sequence, src):
        """De-tokenize (or not) the sequence (with alignment).

        Args:
            sequence (str): The sequence to detokenize, possible with
                alignment seperate by ` ||| `.

        Returns:
            sequence (str): The detokenized sequence.
            align (str): The alignment correspand to detokenized src/tgt
                sorted or None if no alignment in output.
        """
        align = None
        if self.opt.report_align:
            # output contain alignment
            sequence, align = sequence.split(' ||| ')
            align = self.maybe_convert_align(src, sequence, align)
        sequence = self.maybe_detokenize(sequence)
        return (sequence, align)

    def maybe_detokenize(self, sequence):
        """De-tokenize the sequence (or not)

        Same args/returns as :func:`tokenize()`
        """

        if self.tokenizer_opt is not None and ''.join(sequence.split()) != '':
            return self.detokenize(sequence)
        return sequence

    def detokenize(self, sequence):
        """Detokenize a single sequence

        Same args/returns as :func:`tokenize()`
        """

        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded")

        if self.tokenizer_opt["type"] == "sentencepiece":
            detok = self.tokenizer.DecodePieces(sequence.split())
        elif self.tokenizer_opt["type"] == "pyonmttok":
            detok = self.tokenizer.detokenize(sequence.split())

        return detok

    def maybe_convert_align(self, src, tgt, align):
        """Convert alignment to match detokenized src/tgt (or not).

        Args:
            src (str): The tokenized source sequence.
            tgt (str): The tokenized target sequence.
            align (str): The alignment correspand to src/tgt pair.

        Returns:
            align (str): The alignment correspand to detokenized src/tgt.
        """
        if self.tokenizer_marker is not None and ''.join(tgt.split()) != '':
            return to_word_align(src, tgt, align, mode=self.tokenizer_marker)
        return align

    def maybe_postprocess(self, target, source):
        """Postprocess the sequence (or not)

        """
        return self.postprocess(target, source)

    def postprocess(self, target, source):
        """Preprocess a single sequence.

        Args:
            sequence (str): The sequence to process.

        Returns:
            sequence (str): The postprocessed sequence.
        """
        if self.postprocessor is None:
            raise ValueError("No postprocessor loaded")
        for processor in self.postprocessor:
            if processor['need_source']:
                target = processor['func'](target, source)
            else:
                target = processor['func'](target)
        return target
