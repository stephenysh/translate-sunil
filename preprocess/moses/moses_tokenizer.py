from os import path
from preprocess.moses.toolwrapper import ToolWrapper
from mosestokenizer import MosesTokenizer as MT

class MosesTokenizer(ToolWrapper):
    """A module for interfacing with ``tokenizer.perl`` from Moses.

    This class communicates with tokenizer.perl process via pipes. When the
    MosesTokenizer object is no longer needed, the close() method should be
    called to free system resources. The class supports the context manager
    interface. If used in a with statement, the close() method is invoked
    automatically.

    >>> tokenize = MosesTokenizer('en')
    >>> tokenize('Hello World!')
    ['Hello', 'World', '!']
    """

    def __init__(self, lang="en", old_version=False):
        self.lang = lang
        program = path.join(
            path.dirname(__file__),
            "moses_tokenizer.pl"
            # "tokenizer-v1.0.perl"
        )
        protected = path.join(
            path.dirname(__file__),
            "protected_patterns"
        )

        argv = ["perl", program, "-q", "-no-escape", "-l", self.lang, "-protected", protected]
        if not old_version:
        #     # -b = disable output buffering
        #     # -a = aggressive hyphen splitting
            argv.extend(["-b"])
        super().__init__(argv)

    def __str__(self):
        return "MosesTokenizer(lang=\"{lang}\")".format(lang=self.lang)

    def __call__(self, sentence):
        """Tokenizes a single sentence.

        Newline characters are not allowed in the sentence to be tokenized.
        """
        assert isinstance(sentence, str)
        sentence = sentence.rstrip("\n")
        assert "\n" not in sentence
        if not sentence:
            return []
        self.writeline(sentence)
        # return self.readline()


if __name__ == "__main__":
    tokenize = MosesTokenizer('en')
    print(tokenize("i would like to lead the scientific part of this program 2-automated translation: right now, we have internally the English-arabic language is already published."))
    tokenize.close()