
def process_on_sentence_obj(sent_obj, func):
    '''
    add tokenized_list attribute in sentence object
    :param func: func that takes a list of str as input
    :return:
    '''
    sent_list = sent_obj.get_sentence_list()

    sent_obj.tokenized_list = func(sent_list)

    return sent_obj