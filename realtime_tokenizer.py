import sentencepiece as spm
#import tokenizer.model as tokenizer

def construct_tokenid_list(query):
    sp = spm.SentencePieceProcessor()
    sp.load('tokenizer.model')
    return sp.encode_as_ids(query)
