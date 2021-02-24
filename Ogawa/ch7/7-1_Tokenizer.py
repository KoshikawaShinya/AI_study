from janome.tokenizer import Tokenizer
import MeCab

def tokenizer_mecab(text):
    text = m_t.parse(text)
    return text.strip().split()

def tokenizer_janome(text):
    return [tok for tok in j_t.tokenize(text, wakati=True)]

j_t = Tokenizer()
m_t = MeCab.Tagger('-Owakati')

text = '機械学習が好きです。'
print(tokenizer_janome(text))
print(tokenizer_mecab(text))