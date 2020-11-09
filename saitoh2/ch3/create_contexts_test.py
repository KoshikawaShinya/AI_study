import sys
sys.path.append('..')
from common.util import preprocess, convert_one_hot
from create_context_target import create_context_target

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)
print(id_to_word)

# 学習に使うコンテキストとターゲット
contexts, target = create_context_target(corpus, window_size=1)
print(contexts)
print(target)

# one-hotベクトル化
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print(contexts)
print(target)
