import re
from keras.preprocessing.text import Tokenizer

# 物語の始まりの改行を置き換える文字数
seq_length = 20

file_name = 'aesop.txt'

with open(file_name, encoding='utf-8-sig') as f:
    text = f.read()

# 物語の始まりの改行の置き換え用
start_story = '| ' * seq_length

""" 前処理 """
text = start_story + text
# 大文字を小文字に変換
text = text.lower()
# 改行の置き換え
text = text.replace('\n\n\n\n\n', start_story)
text = text.replace('\n', ' ')
# 両端の空白文字を消す
text = re.sub('  +', '. ', text).strip()
# ..を.に置き換え
text = text.replace('..', '.')

# 正規表現による置換
text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1', text)
text = re.sub('\s{2,}', ' ', text)

""" トークン化 """
tokenizer = Tokenizer(char_level=False, filters='')
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1
token_list = tokenizer.texts_to_sequences([text])[0]
