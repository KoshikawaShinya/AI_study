import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch6.LSTMlm import Rnnlm
from ch6.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):

    # start_id : 文章を開始する単語  skip_ids : スキップする単語IDのリスト 例:[12, 20]  sample_size : 生成する文の長さ
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        
        while len(word_ids) < sample_size:
            # self.predictはミニバッチ処理を想定している。そのため、単語IDを一つだけ入力する場合バッチサイズを1と考えて、1×1のnumpy配列にする
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)

            # skip_idsが指定されない時か、sampledがskip_idsにない時
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
            
        # 生成された文の単語IDの配列を返す
        return word_ids


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)