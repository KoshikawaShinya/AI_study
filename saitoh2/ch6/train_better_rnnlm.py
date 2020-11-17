import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from better_rnnlm import BetterRnnlm

# ハイパーパラメータの設定
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25
dropout_rate = 0.5

# 学習データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_val, _, _ = ptb.load_data('val')
corpus_test, _, _ = ptb.load_data('test')

vocab_size = len(word_to_id)
# 入力では最後の一つ以外、教師ラベルでは最初の一つ以外とする
xs = corpus[:-1]
ts = corpus[1:]

# モデルの生成
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout_rate)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

best_ppl = float('inf')
for epoch in range(max_epoch):
    # 勾配クリッピングを適用して学習
    # eval_interval : eval_intervalイテレーションおきにperplexityを評価する
    trainer.fit(xs, ts, max_epoch=1, batch_size=batch_size, time_size=time_size, max_grad=max_grad, eval_interval=20)
    
    model.reset_state()
    ppl = eval_perplexity(model, corpus_val)
    print('valid perplexity : ', ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()
    print('-' * 50)