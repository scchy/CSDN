# Python3
# Create date: 2021-10-05
# Author: Scc_hy
# Func: pytorch 单词纠错
# reference: https://www.kaggle.com/rahuldshetty/word-spelling-correction-with-lstm/
# my kaggle: https://www.kaggle.com/scchuy/nlp-tutorial-word-spelling-correction
# =================================================================================================


__doc__ = """
给定一个单词，我们的任务是选择和它最相似的拼写正确的单词。
单词纠错可以使用字符贝叶斯概率完成，本文尝试使用LSTM完成英文单词纠错模型训练

"""

import pandas as pd
import numpy as np
import random
import torch as t
t.__version__
import tensorflow as tf
tf.__version__
import re


# 1- 训练集生成
class GeneratData:
    def __init__(self, test_size=0.3, embedding_func=None):
        self.test_size = test_size
        self.embedding_func = embedding_func
        self.__char_int_map()
    
    def __char_int_map(self):
        self.char_set = [chr(i) for i in range(ord('a'), ord('z')+1)] + '0 1 2 3 4 5 6 7 8 9'.split() + ['\t', '\n', '#']
        self.char2int = dict(zip(self.char_set, range(len(self.char_set))))
        self.int2char = dict(zip(range(len(self.char_set)), self.char_set))

    def _load_data(self):
        file_path = 'D:/Python_data/my_github/CSDN/data/unigram_freq.csv'
        orignal_df = pd.read_csv(file_path)
        orignal_df['word'] = orignal_df['word'].apply(self._word_process)
        # 只拿取str的
        orignal_df['need'] = orignal_df['word'].map(lambda x: type(x) == type('a'))
        return orignal_df.loc[orignal_df['need'], 'word'].tolist()

    def _word_process(self, word):
        """
        将数据清洗干净
        """
        try:
            word = word.lower()
            word = re.sub(r'[^0-9a-zA-Z]', '', word)
            return word
        except Exception as e:
            return word
    
    def generate(self, thresh=0.2):
        """
        训练数据
        1- 生成 input1 input2
        2- embedding
        """
        lines = self._load_data()
        test_size = int(len(lines) * self.test_size)
        input1_list, input2_list = [], []
        input1_max_len, input2_max_len = 0, 0
        for w in lines[:-test_size]:
            if len(w) > 10:
                input2_word = f'\t{w}\n'
                input1_word = self.gen_gibberish(w, thresh=thresh)
                input1_list.append(input1_word)
                input2_list.append(input2_word)

                input1_max_len = max(input1_max_len, len(input1_word))
                input2_max_len = max(input2_max_len, len(input2_word))

        # 2- embedding
        return self.word_embedding(input1_list, input2_list, input1_max_len, input2_max_len)
    

    def word_embedding(self, input1_list, input2_list, input1_max_len, input2_max_len):
        """
        当没有提供embedding的方法的时候，
        采用最简单的字母位置及出现则标记为1， 否则标记为0。 便于后面一个一个字母预测的时候抽取字母
        """
        samples_count = len(input1_list)
        input1_encode_data = np.zeros((samples_count, input1_max_len, len(self.char_set)), dtype='float64')
        input2_decode_data = np.zeros((samples_count, input2_max_len, len(self.char_set)), dtype='float64')
        target_data = np.zeros((samples_count, input2_max_len, len(self.char_set)), dtype='float64')

        # 将矩阵填充上数据 某个字母出现一次则标记增加1
        for num_idx, (inp1_w, inp2_w) in enumerate(zip(input1_list, input2_list)):
            for w_idx, chr_tmp in enumerate(inp1_w):
                input1_encode_data[num_idx, w_idx, self.char2int[chr_tmp]] = 1

            for w_idx, chr_tmp in enumerate(inp2_w):
                input2_decode_data[num_idx, w_idx, self.char2int[chr_tmp]] = 1
                if w_idx > 0: # 预测起始符后的
                    target_data[num_idx, w_idx - 1, self.char2int[chr_tmp]] = 1
        return input1_encode_data, input2_decode_data, target_data

    def gen_gibberish(self, eng_word, thresh=0.2):
        """
        生成错误单词
        20211017增加修正：针对较短单词
        """
        max_times = len(eng_word) * thresh
        times = int(random.randrange(1, len(eng_word)) * thresh) if max_times >= 2 else random.randrange(0, 2)
        while times != 0:
            times -= 1
            val = random.randrange(0, 10)
            idx = random.randrange(2, len(eng_word))
            insert_index = random.randrange(0, len(self.char_set))
            if val <=3 : # delete
                eng_word = eng_word[:idx] + eng_word[idx+1:]
            elif val <= 5: # add
                eng_word = eng_word[:idx] + self.char_set[insert_index] + eng_word[idx:]
            else: # replace
                eng_word = eng_word[:idx] + self.char_set[insert_index] + eng_word[idx+1:]

        return eng_word




# 定义纠正模型
## 编码 -> 解码
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras import Model
"""
monitor：要监测的数量。
factor：学习速率降低的因素。new_lr = lr * factor
patience：没有提升的epoch数，之后学习率将降低。
verbose：int。0：安静，1：更新消息。
mode：{auto，min，max}之一。在min模式下，当监测量停止下降时，lr将减少；在max模式下，当监测数量停止增加时，它将减少；在auto模式下，从监测数量的名称自动推断方向。
min_delta：对于测量新的最优化的阀值，仅关注重大变化。
cooldown：在学习速率被降低之后，重新恢复正常操作之前等待的epoch数量。
min_lr：学习率的下限。
"""

def de_right_word_tf2(lstm_units, out_dims, encode_max_len, decode_max_len, lr=0.001):
    encoder_lstm = LSTM(lstm_units, return_state=True)
    # 需要将各个隐层的结果作为下一层的输入时，选择设置 return_sequences=True 
    decoder_lstm = LSTM(lstm_units, return_state=True, return_sequences=True)
    fc = Dense(out_dims, activation='softmax')

    input_1 = Input(shape=(None, encode_max_len))
    encode_out, encode_h, encode_c = encoder_lstm(input_1)
    input_2 = Input(shape=(None, decode_max_len))
    decode_out, decode_h, decode_c = decoder_lstm(input_2, initial_state=[encode_h, encode_c])
    predict_out = fc(decode_out)
    model = Model([input_1, input_2], predict_out)
    opt = tf.keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-06)
    model.compile(
        optimizer=opt,
        loss=['categorical_crossentropy']
    )
    return model



## 由于预测的word 不知道何时结束， 所以我们需要对输入的值进行不断的修正，直到预测到末尾符为止
## 所以我们对于一个全新的输入，进行预测的时候需要先使得input2 输入为 一个全空的单词矩阵
def predict(m, input1_test):
    input2_orign = np.zeros((1, 36, 39))
    input2_orign[:, 0, g.char2int['\t']] = 1

    input_word = ''
    pred_word = ''

    for idx in range(input2_orign.shape[1] - 1): # max_encode_len
        p_tmp =  m.predict([tf.constant(input1_test), tf.constant(input2_orign)])
        # update input
        input2_w_idx = np.argmax(p_tmp[:, idx, :], axis=1)[0]
        # input2_orign[:, idx+1, :] = p_tmp[:, idx, :]
        input2_orign[:, idx+1, input2_w_idx] = 1
        
        input1_w_idx = np.argmax(input1_test[:, idx, :], axis=1)[0]
        pred_word += g.int2char[input2_w_idx]
        input_word += g.int2char[input1_w_idx]
#         print(f'[{idx}] input_word: {input_word},  pred_word : {pred_word}' )

        if (pred_word[-1] == '\n'):
            break
    print(f'[{idx}] input_word: {input_word[:-1]},  pred_word : {pred_word}' )
    return pred_word



def word2tensor(word):
    """
    当没有提供embedding的方法的时候，
    采用最简单的字母位置及出现则标记为1， 否则标记为0。 便于后面一个一个字母预测的时候抽取字母
    """
    char_set = [chr(i) for i in range(ord('a'), ord('z')+1)] + '0 1 2 3 4 5 6 7 8 9'.split() + ['\t', '\n', '#']
    char2int = dict(zip(char_set, range(len(char_set))))
    # int2char = dict(zip(range(len(char_set)), char_set))
    input1_encode_data = np.zeros((1, 34, len(char_set)), dtype='float64')

    # 将矩阵填充上数据 某个字母出现一次则标记增加1
    for w_idx, chr_tmp in enumerate(list(word)):
        if w_idx == 34:
            break
        input1_encode_data[0, w_idx, char2int[chr_tmp]] = 1

    return input1_encode_data


def word_correct(m, word):
    input1_encode_data = word2tensor(word)
    return predict(m, input1_encode_data)


if __name__ == '__main__':
    g = GeneratData()
    input1_encode_data, input2_decode_data, target_data = g.generate()

    m = de_right_word_tf2(256, 39, 39, 39)
    m.summary()
    # torch.optim.lr_scheduler.ReduceLROnPlateau 
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', mode='min', 
        patience=3, min_delta=0.001,
        factor = 0.5, verbose=1,
        min_lr=1e-5
    )
    his_ = m.fit([tf.constant(input1_encode_data), tf.constant(input2_decode_data)], tf.constant(target_data),
        epochs=500,
        batch_size=256, # 128,
        validation_split=0.2,
        callbacks=[scheduler]
    )

    # 由于输入输出都是有值的所有可以直接预测出所有值
    p = m.predict([tf.constant(input1_encode_data[1:2, :, :]), tf.constant(input2_decode_data[1:2, :, :])])
    print(p)

    input1_test = input1_encode_data[18:19, :, :]
    predict(input1_test)
