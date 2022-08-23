from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk import data
import numpy as np
import re
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

data.path.append(r"L:\python_re\nltk_data")

#数据处理
def data_handle(path):
    print("开始预处理数据")
    # 读取文件
    with open(path, encoding='utf-8') as f:
        sents = f.read()
    cleaned = re.sub(r'\W+', ' ', sents).lower()
    #将短句分为单词
    tokens = word_tokenize(cleaned)
    print(tokens)
    train_len = 4
    text_sequences = []
    for i in range(train_len, len(tokens)):
        seq = tokens[i - train_len:i]
        text_sequences.append(seq)
    sequences = {}
    count = 1
    for i in range(len(tokens)):
        if tokens[i] not in sequences:
            sequences[tokens[i]] = count
            count += 1
    tokenizer = Tokenizer()
    # 实现分词
    tokenizer.fit_on_texts(text_sequences)
    print("分词后:",text_sequences)
    # 输出向量序列
    sequences = tokenizer.texts_to_sequences(text_sequences)
    print("分词后向量序列:", sequences)
    # Collecting some information
    vocabulary_size = len(tokenizer.word_counts) + 1

    n_sequences = np.empty([len(sequences), train_len], dtype='int32')
    for i in range(len(sequences)):
        n_sequences[i] = sequences[i]
    print('拆分前',(n_sequences[:1]))
    train_inputs = n_sequences[:, :-1]
    print('拆分后输入目标标签',train_inputs[:1])
    train_targets = n_sequences[:, -1]
    print('拆分后输出(预测)目标标签',train_targets[:1])
    # 将类别向量转换为二进制
    train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
    print('One-hot vectors', train_targets[:5])
    seq_len = train_inputs.shape[1]
    model = train_model(vocabulary_size, train_inputs, train_targets, seq_len)
    return tokenizer, seq_len, model


# train_inputs.shape
# print(train_targets[0])
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import os

#训练模型
def train_model(vocabulary_size, train_inputs, train_targets, seq_len):
    if not os.path.exists("mymodel.h5"):
        print('开始创建模型')
        # 训练模型
        model = Sequential()
        model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(vocabulary_size, activation='softmax'))
        # compile network
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train_inputs, train_targets, epochs=500, verbose=1)
        model.save("mymodel.h5")
        print('模型创建完成')
    else:
        # 加载模型
        print('加载模型')
        model = load_model("mymodel.h5")

    return model


if __name__ == "__main__":

    # 数据处理
    tokenizer, seq_len, model = data_handle("doc3.txt")
    # 根据输入对应关系输出的向量序列
    input_text = input().strip().lower()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    # 对上面生成的不定长序列进行补全
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
    print("输入:", input_text)
    l = len(tokenizer.index_word)
    # 开始预测
    for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
            pred_word = tokenizer.index_word[i]
            print("候选词:", pred_word)
