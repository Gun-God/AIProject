import re, time
from nltk.tokenize import word_tokenize
from nltk import data
from collections import defaultdict, Counter

import json

data.path.append(r"L:\python_re\nltk_data")


class MarkovChain:
    def __init__(self):
        # 构建一个默认value为list的空字典
        self.lookup_dict = defaultdict(list)
        # 数据处理，将单词进行分类

    def add_document(self, string):
        self.lookup_dict.clear()
        with open(string, encoding='utf-8') as f:
            sents = f.read()
        # 返回单词的音节列表
        preprocessed_list = self._preprocess(sents)
        # 生成
        pairs = self.__generate_tuple_keys(preprocessed_list)
        # 将生成的词组加入字典中
        for pair in pairs:
            self.lookup_dict[pair[0]].append(pair[1])
        #print(self.lookup_dict)
        pairs2 = self.__generate_2tuple_keys(preprocessed_list)
        for pair in pairs2:
            self.lookup_dict[tuple([pair[0], pair[1]])].append(pair[2])

        #print(self.lookup_dict)
        # json_dicts = json.dumps(self.lookup_dict, indent=4, ensure_ascii=False)
        # print(json_dicts)

        pairs3 = self.__generate_3tuple_keys(preprocessed_list)
        for pair in pairs3:
            self.lookup_dict[tuple([pair[0], pair[1], pair[2]])].append(pair[3])

    # 预处理，返回单词的音节
    def _preprocess(self, string):
        # 将短句分为单词
        cleaned = re.sub(r'\W+', ' ', string).lower()
        tokenized = word_tokenize(cleaned)
        return tokenized

    # 每两个单词作为键
    def __generate_tuple_keys(self, data):
        if len(data) < 1:
            return
        for i in range(len(data) - 1):
            yield [data[i], data[i + 1]]
    # 每三个单词作为键
    def __generate_2tuple_keys(self, data):
        if len(data) < 2:
            return
        for i in range(len(data) - 2):
            yield [data[i], data[i + 1], data[i + 2]]
    # 每四个单词作为键
    def __generate_3tuple_keys(self, data):
        if len(data) < 3:
            return
        for i in range(len(data) - 3):
            yield [data[i], data[i + 1], data[i + 2], data[i + 3]]
    # 根据一个单词查找字典中的候选词
    def oneword(self, string):
        # 用Counter统计出现次数最多的前10个候选词
        return Counter(self.lookup_dict[string]).most_common()[:10]

    # 根据两个单词查找字典中的候选词
    def twowords(self, string):
        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:10]
        # suggest =self.lookup_dict[tuple(string)]
        if len(suggest) == 0:
            return self.oneword(string[-1])
        return suggest

    # 根据三个单词查找字典中的候选词
    def threewords(self, string):
        suggest = Counter(self.lookup_dict[tuple(string)]).most_common()[:10]
        # suggest =self.lookup_dict[tuple(string)]
        if len(suggest) == 0:
            return self.twowords(string[-2:])
        return suggest

    # 根据四个以上单词查找可能的候选字
    def morewords(self, string):
        return self.threewords(string[-3:])

    def generate_text(self, string):
        # print('c',len(self.lookup_dict))
        # 判断字典中是否有数据
        if len(self.lookup_dict) > 0:
            # print("chang:",len(string))
            if len(string) == 0:
                return 0
            print("输入:", string)
            txt = ''
            # 根据空格符分隔词组
            tokens = string.split(" ")
            if len(tokens) == 1:
                # 根据一个单词查找
                txt = self.oneword(string)
                # print("len:",len(txt))
            elif len(tokens) == 2:
                # 根据两个单词查找
                txt = self.twowords(string.split(" "))
            elif len(tokens) == 3:
                # 根据三个单词查找
                txt = self.threewords(string.split(" "))
            elif len(tokens) > 3:
                txt = self.morewords(string.split(" "))
            if txt == set():
                txt = '无'
            print("候选词:", txt)
            write_word(string)
            if len(txt) == 0:
                write_word(string)
                return 0
        return 1


def write_word(word):
    with open("doc3.txt", "a", encoding='utf-8') as file:
        file.write('\n')
        file.write(word)
        file.close()


if __name__ == "__main__":
    # 实例化马尔科夫链
    my_markov = MarkovChain()
    # 加载数据模型
    # 预测候选词
    m = 1
    while m != 0:
        my_markov.add_document("doc3.txt")
        k = input().strip().lower()
        # print("输入:",k)
        m = my_markov.generate_text(k)
    # if  m==0:
    #      break
    #   elif m==1:
    #      continue
