import torch
import torch.nn as nn
import torch.optim as optim
import pdb


def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # space tokenizer
        input = [word_dict[n] for n in word[:-1]]  # create (1~n-1) as input
        target = word_dict[word[-1]]  # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)
    print("input_batch",input_batch)
    print("target_batch",target_batch)
    return input_batch, target_batch


# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)  ### 矩阵Q  (V x m)  V 表示word的字典大小, m 表示词向量的维度
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)  ###
        self.d = nn.Parameter(torch.ones(n_hidden))
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        X = self.C(X)  # X : [batch_size, n_step, m]
        X = X.view(-1, n_step * m)  # [batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.H(X))  # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh)  # [batch_size, n_class]
        return output


if __name__ == '__main__':
    n_step = 2  # number of steps, n-1 in paper,根据前两个单词预测第三个
    n_hidden = 2  # number of hidden size, h in paper,隐层个数
    m = 2  # embedding size, m in paper ,词向量的维度  m =2

    sentences = ["how many people", "how many countries", "how many episodes", "how many seasons", "how many  states"]  ###训练数据

    word_list = " ".join(sentences).split()  ###  按照空格分词,统计 sentences的分词的个数
    word_list = list(set(word_list))  ###  去重 统计词典个数
    print("word_list:",word_list)
    word_dict = {w: i for i, w in enumerate(word_list)}  ### {word : index   ,词典}
    print("word_dict:", word_dict)
    number_dict = {i: w for i, w in enumerate(word_list)}  ###  {index : word ,词典}
    print("number_dict:", number_dict)
    n_class = len(word_dict)  # number of Vocabulary   词典的个数，也是softmax 最终分类的个数

    # pdb.set_trace()
    model = NNLM()

    ### 损失函数定义
    criterion = nn.CrossEntropyLoss()  ### 交叉熵损失函数
    ### 采用 Adam 优化算法 学习率定义为   0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    input_batch, target_batch = make_batch()  ###构建输入数据和 target label
    input_batch = torch.LongTensor(input_batch)  ### 模型输入 tensor 形式
    target_batch = torch.LongTensor(target_batch)
    print("input_batch", input_batch)
    print("target_batch", target_batch)
    # 训练模型 迭代 5000次
    for epoch in range(5000):
        optimizer.zero_grad()  ###梯度归零
        output = model(input_batch)
        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()  ### 反向传播计算 每个参数的梯度值
        optimizer.step()  ### 每一个参数的梯度值更新
    # 预测
    predict = model(input_batch).data.max(1, keepdim=True)[1]

    # Test
    print("输入:",[sen.split()[:2] for sen in sentences], '\n候选词:', [number_dict[n.item()] for n in predict.squeeze()])
