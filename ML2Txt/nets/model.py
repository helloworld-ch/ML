import torch
from torch import nn

def createRnn():
    """
    batch : 表示时间序列上，每个时间点所包含的信息数，在nlp中表示，有多少句话，实际上也就是当时间滑块
    到了第一单词位置的时候，该位置有多少个单词
    seq_len : 表示时间轴长，也就是时间序列
    emb_len : 表示信息表示需要的长度，在单词就是一个向量
    :return:

    这里需要注意，X_t@W_xh + H_(t-1)@W_hh = H_(t) 如何理解
    X_t表示每个时间点输入的信息 ：(batch,emb_len) ==>所以W_xh就会表示为(emb_len,hidden_len)
    这里的hidden_len是超参数，也就是最后输出的信息表征多少
    这里一算，所以得到的是【batch,hidden_len】的举证
    所以H_(t-1)必须是【batch,_】
    而H_(t)必须是[_,hidden_len]
    所以 H_(t-1)是【batch,hidden_len】,W_hh是【hidden_len,hidden_len】
    最后，这个hidden_len就是这个模型包含的信息量
    """
    ## 1. 初始化使用Rnn
    rnn = nn.RNN(100,10) # 100 表示emb_len也就是表征长，10表示hidden_len
    print(rnn.all_weights)

def getOneRnn():
    """
    使用 1 层的 Rnn网络
    :return:
    """
    # 再次解释一下，input_size 表示信息的表征
    # hidden_size 表示最后的信息量 ，也就是hidden_len
    # num_layers : 表示循环有多少层，不是有多长的时间序列
    rnn = nn.RNN(input_size=100,hidden_size=20,num_layers=1)
    x = torch.randn(10,3,100) # 时间序列长为10, batch是3 ，表征是100

    h0 = torch.zeros(1,3,20) # 也就是初始化是一个[num_layers,batch,hidden_len],也是最后的信息shape
    out,h_next = rnn(x,h0)

    print(out.shape,h_next.shape) # out表示之前所有得到的暂存h,而h_next表示最后一个 ，所以相较于h_next，out的第一位是序列长
    # ： torch.Size([10, 3, 20]) torch.Size([1, 3, 20])

def getFourLayersRnn():
    """
    使用 4 层 的 Rnn网路
    :return:
    """

    rnn = nn.RNN(100,20,num_layers=4)
    x = torch.randn(10,3,100)
    h0 = torch.zeros(4,3,20)

    print(torch.cuda.is_available())

    if torch.cuda.is_available():
        rnn = rnn.cuda()
        x = x.cuda()
        h0 = h0.cuda()

    out,h_next = rnn(x,h0)
    print(rnn._parameters.keys())
    print(out.shape,h_next.shape)

if __name__ == '__main__':
    # createRnn()
    # getOneRnn()
    getFourLayersRnn()
