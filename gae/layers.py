import paddle
import paddle.nn.functional as F
import paddle.nn as nn

class GraphConvolution(nn.layer):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.XavierUniform())

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = paddle.matmul(input, self.weight)
        output = paddle.matmul(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'