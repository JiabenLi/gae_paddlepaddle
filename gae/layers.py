import paddle
import paddle.nn.functional as F
import paddle.nn as nn

class GraphConvolution(nn.Layer):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.w_attr = self._init_weights()
        self.linear = nn.Linear(in_features, out_features, weight_attr=self.w_attr)

    def _init_weights(self):
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())

        return weight_attr

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = self.linear(input)


        output = paddle.matmul(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'