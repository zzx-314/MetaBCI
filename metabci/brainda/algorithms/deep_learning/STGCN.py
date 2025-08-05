import tensorflow as tf
from keras import layers, models
from demos.brainflow_demos.k import CentralityEncoding,GraphormerBlock
from keras import backend as K

class Graphormer(tf.keras.layers.Layer):  #图注意力网络

    def __init__(self, num_layers, d_model, num_heads,
                 dropout=0.1, dff=512, d_sp_enc=128, sp_enc_activation='relu',
                 max_num_nodes=256, model_head='VNode', concat_n_layers=1):
        super(Graphormer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads  # Add this line to initialize num_heads
        self.max_num_nodes = max_num_nodes
        self.num_layers = num_layers
        self.centr_encoding = CentralityEncoding(max_num_nodes, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)

        # Save other parameters to instance variables


        self.dff = dff
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.model_head = model_head
        self.concat_n_layers = concat_n_layers

        self.graphormer_layers = [GraphormerBlock(d_model, num_heads, dff, dropout, d_sp_enc, sp_enc_activation)
                                  for _ in range(num_layers)]

    def create_padding_mask(self, nodes):
        return tf.cast(tf.math.equal(nodes, 0), tf.float32)

    def call(self, inputs, training=None):
        node_features, distance_matrix = inputs
        mask = self.create_padding_mask(node_features)
        attention_mask = mask[:, tf.newaxis, :]
        # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa*10")
        embed = node_features
        embed += self.centr_encoding(distance_matrix)
        out = self.dropout(embed, training=training)

        for i in range(self.num_layers):
            out = self.graphormer_layers[i](out, training, attention_mask, distance_matrix)
            # print('fkjfhsfshjdbfsdhjfbdshjfbdshjfbsd',out.shape)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout.rate,
            'dff': self.dff,
            'd_sp_enc': self.d_sp_enc,
            'sp_enc_activation': self.sp_enc_activation,
            'max_num_nodes': self.max_num_nodes,
            'model_head': self.model_head,
            'concat_n_layers': self.concat_n_layers
        })
        return config


def diff_loss(diff, S):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape)==4:
        # batch input
        return K.mean(K.sum(K.sum(diff**2,axis=3)*S, axis=(1,2)))
    else:
        return K.sum(K.sum(diff**2,axis=2)*S)


def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape)==3:
        # batch input
        return Falpha * K.sum(K.mean(S**2,axis=0))
    else:
        return Falpha * K.sum(S**2)


class Graph_Learn(tf.keras.layers.Layer):  #图结构学习模块

    def __init__(self, alpha, **kwargs):
        super(Graph_Learn, self).__init__(**kwargs)
        self.alpha = alpha
        self.S = tf.convert_to_tensor([[[0.0]]])  # 类似于占位符
        self.diff = tf.convert_to_tensor([[[[0.0]]]])  # 类似于占位符

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        # 在层中添加 L_{graph_learning} 的损失
        self.add_loss(F_norm_loss(self.S, self.alpha))
        self.add_loss(diff_loss(self.diff, self.S))
        super(Graph_Learn, self).build(input_shape)

    def call(self, x):
        # 输入:  [N, timesteps, vertices, features]
        _, T, V, F = x.shape
        N = tf.shape(x)[0]

        # 形状: (N,V,F) 使用当前切片（中间切片）
        x = x[:, int(x.shape[1]) // 2, :, :]
        # 形状: (N,V,V,F)
        diff = tf.transpose(tf.transpose(tf.broadcast_to(x, [V, N, V, F]), perm=[2, 1, 0, 3]) - x, perm=[1, 0, 2, 3])
        # 形状: (N,V,V)
        tmpS = K.exp(K.relu(K.reshape(K.dot(K.abs(diff), self.a), [N, V, V])))
        # 归一化
        S = tmpS / K.sum(tmpS, axis=1, keepdims=True)

        self.diff = diff
        self.S = S
        return S

    def get_config(self):
        config = super(Graph_Learn, self).get_config()
        config.update({"alpha": self.alpha})
        return config

def STGCNBlock(x,GLalpha,num_of_time_filters, time_conv_strides,time_conv_kernel):
    distance_matrix = Graph_Learn(alpha=GLalpha)(x)
    time_conv_output = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(time_conv_kernel, 1),
        strides=(time_conv_strides, 1)
    )(x)
    time_conv_output = tf.squeeze(time_conv_output, axis=1)
    out = Graphormer(num_layers=1, d_model=256, num_heads=8, )([time_conv_output, distance_matrix])
    return out

def build_STGCN(sample_shape, dense_size, opt, GLalpha, regularizer, dropout,num_of_time_filters, time_conv_strides,time_conv_kernel):
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = layers.Input(shape=sample_shape, name='Input-Data')
    block_out = STGCNBlock(data_layer,GLalpha,num_of_time_filters, time_conv_strides,time_conv_kernel)
    block_out = layers.Flatten()(block_out)
    for size in dense_size:
        block_out = layers.Dense(size,activation='relu',kernel_regularizer=regularizer)(block_out)

    # dropout
    if dropout != 0:
        block_out = layers.Dropout(dropout)(block_out)

    softmax = layers.Dense(5, activation='softmax')(block_out)


    model = models.Model(inputs=data_layer, outputs=softmax)

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    return model