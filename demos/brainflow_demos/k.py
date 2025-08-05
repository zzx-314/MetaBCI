import tensorflow as tf

class CentralityEncoding(tf.keras.layers.Layer):

    def __init__(self, max_degree, d_model):
        super(CentralityEncoding, self).__init__()
        self.centr_embedding = tf.keras.layers.Embedding(max_degree, d_model)

    def centrality(self, distances):
        centrality = tf.cast(tf.math.equal(tf.math.abs(distances), 1), tf.float32)
        centrality = tf.math.reduce_sum(centrality, axis=-1, keepdims=False)
        return tf.cast(centrality, tf.float32)

    def call(self, distances):
        centrality = self.centrality(distances)
        centrality_encoding = self.centr_embedding(centrality)
        return centrality_encoding


class SpatialEncoding(tf.keras.layers.Layer):

    def __init__(self, d_sp_enc=16, activation='relu'):
        super(SpatialEncoding, self).__init__()
        self.d_sp_enc = d_sp_enc
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(d_sp_enc, activation=activation)
        self.dense2 = tf.keras.layers.Dense(1, activation=activation)

    def call(self, distances):
        expanded_inputs = tf.expand_dims(distances, axis=-1)
        outputs = self.dense1(expanded_inputs)
        outputs = self.dense2(outputs)
        spatial_encoding = tf.squeeze(outputs, axis=-1)
        return spatial_encoding


def scaled_dot_product_attention(q, k, v, min_distance_matrix, spatial_encoding, num_heads):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    spatial_encoding_bias = spatial_encoding(min_distance_matrix)
    spatial_encoding_bias = tf.expand_dims(spatial_encoding_bias, axis=1)
    spatial_encoding_bias = tf.tile(spatial_encoding_bias, [1, num_heads, 1, 1])
    scaled_attention_logits += spatial_encoding_bias

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, d_sp_enc, sp_enc_activation):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_sp_enc = d_sp_enc
        self.sp_enc_activation = sp_enc_activation
        self.spatial_encoding = SpatialEncoding(self.d_sp_enc, self.sp_enc_activation)

        assert d_model % self.num_heads == 0

        self.depth = d_model

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        depth_per_head = self.depth // self.num_heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, depth_per_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, min_distance_matrix, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, min_distance_matrix, self.spatial_encoding, self.num_heads)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class GraphormerBlock(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, rate=0.1, d_sp_enc=128, sp_enc_activation='relu'):
        super(GraphormerBlock, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, d_sp_enc, sp_enc_activation)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.epsilon = 1e-6

    def layer_norm(self, inputs, epsilon=1e-6):
        mean, variance = tf.nn.moments(inputs, axes=[-1], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + epsilon)

        return normalized

    def call(self, x, training, mask, min_distance_matrix):
        residual = x
        x_norm = self.layer_norm(x)
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, min_distance_matrix, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = residual + attn_output
        residual = out1
        out1_norm = self.layer_norm(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = residual + ffn_output
        return out2
