import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.keras.layers import Layer


class TimeJointEmbedding(Layer):

    def __init__(self,
                 embedding_size=10,
                 time_embedding_size=10,
                 projection_size=10,
                 time_initializer='glorot_uniform',
                 time_regularizer=None,
                 time_constraint=None,
                 ):
        super(time_joint_embedding, self).__init__()

        self.time_embedding_size = time_embedding_size
        self.embedding_size = embedding_size
        self.projection_size = projection_size
        self.time_activation = activations.get('sigmoid')
        self.time_regularizer = regularizers.get(time_regularizer)
        self.time_initializer = initializers.get(time_initializer)
        self.time_constraint = constraints.get(time_constraint)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        input_dim -= 1

        self.input_embedding = self.add_weight(shape=(input_dim, self.embedding_size),
                                               name='embedding_matrix',
                                               trainable=True
                                               )

        self.time_kernel = self.add_weight(shape=(1, self.projection_size),
                                           name='time_kernel',
                                           initializer=self.time_initializer,
                                           regularizer=self.time_regularizer,
                                           constraint=self.time_constraint,
                                           trainable=True)

        self.embedding_projector = self.add_weight(shape=(self.projection_size, self.embedding_size),
                                                   name='time_projection_onto_embedding_space',
                                                   initializer=self.time_initializer,
                                                   regularizer=self.time_regularizer,
                                                   constraint=self.time_constraint,
                                                   trainable=True
                                                   )

        self.time_bias = self.add_weight(shape=(self.projection_size,),
                                         name='time_bias')
        self.built = True

    def call(self, inputs):
        dt = inputs[:, :, -1]
        # dt = tf.reshape(dt, (-1,1))
        dt = tf.expand_dims(dt, -1)
        p_d = dt * self.time_kernel
        p_d = K.bias_add(p_d, self.time_bias)

        # aux_mat = self.time_activation(p_d)

        # s_d = aux_mat/tf.reshape(tf.reduce_sum(aux_mat, axis=1), (-1,1))
        s_d = tf.nn.softmax(p_d)

        g_d = K.dot(s_d, self.embedding_projector)

        x_t = inputs[:, :, :-1]

        x_t = K.dot(x_t, self.input_embedding)
        x_t = tf.add(x_t, g_d) / 2

        return x_t


class TimeMaskEmbedding(Layer):

    def __init__(self,
                 embedding_size=10,
                 time_embedding_size=10,
                 projection_size=10,
                 time_initializer='glorot_uniform',
                 time_regularizer=None,
                 time_constraint=None,
                 ):
        super(time_mask_embedding, self).__init__()

        self.time_embedding_size = time_embedding_size
        self.embedding_size = embedding_size
        self.projection_size = projection_size
        self.time_activation = activations.get('sigmoid')
        self.time_regularizer = regularizers.get(time_regularizer)
        self.time_initializer = initializers.get(time_initializer)
        self.time_constraint = constraints.get(time_constraint)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        input_dim -= 1
        tf.print(input_dim)
        self.input_embedding = self.add_weight(shape=(input_dim, self.embedding_size),
                                               name='embedding_matrix',
                                               trainable=True
                                               )

        self.time_kernel = self.add_weight(shape=(1, self.projection_size),
                                           name='time_kernel',
                                           initializer=self.time_initializer,
                                           regularizer=self.time_regularizer,
                                           constraint=self.time_constraint,
                                           trainable=True)

        self.embedding_projector = self.add_weight(shape=(self.projection_size, self.embedding_size),
                                                   name='time_projection_onto_embedding_space',
                                                   initializer=self.time_initializer,
                                                   regularizer=self.time_regularizer,
                                                   constraint=self.time_constraint,
                                                   trainable=True
                                                   )

        tf.print(self.input_embedding.shape)

        self.time_bias = self.add_weight(shape=(self.embedding_size,),
                                         name='time_bias')
        self.built = True

    def call(self, inputs):
        dt = inputs[:, :, -1]
        # dt = tf.reshape(dt, (-1,1))
        dt = tf.expand_dims(dt, -1)

        log_dt = tf.math.log(1 + dt)
        c_d = tf.nn.relu(K.dot(log_dt, self.time_kernel))
        c_d_W_d = K.dot(c_d, self.embedding_projector)
        c_d_W_d = K.bias_add(c_d_W_d, self.time_bias)
        m_d = tf.math.sigmoid(c_d_W_d)

        x_t = inputs[:, :, :-1]
        x_t = K.dot(x_t, self.input_embedding)

        x_t = x_t * m_d
        return x_t