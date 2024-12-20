from tensorflow.keras.layers import Dense, Input, Dropout
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras


def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    t_loss = weights*tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(t_loss)


def MLP(nvars,
        layer_sizes = [64,128,64],
        activation = 'gelu',nensemb=1):    
    '''
    Define a simple fully conneted model to be used during unfolding
    Function Inputs:
    nvars (int): The number of input variables for the model, defining the dimensionality of the input data.
    layer_sizes (list of int, optional, default=[64, 128, 64]): A list specifying the number of neurons in each hidden layer. The length of the list determines the number of layers, and each value corresponds to the number of units in that layer.
    activation (str, optional, default='gelu'): The activation function applied to each hidden layer. By default, it is set to 'gelu' (Gaussian Error Linear Unit), which adds non-linearity to the model.
    nensemb (int, optional, default=1): Will create nensemb models with output taken as the average of each model response
    '''

    inputs = Input((nvars, ))
    layer = Dense(layer_sizes[0],activation=activation)(inputs)

    for layer_size in layer_sizes[1:]:
        layer = Dense(layer_size,activation=activation)(layer)

    outputs = Dense(1)(layer)
    model = Model(inputs = inputs, outputs = outputs)
    return model

class PET(Model):
    """OmniFold Classifier class"""
    def __init__(self,
                 num_feat,
                 num_evt = 0,
                 num_part=100,
                 num_heads=2,
                 num_transformer= 2,
                 projection_dim= 32,
                 local = True,
                 K = 3,
                 layer_scale = True,
                 layer_scale_init = 1e-3
                 ):
        super(PET, self).__init__()
        '''
        Class Variables:
        num_feat (int): The number of input features per particle.
        num_evt (int, optional, default=0): Number of high-level featuresto use. If 0, no high level feature is considered
        num_part (int, optional, default=100): Number of particles or objects per event in the input data.
        num_heads (int, optional, default=2): Number of attention heads in the transformer layer, for multi-head self-attention.
        num_transformer (int, optional, default=2): Number of transformer layers stacked in the model.
        projection_dim (int, optional, default=32): Dimensionality of the projection layer for transforming input features.
        local (bool, optional, default=True): Flag to indicate whether a local GNN using k-nearest neighbors is used..
        K (int, optional, default=3): Parameter controlling the number of k-nearest neighbors to use.
        layer_scale (bool, optional, default=True): Determines whether layer scaling is applied for stabilizing training.
        layer_scale_init (float, optional, default=1e-3): Initial value for the layer scaling factor to ensure stable gradient behavior.
        '''
        
        self.num_feat = num_feat
        self.num_evt = num_evt
        self.num_part = num_part
        inputs_part = layers.Input((self.num_part,self.num_feat))
        if local:
            assert self.num_part >= K, "ERROR: K neighbors exceeding the number of particles"
            
        if self.num_evt>0:
            inputs_evt = layers.Input((self.num_evt))
        else:
            inputs_evt =  None
                        
        outputs_body = self.PET_body(inputs_part,
                                     num_part,
                                     num_heads,
                                     num_transformer,
                                     projection_dim,
                                     local = local, K = K,
                                     layer_scale = layer_scale,
                                     layer_scale_init = layer_scale_init
                                     )
                
        outputs_head = self.PET_head(outputs_body,inputs_evt,projection_dim)

        if inputs_evt is None:
            self.model = Model(inputs=inputs_part,
                               outputs=outputs_head)
        else:
            self.model = Model(inputs=inputs_part,
                               outputs=outputs_head)

        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def call(self,x, training=True):
        return self.model(x,training)
        

    def PET_body(
            self,
            inputs_part,
            num_parts,
            num_heads=4,
            num_transformer= 8,
            projection_dim= 64,
            local = True, K = 5,
            num_local = 2,
            layer_scale = True,
            layer_scale_init = 1e-3,
    ):
    
        encoded = get_encodding(inputs_part,projection_dim)
        inputs_mask = tf.cast(inputs_part[:,:,0,None] != 0, dtype='float32')
        if local:
            coord_shift = tf.multiply(999., tf.cast(tf.equal(inputs_mask, 0), dtype='float32'))        
            points = inputs_part[:,:,:2] #assume first 2 coordinates are eta-phi
            local_features = inputs_part
            for _ in range(num_local):
                local_features = get_neighbors(coord_shift+points,local_features,projection_dim,K)
                points = local_features
                
            encoded = layers.Add()([local_features,encoded])
            
        skip_connection = encoded*inputs_mask                     
        for i in range(num_transformer):
            x1 = layers.GroupNormalization(groups=1)(encoded)
            updates = layers.MultiHeadAttention(num_heads=num_heads,
                                                key_dim=projection_dim//num_heads)(x1,x1)
                
            if layer_scale:
                updates = LayerScale(layer_scale_init, projection_dim)(updates,inputs_mask)
            
            x2 = layers.Add()([updates,encoded])
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
            x3 = layers.Dense(projection_dim)(x3)
            if layer_scale:
                x3 = LayerScale(layer_scale_init, projection_dim)(x3,inputs_mask)
            encoded = layers.Add()([x3,x2])*inputs_mask
                        
        return encoded + skip_connection


    def PET_head(
            self,
            encoded,
            input_evt,
            projection_dim= 64,
            num_heads=4,
            num_class_layers=2,
            layer_scale = True,
            layer_scale_init = 1e-3,
    ):


        if input_evt is not None:
            conditional = layers.Dense(2*projection_dim,activation='gelu')(input_evt)
            conditional = tf.tile(conditional[:,None, :], [1,tf.shape(encoded)[1], 1])
            scale,shift = tf.split(conditional,2,-1)
            encoded = encoded*(1.0 + scale) + shift

        class_tokens = tf.Variable(tf.zeros(shape=(1, projection_dim)),trainable = True)    
        class_tokens = tf.tile(class_tokens[None, :, :], [tf.shape(encoded)[0], 1, 1])
        
        for _ in range(num_class_layers):
            concatenated = tf.concat([class_tokens, encoded],1)

            x1 = layers.GroupNormalization(groups=1)(concatenated)            
            updates = layers.MultiHeadAttention(num_heads=num_heads,
                                                key_dim=projection_dim//num_heads)(
                                                    query=x1[:,:1], value=x1, key=x1)
            updates = layers.GroupNormalization(groups=1)(updates)
            if layer_scale:
                updates = LayerScale(layer_scale_init, projection_dim)(updates)

            x2 = layers.Add()([updates,class_tokens])
            x3 = layers.GroupNormalization(groups=1)(x2)
            x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
            x3 = layers.Dense(projection_dim)(x3)
            if layer_scale:
                x3 = LayerScale(layer_scale_init, projection_dim)(x3)
            class_tokens = layers.Add()([x3,x2])

        class_tokens = layers.GroupNormalization(groups=1)(class_tokens)
        outputs_pred = layers.Dense(1,activation=None)(class_tokens[:,0])
                
        return outputs_pred


    def train_step(self, inputs):
        x,y = inputs
        with tf.GradientTape(persistent=True) as tape:
            y_pred = self.model(x)
            loss = weighted_binary_crossentropy(y, y_pred)
            
        self.optimizer.minimize(loss,self.model.trainable_variables,tape=tape)        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        x,y = inputs            
        y_pred = self.model(x)
        loss = weighted_binary_crossentropy(y, y_pred)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        

def get_neighbors(points,features,projection_dim,K):
    drij = pairwise_distance(points)  # (N, P, P)
    _, indices = tf.nn.top_k(-drij, k=K + 1)  # (N, P, K+1)
    indices = indices[:, :, 1:]  # (N, P, K)
    knn_fts = knn(tf.shape(points)[1], K, indices, features)  # (N, P, K, C)
    knn_fts_center = tf.broadcast_to(tf.expand_dims(features, 2), tf.shape(knn_fts))
    local = tf.concat([knn_fts-knn_fts_center,knn_fts_center],-1)
    local = layers.Dense(2*projection_dim,activation='gelu')(local)
    local = layers.Dense(projection_dim,activation='gelu')(local)
    local = tf.reduce_mean(local,-2)
    
    return local


def pairwise_distance(point_cloud):
    r = tf.reduce_sum(point_cloud * point_cloud, axis=2, keepdims=True)
    m = tf.matmul(point_cloud, point_cloud, transpose_b = True)
    D = r - 2 * m + tf.transpose(r, perm=(0, 2, 1))
    return tf.abs(D)


def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)    
    batch_size = tf.shape(features)[0]

    batch_indices = tf.reshape(tf.range(batch_size), (-1, 1, 1))
    batch_indices = tf.tile(batch_indices, (1, num_points, k))
    indices = tf.stack([batch_indices, topk_indices], axis=-1)
    return tf.gather_nd(features, indices)


def get_encodding(x,projection_dim):
    x = layers.Dense(2*projection_dim,activation='gelu')(x)
    x = layers.Dense(projection_dim,activation='gelu')(x)
    return x

class LayerScale(layers.Layer):
    def __init__(self, init_values, projection_dim, **kwargs):
        super(LayerScale, self).__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.gamma_initializer = tf.keras.initializers.Constant(self.init_values)

    def build(self, input_shape):
        # Ensure the layer is properly built by defining its weights in the build method
        self.gamma = self.add_weight(
            shape=(self.projection_dim,),
            initializer=self.gamma_initializer,
            trainable=True,
            name='gamma'
        )
        super(LayerScale, self).build(input_shape)


    def call(self, inputs,mask=None):
        # Element-wise multiplication of inputs and gamma
        if mask is not None:
            return inputs * self.gamma* mask
        else:
            return inputs * self.gamma

    def get_config(self):
        # Returns the configuration of the layer for serialization
        config = super(LayerScale, self).get_config()
        config.update({
            'init_values': self.init_values,
            'projection_dim': self.projection_dim
        })
        return config

