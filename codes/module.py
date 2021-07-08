import tensorflow as tf
import tensorflow.keras as keras
import layers
# tf.keras.backend.set_floatx('float64')


def generator_attention(input_shape=512, 
                      filter_size=[64, 128, 256, 512, 512, 512],
                      kernel_size=[16, 16, 16, 16, 16, 16],
                      n_downsample=6,
                      norm='layer_norm', 
                      skip_connection=True):
    
    """ 
    input_shape = 128*4
    """
        
    
    def _downsample(ip, filter_size, kernel_size, norm, stride_size=2):
        
        ip = layers.Conv1D(filters=filter_size, kernel_size=kernel_size, strides=stride_size, padding='same', use_bias=False)(ip)
        # ip = tf.dtypes.cast(ip, tf.float32)
        if norm != 'none':
            ip = layers.normalization(norm)(ip)
        ip = layers.Activation(ip, activation='leaky_relu')
        
        return ip
    
    def _upsample(ip, filter_size, kernel_size, norm, stride_size=2, drop_rate = 0.5, apply_dropout=False):
        
        ip = layers.DeConv1D(filters=filter_size, kernel_size=kernel_size, strides=stride_size, padding='same', use_bias=False)(ip)
        # ip = tf.dtypes.cast(ip, tf.float32)
        if norm != 'none':
            ip = layers.normalization(norm)(ip)
        if apply_dropout:
            ip = layers.Dropout(rate=drop_rate)
        ip = layers.Activation(ip, activation='relu')   
        
        return ip
        
        
    ## input
    h = inputs = keras.Input(shape=input_shape) # None, 512
    h = tf.expand_dims(h, axis=1) # None, 1, 512
    h = tf.expand_dims(h, axis=3) # None, 1, 512, 1
    
    ## downsample
    connections = []
    for k in range(n_downsample): 
        
        # filter_size *=2
        # kernel_size = kernel_size
        if k==0:
            h =  _downsample(h, filter_size[k], kernel_size[k], 'none')
        else:
            h =  _downsample(h, filter_size[k], kernel_size[k], norm)
            
        connections.append(h)
        
    ## upsampling`
    # filter_size = filter_size//2
    h = _upsample(h, filter_size[k], kernel_size[k], norm, stride_size=1)
    if skip_connection:
        _h = layers.attention_block_1d(curr_layer= h, conn_layer= connections[n_downsample-1])   
        h  = keras.layers.add([h, _h])


    for l in range(1, n_downsample):

        h  = _upsample(h, filter_size[k-l], kernel_size[k-l], norm)
        if skip_connection:
            _h = layers.attention_block_1d(curr_layer= h, conn_layer= connections[k-l])
            h  = keras.layers.add([h, _h])        
            
    ## output
    h = layers.DeConv1D(filters=1, kernel_size=kernel_size[k-l], strides=2, padding='same')(h)
    h = layers.Activation(h, activation='tanh')
    h = tf.squeeze(h, axis=1)
    h = tf.squeeze(h, axis=2)

    return keras.Model(inputs=inputs, outputs=h)
