# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
from pc_distance import tf_nndistance, tf_approxmatch

def _variable_on_cpu(name, shape, initializer, use_fp16=False, trainable=True):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def mlp(features, layer_dims, bn=None, bn_params=None):
    print('Ft shape is {}'.format(features.get_shape()))
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs

def mlp_conv2d(inputs, layer_dims, bn=False, activation_fn=tf.nn.relu, bn_decay=None, is_training=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=[1,1], use_bias=True
        )

    outputs = tf.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=[1,1], use_bias=True)
   
    return outputs


def mlp_conv2d_reg(inputs, layer_dims, bn=False, weight_decay=0.00001, activation_fn=tf.nn.relu, 
        bn_decay=None, is_training=None, padding="VALID", kernel_size=[1,1]):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=[5,5], padding="SAME", use_bias=True,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
        )

        if bn:
            inputs = tf.layers.batch_normalization(
              inputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)

        if activation_fn is not None:
            inputs = activation_fn(inputs)

    outputs = tf.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=[5,5], 
        use_bias=True,
        padding="SAME",
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay)
      )
      
    if bn:
        outputs = tf.layers.batch_normalization(
        outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)    

def conv_layers(inputs, num_layers, out_dims, args):
    outputs = []
    for i in range(num_layers):
        for j in range(len(out_dims)):
            random_grid = tf.get_variable('random_grid',shape=[args.batch_size, 64, 2],
                        dtype = tf.float32, initializer =tf.random_normal_initializer())
            inputs_tmp = tf.concat([random_grid, inputs], axis=2)
            if j==0:
                out = tf.layers.conv1d(inputs_tmp, out_dims[j],
                    kernel_size=1,use_bias=True)
            else:
                out = tf.layers.conv1d(out, out_dims[j],
                    kernel_size=1, use_bias=True)
        outputs.append(out)
    output_pc = tf.concat(outputs, axis=1)
    return output_pc

def mlp_conv(inputs, layer_dims, bn=None, bn_decay=None, is_training=None, activation_fn=tf.nn.relu):
    print('shape is {}'.format(inputs.shape))
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1, use_bias=True)

    outputs = tf.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1, use_bias=True)

    return outputs

def mlp_conv_reg(inputs, layer_dims, bn=None, weight_decay=0.00001, 
      bn_decay=None, is_training=None, activation_fn=tf.nn.relu):

    print('shape is {}'.format(inputs.shape))
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1, kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            use_bias=True)

        if bn:
            inputs = tf.layers.batch_normalization(
            inputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)

        if activation_fn is not None:
            inputs = activation_fn(inputs)

    outputs = tf.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
          bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
            kernel_size=1, use_bias=True)

    if bn:
        outputs = tf.layers.batch_normalization(
        outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)

    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs

def point_maxpool(inputs, npts, keepdims=False):
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
        for f in tf.split(inputs, npts, axis=1)]
    return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    inputs = tf.split(inputs, inputs.shape[0], axis=0)
    outputs = [tf.tile(f, [1, npts[i], 1]) for i,f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)


def chamfer(pcd1, pcd2):
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))
    return (dist1 + dist2) / 2


def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    return tf.reduce_mean(cost / num_points)


def add_train_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on 2D convolutional maps.
  
  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,1,2], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)
    
def batch_norm_dist_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ The batch normalization for distributed training.
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = _variable_on_cpu('beta', [num_channels], initializer=tf.zeros_initializer())
    gamma = _variable_on_cpu('gamma', [num_channels], initializer=tf.ones_initializer())

    pop_mean = _variable_on_cpu('pop_mean', [num_channels], initializer=tf.zeros_initializer(), trainable=False)
    pop_var = _variable_on_cpu('pop_var', [num_channels], initializer=tf.ones_initializer(), trainable=False)

    def train_bn_op():
      batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
      decay = bn_decay if bn_decay is not None else 0.9
      train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay)) 
      train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
      with tf.control_dependencies([train_mean, train_var]):
        return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, 1e-3)

    def test_bn_op():
      return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, 1e-3)

    normed = tf.cond(is_training,
                     train_bn_op,
                     test_bn_op)
    return normed


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(is_training,
                           lambda: ema.apply([batch_mean, batch_var]),
                           lambda: tf.no_op())
    
    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    
    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed

def conv2d(inputs,
          num_input_channels,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      # kernel_shape = [kernel_h, kernel_w,
      #                 num_in_channels, num_output_channels]
      kernel_shape = [kernel_h, kernel_w,
                      num_input_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn', is_dist=is_dist)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

def conv2d_reg(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv2d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def fully_connected(inputs, num_input_channels,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    is_dist=False):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_channels, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn', is_dist=is_dist)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs
    
def batch_norm_for_fc(inputs, is_training, bn_decay, scope, is_dist=False):
  """ Batch normalization on FC data.
  
  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      is_dist:     true indicating distributed training scheme
  Return:
      normed:      batch-normalized maps
  """
  if is_dist:
    return batch_norm_dist_template(inputs, is_training, scope, [0,], bn_decay)
  else:
    return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)

def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def pairwise_distance(point_cloud):
  """Compute pairwise distance of a point cloud.

  Args:
    point_cloud: tensor (batch_size, num_points, num_dims)

  Returns:
    pairwise distance: (batch_size, num_points, num_points)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  og2 = point_cloud.get_shape().as_list()[1]
  og3 = point_cloud.get_shape().as_list()[2]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)
  point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
  point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
  point_cloud_inner = -2*point_cloud_inner
  point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
  point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
  return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose

def knn(adj_matrix, k=20):
  """Get KNN based on the pairwise distance.
  Args:
    pairwise distance: (batch_size, num_points, num_points)
    k: int

  Returns:
    nearest neighbors: (batch_size, num_points, k)
  """
  neg_adj = -adj_matrix
  _, nn_idx = tf.nn.top_k(neg_adj, k=k)
  return nn_idx

def get_edge_feature(point_cloud, nn_idx, k=20):
  """Construct edge feature for each point
  Args:
    point_cloud: (batch_size, num_points, 1, num_dims)
    nn_idx: (batch_size, num_points, k)
    k: int
  Returns:
    edge features: (batch_size, num_points, k, num_dims)
  """
  og_batch_size = point_cloud.get_shape().as_list()[0]
  point_cloud = tf.squeeze(point_cloud)
  if og_batch_size == 1:
    point_cloud = tf.expand_dims(point_cloud, 0)

  point_cloud_central = point_cloud

  point_cloud_shape = point_cloud.get_shape()
  batch_size = point_cloud_shape[0].value
  num_points = point_cloud_shape[1].value
  num_dims = point_cloud_shape[2].value

  idx_ = tf.range(batch_size) * num_points
  idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

  point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
  point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
  point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

  point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

  edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
  return edge_feature