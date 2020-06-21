import tensorflow as tf

rs = tf.test.is_built_with_cuda()

print(rs)
tf.config.list_physical_devices('GPU')