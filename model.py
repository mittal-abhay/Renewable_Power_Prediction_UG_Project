import tensorflow as tf

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean(tf.square(X-mean), [0,1,2])
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

class GAN(tf.keras.Model):
    def __init__(
            self,
            batch_size=32,
            image_shape=[24,24,1],
            dim_z=100,
            dim_y=6,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            ):
        super(GAN, self).__init__()
        
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        # Generator weights
        self.gen_W1 = tf.Variable(tf.random.normal([dim_z+dim_y, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random.normal([dim_W1+dim_y, dim_W2*6*6], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random.normal([5,5,dim_W3,dim_W2+dim_y], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random.normal([5,5,dim_channel,dim_W3+dim_y], stddev=0.02), name='gen_W4')

        # Discriminator weights
        self.discrim_W1 = tf.Variable(tf.random.normal([5,5,dim_channel+dim_y,dim_W3], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random.normal([5,5,dim_W3+dim_y,dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random.normal([dim_W2*6*6+dim_y,dim_W1], stddev=0.02), name='discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random.normal([dim_W1+dim_y,1], stddev=0.02), name='discrim_W4')

    def build_model(self):
        """Build the GAN model and return necessary tensors"""
        # Input placeholders
        Z = tf.compat.v1.placeholder(tf.float32, [None, self.dim_z])
        Y = tf.compat.v1.placeholder(tf.float32, [None, self.dim_y])
        image = tf.compat.v1.placeholder(tf.float32, [None] + self.image_shape)

        # Generate fake samples
        G = self.generate(Z, Y)
        G = tf.nn.sigmoid(G)  # Apply sigmoid to generator output

        # Discriminator outputs for real and fake samples
        D_real = self.discriminate(image, Y)
        D_fake = self.discriminate(G, Y)

        # Calculate probabilities
        p_real = tf.nn.sigmoid(D_real)
        p_fake = tf.nn.sigmoid(D_fake)

        # Define losses using WGAN approach
        d_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
        g_loss = -tf.reduce_mean(D_fake)

        return Z, Y, image, d_loss, g_loss, p_real, p_fake

    def generate(self, Z, Y):
        """Generator function"""
        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat([Z, Y], 1)
        
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.concat([h1, Y], 1)
        
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size, 6, 6, self.dim_W2])
        h2 = tf.concat([h2, yb*tf.ones([self.batch_size, 6, 6, self.dim_y])], 3)

        output_shape_l3 = [self.batch_size, 12, 12, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu(batchnormalize(h3))
        h3 = tf.concat([h3, yb*tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)

        output_shape_l4 = [self.batch_size, 24, 24, self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        return h4

    def discriminate(self, image, Y):
        """Discriminator function"""
        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        X = tf.concat([image, yb * tf.ones([self.batch_size, 24, 24, self.dim_y])], 3)
        
        h1 = lrelu(tf.nn.conv2d(X, self.discrim_W1, strides=[1,2,2,1], padding='SAME'))
        h1 = tf.concat([h1, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)
        
        h2 = lrelu(batchnormalize(tf.nn.conv2d(h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')))
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat([h2, Y], 1)
        
        h3 = lrelu(batchnormalize(tf.matmul(h2, self.discrim_W3)))
        h3 = tf.concat([h3, Y], 1)
        return tf.matmul(h3, self.discrim_W4)

    def samples_generator(self, batch_size):
        """Setup the sampling operation for generating samples"""
        Z = tf.compat.v1.placeholder(tf.float32, [None, self.dim_z])
        Y = tf.compat.v1.placeholder(tf.float32, [None, self.dim_y])
        
        with tf.compat.v1.variable_scope('gen', reuse=True):
            samples = self.generate(Z, Y)
            samples = tf.nn.sigmoid(samples)
        
        return Z, Y, samples