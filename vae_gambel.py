# 参考  https://github.com/ericjang/gumbel-softmax

import numpy as np
from keras import objectives
from keras.objectives import binary_crossentropy as bce
import tensorflow as tf
layers = tf.keras.layers

opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
conf = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config=conf)

PARAMS = {
    "batch_size": 100,
    "data_dim": 784,
    "M": 10,
    "N": 30,
    "nb_epoch": 100, 
    "epsilon_std": 0.01,
    "anneal_rate": 0.0003,
    "init_temperature": 5.0,
    "min_temperature": 0.5,
    "learning_rate": 1e-3,
    "hard": True,
}


class GambelVAE(tf.keras.Model):
    def __init__(self, params):
        super(GambelVAE, self).__init__()
        self.params = params
        
        # encoder
        self.enc_dense1 = layers.Dense(512, activation='relu')
        self.enc_dense2 = layers.Dense(256, activation='relu')
        self.enc_dense3 = layers.Dense(params["N"]*params["M"])

        # decoder
        self.flatten = layers.Flatten()
        self.dec_dense1 = layers.Dense(256, activation='relu')
        self.dec_dense2 = layers.Dense(512, activation='relu')
        self.dec_dense3 = layers.Dense(params["data_dim"])


    def sample_gumbel(self, shape, eps=1e-20): 
        """Sample from Gumbel(0, 1)"""
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)


    def gumbel_softmax_sample(self, logits, temperature): 
        """ Draw a sample from the Gumbel-Softmax distribution"""
        # logits: [batch_size, n_class] unnormalized log-probs
        y = logits + self.sample_gumbel(tf.shape(logits))
        return tf.nn.softmax( y / temperature)   # 每行之和为1

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        """
        # 返回值y.shape=(batchsize, n_class), 每行之和为1，每个数代表概率
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard: 
            # 将 y 转成one-hot向量，每一行最大值处为1，其余地方为0
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y       # y_hard = y
        return y

    def decoder(self, x):
        # decoder
        h = self.flatten(x)
        h = self.dec_dense1(h)
        h = self.dec_dense2(h)
        h = self.dec_dense3(h)
        return h

    def call(self, x, tau, hard=False):
        N = self.params["N"]   # 30
        M = self.params["M"]   # 10

        # encoder
        x = self.enc_dense1(x)
        x = self.enc_dense2(x)
        x = self.enc_dense3(x)   # (batch, N*M)
        logits_y = tf.reshape(x, [-1, M])   # (batshsize*30, 10)

        # sample
        # 加入Sample中的随机过程之后，返回向量 (batchsize*N, 10). 最后2维代表概率
        y = self.gumbel_softmax(logits_y, tau, hard=hard)
        assert y.shape == (self.params["batch_size"]*N, M)
        y = tf.reshape(y, [-1, N, M])
        self.sample_y = y

        # decoder
        logits_x = self.decoder(y)                  # (batch, 28*28)
        return logits_y, logits_x


def gumbel_loss(model, x, tau, hard):
    M = PARAMS["M"]   # 10
    N = PARAMS["N"]   # 30
    data_dim = PARAMS['data_dim']
    logits_y, logits_x = model(x, tau, hard)
    
    # cross-entropy
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits_x)
    cross_ent = tf.reduce_sum(cross_ent, 1)
    cross_ent = tf.reduce_mean(cross_ent, 0)
    
    # KL loss
    q_y = tf.nn.softmax(logits_y)   # (batshsize*30, 10)  softmax之后
    log_q_y = tf.log(q_y + 1e-20)   # (batshsize*30, 10)  转成log概率之后
    kl_tmp = tf.reshape(q_y*(log_q_y-tf.log(1.0/M)), [-1,N,M])  # (batch_size,N,K)
    KL = tf.reduce_sum(kl_tmp, [1, 2])    # shape=(batch_size, 1)
    KL_mean = tf.reduce_mean(KL)
    # print("**", cross_ent.numpy(), KL_mean.numpy())
    return cross_ent + KL_mean


def compute_gradients(model, x, tau, hard):
    with tf.GradientTape() as tape:
        loss = gumbel_loss(model, x, tau, hard)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(optimizer, gradients, variables, global_step):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


def get_learning_rate(step, init=PARAMS["learning_rate"]):
    return tf.convert_to_tensor(init * pow(0.95, (step / 1000.)), dtype=tf.float32)


if __name__ == '__main__':
    model = GambelVAE(PARAMS)
    learning_rate = tf.Variable(PARAMS["learning_rate"], trainable=False, name="LR")
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    TRAIN_BUF = 60000
    BATCH_SIZE = 100
    TEST_BUF = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(TEST_BUF).batch(BATCH_SIZE)

    # temperature
    tau = PARAMS["init_temperature"]
    anneal_rate = PARAMS["anneal_rate"]
    min_temperature = PARAMS["min_temperature"]

    # Train
    global_step = tf.train.get_or_create_global_step()
    for epoch in range(1, PARAMS["nb_epoch"] + 1):
        tau = np.maximum(tau * np.exp(-anneal_rate*epoch), min_temperature)

        for train_x in train_dataset:
            gradients, loss = compute_gradients(model, train_x, tau, hard=PARAMS["hard"])
            apply_gradients(optimizer, gradients, model.trainable_variables, global_step)
            
            # change lr
            new_lr = get_learning_rate(global_step.numpy())
            learning_rate.assign(new_lr)

        print("Epoch:", epoch, ", TRAIN loss:", loss.numpy(), ", Temperature:", tau)
            
        if epoch % 1 == 0:
            losses = []
            for test_x in test_dataset:
                losses.append(gumbel_loss(model, test_x, tau, hard=PARAMS["hard"]).numpy())
            eval_loss = np.mean(losses)
            print("      Eval Loss:", eval_loss, "\n")

        if PARAMS['hard'] == True:
            model.save_weights("model.h5")
        else:
            model.save_weights("model_hard.h5")

