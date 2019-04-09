from vae_gambel import PARAMS, GambelVAE, gumbel_loss
import tensorflow as tf 
tf.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt

HARD = False
Filename = "vae_pic/"

# Model
model = GambelVAE(PARAMS)
logits_y, logits_x = model(tf.convert_to_tensor(np.random.random((PARAMS['batch_size'], PARAMS['data_dim'])), dtype=tf.float32), tau=1.0, hard=False)

print("Load weights...")
if HARD == False:
    model.load_weights("model.h5")
else:
    model.load_weights("model_hard.h5")
print("done.")

# 可视化VAE重建结果
(x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Model Forward
logits_y, logits_x = model(x_train[:100], tau=0.5, hard=HARD)
sample_y = model.sample_y.numpy()                # shape=(100,30,10)
logits_x = tf.sigmoid(logits_x).numpy()          # shape=(100,784)
print("sample_y: ", sample_y.shape, ", logits_x.shape:", logits_x.shape)
code = model.sample_y


def save_plt(original_img, construct_img, code):
    plt.figure(figsize=(10, 30))
    for i in range(0, 15, 3):
        # input img
        plt.subplot(5, 3, i+1)
        plt.imshow(original_img[i, :].reshape(28, 28), cmap='gray')
        plt.axis('off')

        # code
        plt.subplot(5, 3, i+2)
        plt.imshow(code[i, :, :], cmap='gray')
        plt.axis('off')

        # output img
        plt.subplot(5, 3, i+3)
        plt.imshow(construct_img[i, :,].reshape((28, 28)), cmap='gray')
        plt.axis('off')

    plt.savefig('vae-pic/vae_rebuilt.png')

save_plt(x_train[:100], logits_x, code)


# T-SNE 对6000张输入图片得到的 编码 进行降维处理
def save_embed():
    C = np.zeros((6000, PARAMS["M"]*PARAMS["N"]))
    for i in range(0, 6000, 100):
        logits_y, _ = model(x_test[i: i+100], tau=0.5, hard=HARD)
        code = tf.reshape(tf.reshape(logits_y, [-1, 30, 10]), [-1,300])
        # code = tf.reshape(model.sample_y, (100, PARAMS["M"]*PARAMS["N"]))  # (100, 300)
        C[i: i+100] = code

    from sklearn.manifold.t_sne import TSNE
    tsne = TSNE()
    viz = tsne.fit_transform(C)

    color = ['aliceblue', 'cyan', 'darkorange', 'fuchsia', 'lightpink', 
             'pink', 'springgreen', 'yellow', 'orange', 'mediumturquoise']
    for i in range(0, 6000):
        plt.scatter(viz[i, 0], viz[i, 1], c=color[y_test[i]])
    plt.savefig('vae-pic/vae_embed.png')

# save_embed()
