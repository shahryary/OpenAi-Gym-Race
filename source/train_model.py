import tensorflow as tf
import numpy as np
import glob, random, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_path = "saved_models/"
model_name = model_path + 'model'


# using tensorflow as back-end 
class NetworkModel(object):
    # Create model
    def __init__(self):
        self.image = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [64, 64])
        tf.summary.image('resized_image', self.resized_image, 20)

        self.mulZ, self.z_logvar = self.encoder(self.resized_image)
        self.z = self.sample_z(self.mulZ, self.z_logvar)
        self.recons = self.decoder(self.z)
        tf.summary.image('recons', self.recons, 20)

        self.merged = tf.summary.merge_all()

        self.loss = self.com_loss()

    def sample_z(self, mu, logvar):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(logvar / 2) * eps

    def encoder(self, x):
        encod_x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        encod_x = tf.layers.conv2d(encod_x, filters=64, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        encod_x = tf.layers.conv2d(encod_x, filters=128, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)
        encod_x = tf.layers.conv2d(encod_x, filters=256, kernel_size=4, strides=2, padding='valid', activation=tf.nn.relu)

        encod_x = tf.layers.flatten(encod_x)
        mulZ = tf.layers.dense(encod_x, units=32, name='mulZ')
        z_logvar = tf.layers.dense(encod_x, units=32, name='z_logvar')
        return mulZ, z_logvar

    def decoder(self, z):
        decod_z = tf.layers.dense(z, 1024, activation=None)
        decod_z = tf.reshape(decod_z, [-1, 1, 1, 1024])
        decod_z = tf.layers.conv2d_transpose(decod_z, filters=128, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        decod_z = tf.layers.conv2d_transpose(decod_z, filters=64, kernel_size=5, strides=2, padding='valid', activation=tf.nn.relu)
        decod_z = tf.layers.conv2d_transpose(decod_z, filters=32, kernel_size=6, strides=2, padding='valid', activation=tf.nn.relu)
        decod_z = tf.layers.conv2d_transpose(decod_z, filters=3, kernel_size=6, strides=2, padding='valid', activation=tf.nn.sigmoid)
        return decod_z

    def com_loss(self):
        log_flat = tf.layers.flatten(self.recons)
        lab_flat = tf.layers.flatten(self.resized_image)
        recon_loss = tf.reduce_sum(tf.square(log_flat - lab_flat), axis = 1)
        kl_loss = 0.5 * tf.reduce_sum(tf.exp(self.z_logvar) + self.mulZ**2 - 1. - self.z_logvar, 1)
        vae_loss = tf.reduce_mean(recon_loss + kl_loss)
        return vae_loss

def feed_data(batch_size):
    data_files = glob.glob('data/TR_data_*')
    while True:
        data = np.load(random.sample(data_files, 1)[0])
        np.random.shuffle(data)
        np.random.shuffle(data)
        N = data.shape[0]
        start = np.random.randint(0, N-batch_size)
        yield data[start:start+batch_size]

def train_model():
    sess = tf.InteractiveSession()

    global_track = tf.Variable(0, name='global_step', trainable=False)

    writer = tf.summary.FileWriter('logdir')

    model = NetworkModel()
    print(model)
    train_op = tf.train.AdamOptimizer(0.001).minimize(model.loss, global_step=global_track)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=1)
    step = global_track.eval()
    training_data = feed_data(batch_size=128)

    try:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Model restored from: {}".format(model_path))
    except:
        print("Could not restore saved model")

    try:
        while True:
            images = next(training_data)
            _, loss_value, summary = sess.run([train_op, model.loss, model.merged],
                                feed_dict={model.image: images})
            writer.add_summary(summary, step)

            if np.isnan(loss_value):
                raise ValueError('Loss value is NaN')
            if step % 10 == 0 and step > 0:
                print ('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_track)
            if loss_value <= 27:
                print ('step {}: training loss {:.6f}'.format(step, loss_value))
                save_path = saver.save(sess, model_name, global_step=global_track)
                break
            step+=1

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")

    except Exception as e:
        print("Exception: {}".format(e))

def load_trian():

    gfgraph = tf.Graph()
    with gfgraph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=gfgraph)

        model = NetworkModel()
        print(model)
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        training_data = feed_data(batch_size=128)

        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except:
            raise ImportError("Could not restore saved model")

        return sess, model

if __name__ == '__main__':
    train_model()