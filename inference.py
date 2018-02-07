from tensorflow.examples.tutorials.mnist import input_data
import model_helper
import tensorflow as tf
import vae_model
import numpy as np
import scipy
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_images(outputs, xs, hparams, output_infer):
    for i in range(outputs.shape[0]):
        out_img = outputs[i, :]
        out_img = np.reshape(out_img, (hparams.img_width, hparams.img_height))
        scipy.misc.imsave(output_infer + '/gen_img_%d_z_%d.png' % (i,hparams.z_dim), out_img)
        orig_img = xs[i, :]
        orig_img = np.reshape(orig_img, (hparams.img_width, hparams.img_height))
        scipy.misc.imsave(output_infer + '/orig_img_%d_z_%d.png' % (i,hparams.z_dim), orig_img)


def plot_zs(data,labels, output_infer):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=np.argmax(labels, 1))
    plt.colorbar()
    plt.grid()
    plt.savefig(output_infer+'/distr.png')


def inference(ckpt, inference_output_folder, hparams):
    output_infer = inference_output_folder

    # Read data
    mnist = input_data.read_data_sets("data/", one_hot=True)
    if hparams.infer_source == "test":
        infer_data = mnist.test.images
        infer_labels = mnist.test.labels
    elif hparams.infer_source == "validation":
        infer_data = mnist.validation.images
        infer_labels = mnist.validation.labels
    elif hparams.infer_source == "train":
        infer_data = mnist.train.images
        infer_labels = mnist.train.labels
    infer_sample_idxs = random.sample(range(infer_data.shape[0]), hparams.infer_sample)
    infer_sample = infer_data[infer_sample_idxs, :]
    infer_sample_labels = infer_labels[infer_sample_idxs, :]

    dataset, x_placeholder, batch_placeholder = model_helper.create_batched_dataset(hparams.x_dim)
    iterator = tf.contrib.data.Iterator.from_structure(mnist.train.images.dtype,
                                                       tf.TensorShape([None, hparams.x_dim]))
    init_op = iterator.make_initializer(dataset)
    next_x = iterator.get_next()
    dropout = tf.placeholder(tf.float32, name='dropout')
    model = vae_model.FullyConnectedVAE(hparams, next_x,dropout)

    with tf.Session() as session:
        model = model_helper.load_model(model, ckpt, session)
        session.run(init_op, feed_dict={x_placeholder: infer_sample,
                                        batch_placeholder: infer_sample.shape[0],
                                        dropout:0.0})
        # Decode
        outputs, zs, xs = model.infer(session)
        plot_images(outputs, xs, hparams, output_infer)
        if zs.shape[1] == 2:
            plot_zs(zs, infer_sample_labels, output_infer)
