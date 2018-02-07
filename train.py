import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import vae_model
import time
import os
import model_helper
import numpy as np


def train(hparams):
    num_epochs = hparams.num_epochs
    num_ckpt_epochs = hparams.num_ckpt_epochs
    summary_name = "train_log"
    out_dir = hparams.out_dir

    mnist = input_data.read_data_sets("data/", one_hot=True)

    train_dataset,train_x_placeholder,train_batch_placeholder = model_helper.create_batched_dataset(hparams.x_dim)
    dev_dataset, dev_x_placeholder, dev_batch_placeholder = model_helper.create_batched_dataset(hparams.x_dim)

    iterator = tf.contrib.data.Iterator.from_structure(mnist.train.images.dtype,
                                                       tf.TensorShape([None, hparams.x_dim]))
    train_init_op = iterator.make_initializer(train_dataset)
    dev_init_op = iterator.make_initializer(dev_dataset)
    next_x = iterator.get_next()

    dropout = tf.placeholder(tf.float32, name='dropout')
    model = vae_model.FullyConnectedVAE(hparams, next_x, dropout)

    session = tf.Session()
    summary_writer = tf.summary.FileWriter(
        os.path.join(out_dir, summary_name), session.graph)
    model, global_step = model_helper.create_or_load_model(model, out_dir, session)
    dev_feed_dict = {dev_x_placeholder: mnist.validation.images,
                     dev_batch_placeholder: hparams.batch_size,
                     dropout: 0.0}
    dev_loss = run_evaluation(model, session, dev_init_op, dev_feed_dict)

    print(
        "  global step %d ""val_loss %.3f" % (0, dev_loss))
    start_train_time = time.time()
    epoch_time = 0.0
    step_loss, epoch_loss = 0.0, 0.0
    steps_count = 0.0
    # train the model for num_epochs steps
    train_losses=[]
    val_losses=[]
    for epoch in range(num_epochs):
        # Initialize the train dataset in order to proceed to the next epoch.
        session.run(train_init_op, feed_dict={train_x_placeholder: mnist.train.images,
                                              train_batch_placeholder: hparams.batch_size,
                                              dropout: hparams.dropout})
        #go through all batches for the current epoch
        while True:
            start_step_time = time.time()
            try:
                # run a train step with the next batch
                step_result = model.train(session)
                (_, step_loss, step_summary, global_step, xx, targets, output) = step_result
                # Write step summary.
                summary_writer.add_summary(step_summary, global_step)
                # Update step time and step loss
                epoch_time += (time.time() - start_step_time)
                epoch_loss += step_loss
                steps_count += 1.0
            except tf.errors.OutOfRangeError:
                break
        # average epoch loss over all batches
        epoch_loss /= steps_count
        epoch_time /= steps_count
        # print results if the current epoch is a checkpoint epoch
        if (epoch+1) % num_ckpt_epochs == 0:
            print("## Save checkpoint: ")
            # save checkpoint
            model_helper.add_summary(summary_writer, global_step, "train_loss", epoch_loss)
            model.saver.save(session, os.path.join(out_dir, "vae.ckpt"), global_step=epoch)
            print("## Evaluation: ")
            dev_feed_dict = {dev_x_placeholder: mnist.validation.images,
                     dev_batch_placeholder: hparams.batch_size,dropout: 0.0}
            dev_loss = run_evaluation(model, session, dev_init_op, dev_feed_dict)
            print(
                "  epoch %d lr %g "
                "train_loss %.3f val_loss %.3f" %
                (epoch, model.learning_rate.eval(session=session), epoch_loss, dev_loss))
            train_losses.append(epoch_loss)
            val_losses.append(dev_loss)

        # Reset timer and loss.
        epoch_time, epoch_loss, steps_count = 0.0, 0.0, 0.0
    # save final model
    model.saver.save(session, os.path.join(out_dir, "vae.ckpt"), global_step=epoch)
    # print final results
    print(
        "# Final, epoch %d lr %g "
        "step-time %.2fK train loss %.3f, %s" % (num_epochs,
                                                 model.learning_rate.eval(session=session), epoch_time, epoch_loss,
                                                 time.ctime()))
    print("# Done training!", time.time() -start_train_time)
    min_val_loss=np.min(val_losses)
    min_val_idx=np.argmin(val_losses)
    print("Min val loss: %f at epoch: %d"%(min_val_loss,min_val_idx))
    summary_writer.close()


def run_evaluation(model, session, iterator_init_op, dev_feed_dict):
    # model, global_step = create_or_load_model(model, out_dir, session)
    session.run(iterator_init_op, feed_dict=dev_feed_dict)
    dev_loss = model_helper.compute_loss(model, session)
    return dev_loss


def run_inference(model, session, iterator_init_op, feed_dict):
    session.run(iterator_init_op, feed_dict=feed_dict)
    outputs=model.infer(session)
    return outputs