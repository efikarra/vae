import argparse
import sys
import train
import tensorflow as tf
import utils
import inference
import os


FLAGS=None


def create_hparams(flags):
    return tf.contrib.training.HParams(
        # data
        out_dir=flags.out_dir,
        # network
        hidden_dim=flags.hidden_dim,
        img_width=flags.img_width,
        img_height=flags.img_height,
        z_dim=flags.z_dim,
        # training
        batch_size=flags.batch_size,
        num_epochs=flags.num_epochs,
        num_ckpt_epochs=flags.num_ckpt_epochs,
        dropout=flags.dropout,
        # optimizer
        learning_rate=flags.learning_rate,
        optimizer=flags.optimizer,
        start_decay_step=flags.start_decay_step,
        decay_steps=flags.decay_steps,
        decay_factor=flags.decay_factor,
        ckpt=flags.ckpt,
        #inference
        inference_output_folder=flags.inference_output_folder,
        infer_source=flags.infer_source,
        infer_sample=flags.infer_sample,
        gpu=flags.gpu
    )


def create_or_load_hparams(out_dir,default_hparams,flags):
    """Create hparams or load hparams from out_dir."""
    hparams = utils.load_hparams(out_dir)
    if not hparams:
        hparams = default_hparams
        hparams = utils.maybe_parse_standard_hparams(
            hparams,flags.hparams_path)
        hparams.add_hparam("x_dim", hparams.img_width * hparams.img_height)
    else:
        hparams = utils.ensure_compatible_hparams(hparams,default_hparams,flags)

    # Save HParams
    utils.save_hparams(out_dir,hparams)

    # Print HParams
    utils.print_hparams(hparams)
    return hparams


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument("--gpu", type=int, default=0,
                        help="Gpu machine to run the code (if gpus available)")
    # data
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")

    # network
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden representation size.")
    parser.add_argument("--img_width", type=int, default=28,
                        help="Image width.")
    parser.add_argument("--img_height", type=int, default=28,
                        help="Image width.")
    parser.add_argument("--z_dim", type=int, default=2, help="z size.")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout.")
    # training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--num_ckpt_epochs", type=int, default=2,
                        help="Number of epochs until the next checkpoint saving.")
    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--start_decay_step", type=int, default=0,
                        help="When we start to decay")
    parser.add_argument("--decay_steps", type=int, default=10000,
                        help="How frequent we decay")
    parser.add_argument("--decay_factor", type=float, default=0.98,
                        help="How much we decay.")

    parser.add_argument("--hparams_path",type=str,default=None,
                            help=("Path to standard hparams json file that overrides"
                                  "hparams values from flags."))
    parser.add_argument("--inference_output_folder", type=str, default=False,
                            help="Output folder to save inference data.")
    parser.add_argument("--ckpt", type=str, default=False,
                                help="Checkpoint file.")
    parser.add_argument("--infer_sample", type=int, default=10,
                        help="Sample size to perform inference.")
    parser.add_argument("--infer_source", type=str, default="test",
                        help="Source data to perform inference. test|train|validation, default=test")

def main(unused_argv):
    default_hparams = create_hparams(FLAGS)
    out_dir = FLAGS.out_dir
    if not tf.gfile.Exists(out_dir): tf.gfile.MakeDirs(out_dir)
    hparams = create_or_load_hparams(out_dir, default_hparams, FLAGS)
    # Check out_dir
    if not tf.gfile.Exists(hparams.out_dir):
        utils.print_out("# Creating output directory %s ..." % hparams.out_dir)
        tf.gfile.MakeDirs(hparams.out_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hparams.gpu)
    if FLAGS.inference_output_folder:
        # Inference
        trans_file = FLAGS.inference_output_folder
        ckpt = FLAGS.ckpt
        if not ckpt:
            ckpt = tf.train.latest_checkpoint(out_dir)
        inference.inference(ckpt,trans_file,hparams)
    else:
        train.train(hparams)


if __name__ == '__main__':
    vae_parser = argparse.ArgumentParser()
    add_arguments(vae_parser)
    FLAGS,unparsed=vae_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
