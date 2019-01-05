
This repository implements a Variational Autoencoder model in Tensorflow. The code applies the model on MNIST data.

The code is written in Python 3.6 and Tensorflow 1.4.

<h3> How to run the code </h3>
In order to train a model in the MNIST data, run the following command:

*python main.py --out_dir=model --hidden_dim=500 --img_width=28 --img_height=28 
--z_dim=2 --optimizer=adam --learning_rate=0.001 --batch_size=128 --num_epochs=60 --num_ckpt_epochs=1*

In order to plot the original and reconstructed images for a sample of the test data, run the following command:

*python main.py --out_dir=model --inference_output_folder=out_infer --infer_sample=40 --ckpt=$CHECKPOINT_PATH*
