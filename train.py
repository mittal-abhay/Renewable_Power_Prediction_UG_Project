import os
import pandas as pd
import numpy as np
import tensorflow as tf
from model import GAN
from util import OneHot
from load import load_wind, load_solar_data, load_wind_data_spatial
import matplotlib.pyplot as plt
import csv
from numpy import shape

# Disable eager execution as this is a TF1.x style code
tf.compat.v1.disable_eager_execution()

def train_gan():
    # Hyperparameters
    n_epochs = 70
    learning_rate = 0.0002
    batch_size = 32
    image_shape = [24, 24, 1]
    dim_z = 100
    dim_W1 = 1024
    dim_W2 = 128
    dim_W3 = 64
    dim_channel = 1
    mu, sigma = 0, 0.1
    events_num = 5
    visualize_dim = 32
    generated_dim = 32
    max_wind_capacity = 16.0  # Normalization factor

    # Load data
    trX, trY = load_wind()
    print("Shape of training samples:", shape(trX))
    print("Training data loaded")

    # Initialize model
    dcgan_model = GAN(dim_y=events_num)
    print("GAN model initialized")

    # Build model
    Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()

    # Initialize session
    sess = tf.compat.v1.InteractiveSession()
    saver = tf.compat.v1.train.Saver(max_to_keep=10)

    # Get trainable variables
    discrim_vars = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('discrim')]
    gen_vars = [var for var in tf.compat.v1.trainable_variables() if var.name.startswith('gen')]

    # Setup optimizers
    train_op_discrim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4).minimize(
        -d_cost_tf, var_list=discrim_vars
    )
    train_op_gen = tf.compat.v1.train.RMSPropOptimizer(learning_rate=1e-4).minimize(
        g_cost_tf, var_list=gen_vars
    )

    # Setup sample generator
    Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)
    
    # Initialize variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # Training metrics
    gen_loss_all = []
    P_real = []
    P_fake = []
    discrim_loss = []
    iterations = 0
    k = 4  # Balance factor for D/G training

    # Training loop
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        
        # Shuffle data
        index = np.arange(len(trY))
        np.random.shuffle(index)
        trX = trX[index]
        trY = trY[index]
        trY2 = OneHot(trY, n=events_num)

        # Batch training
        for start, end in zip(
                range(0, len(trY), batch_size),
                range(batch_size, len(trY), batch_size)
        ):
            Xs = trX[start:end].reshape([-1, 24, 24, 1])
            Ys = trY2[start:end]
            Zs = np.random.normal(mu, sigma, size=[batch_size, dim_z]).astype(np.float32)

            # Train generator
            if np.mod(iterations, k) == 0:
                _, gen_loss_val = sess.run(
                    [train_op_gen, g_cost_tf],
                    feed_dict={Z_tf: Zs, Y_tf: Ys, image_tf: Xs}
                )
                discrim_loss_val, p_real_val, p_gen_val = sess.run(
                    [d_cost_tf, p_real, p_gen],
                    feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys}
                )
            
            # Train discriminator
            else:
                _, discrim_loss_val = sess.run(
                    [train_op_discrim, d_cost_tf],
                    feed_dict={Z_tf: Zs, Y_tf: Ys, image_tf: Xs}
                )
                gen_loss_val, p_real_val, p_gen_val = sess.run(
                    [g_cost_tf, p_real, p_gen],
                    feed_dict={Z_tf: Zs, image_tf: Xs, Y_tf: Ys}
                )

            # Record metrics
            P_real.append(p_real_val.mean())
            P_fake.append(p_gen_val.mean())
            discrim_loss.append(discrim_loss_val)

            # Generate samples periodically
            if np.mod(iterations, 1000) == 0:
                print(f"Iteration {iterations}")
                print(f"Average P(real)= {p_real_val.mean():.4f}")
                print(f"Average P(gen)= {p_gen_val.mean():.4f}")
                print(f"Discriminator loss: {discrim_loss_val:.4f}")
                
                # Generate and save samples
                Y_np_sample = OneHot(np.random.randint(5, size=[visualize_dim]), n=events_num)
                Z_np_sample = np.random.normal(mu, sigma, size=[batch_size, dim_z]).astype(np.float32)
                generated_samples = sess.run(
                    image_tf_sample,
                    feed_dict={Z_tf_sample: Z_np_sample, Y_tf_sample: Y_np_sample}
                )
                
                generated_samples = generated_samples.reshape([-1, 576])
                generated_samples = generated_samples * max_wind_capacity
                
                # Save samples
                with open(f'samples_{iterations}.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(generated_samples)

            iterations += 1

    # Generate final samples
    Y_np_sample = OneHot(np.random.randint(5, size=[visualize_dim]), n=events_num)
    Z_np_sample = np.random.normal(mu, sigma, size=[batch_size, dim_z]).astype(np.float32)
    generated_samples = sess.run(
        image_tf_sample,
        feed_dict={Z_tf_sample: Z_np_sample, Y_tf_sample: Y_np_sample}
    )
    
    generated_samples = generated_samples.reshape([-1, 576])
    generated_samples = generated_samples * max_wind_capacity

    # Save final results
    with open('final_samples.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(generated_samples)
    
    with open('final_labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Y_np_sample)

    # Plot training metrics
    plt.figure(figsize=(10, 5))
    plt.plot(P_real, label="real")
    plt.plot(P_fake, label="fake")
    plt.title("Discriminator Probabilities")
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig('discriminator_probabilities.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(discrim_loss, label="discriminator_loss")
    plt.title("Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('discriminator_loss.png')
    plt.close()

    sess.close()

if __name__ == "__main__":
    train_gan()