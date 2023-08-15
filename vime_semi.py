"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_semi.py
- Semi-supervised learning parts of the VIME framework
- Using both labeled and unlabeled data to train the predictor with the help of trained encoder
"""

# Necessary packages
import keras
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.keras import layers as contrib_layers

from vime_utils import mask_generator, pretext_generator


def vime_semi(x_train, y_train, x_unlab, x_test, parameters, 
              p_m, K, beta, file_name, encoder_output_dim):
  """Semi-supervied learning part in VIME.
  
  Args:
    - x_train, y_train: training dataset
    - x_unlab: unlabeled dataset
    - x_test: testing features
    - parameters: network parameters (hidden_dim, batch_size, iterations)
    - p_m: corruption probability
    - K: number of augmented samples
    - beta: hyperparameter to control supervised and unsupervised loss
    - file_name: saved filed name for the encoder function
    
  Returns:
    - y_test_hat: prediction on x_test
  """
  graph = tf1.Graph()
  with graph.as_default():      
      # Network parameters
      hidden_dim = parameters['hidden_dim']
      import tensorflow as tf

      act_fn = tf.nn.relu
      batch_size = parameters['batch_size']
      iterations = parameters['iterations']

      # Basic parameters
      data_dim = encoder_output_dim
      label_dim = len(y_train[0, :])

      # Divide training and validation sets (9:1)
      idx = np.random.permutation(len(x_train[:, 0]))
      train_idx = idx[:int(len(idx)*0.9)]
      valid_idx = idx[int(len(idx)*0.9):]

      x_valid = x_train[valid_idx, :]
      y_valid = y_train[valid_idx, :]

      x_train = x_train[train_idx, :]
      y_train = y_train[train_idx, :]  

      # Input placeholder
      # Labeled data

      import tensorflow as tf
      tf1.compat.v1.disable_v2_behavior()


      x_input = tf1.placeholder(tf.float32, [None, data_dim])
      y_input = tf1.placeholder(tf.float32, [None, label_dim])

      # Augmented unlabeled data
      xu_input = tf1.placeholder(tf.float32, [None, None, data_dim])

      ## Predictor
      def predictor(input_shape):
        """Returns prediction model.

        Args: 
          - input_shape: Shape of the input feature

        Returns:
          - model: Keras model that outputs both logit and softmax predictions
        """
        model_input = tf.keras.layers.Input(shape=input_shape)

        # Stacks multi-layered perceptron
        inter_layer = tf.keras.layers.Dense(hidden_dim, activation='relu')(model_input)
        inter_layer = tf.keras.layers.Dense(hidden_dim, activation='relu')(inter_layer)

        y_hat_logit = tf.keras.layers.Dense(label_dim, activation=None)(inter_layer)
        y_hat = tf.keras.layers.Softmax()(y_hat_logit)

        model = tf.keras.models.Model(inputs=model_input, outputs=[y_hat_logit, y_hat])
        return model


      # Build model
      pred_model = predictor((data_dim,))

      # Since `pred_model` is a Keras model, when you invoke it, it's returning a list of two items ([y_hat_logit, y_hat])
      # Therefore, you need to destructure this output.
      y_hat_logit, y_hat = pred_model(x_input)

      yv_hat_logit, yv_hat = pred_model(xu_input)

      # Defin losses
      # Supervised loss
      y_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_input, y_hat_logit)

      # Unsupervised loss
      yu_loss = tf.reduce_mean(tf.nn.moments(yv_hat_logit, axes = 0)[1])

    #   print("All variable names:")
    #   for var in tf1.global_variables():
    #     print(var.name)

      # Define variables
      p_vars = [v for v in tf1.trainable_variables() if v.name.startswith('dense')]

      # Define solver
      solver = tf1.train.AdamOptimizer().minimize(y_loss + beta * yu_loss, var_list=p_vars)


      # Load encoder from self-supervised model
      encoder = keras.models.load_model(file_name)

      x_train_encoded = encoder.predict(x_train)
      x_unlab_encoded = encoder.predict(x_unlab)
      x_test_encoded = encoder.predict(x_test)

    #   print("x_train_encoded shape:", x_train_encoded.shape)
    #   print("x_unlab_encoded shape:", x_unlab_encoded.shape)
    #   print("x_test_encoded shape:", x_test_encoded.shape)

      # Encode validation and testing features
      x_valid = encoder.predict(x_valid)  
      x_test = encoder.predict(x_test)

      # Start session
      sess = tf1.Session()
      sess.run(tf1.global_variables_initializer())

      # Setup early stopping procedure
      class_file_name = './save_model/class_model.ckpt'
      saver = tf1.train.Saver(p_vars)

      yv_loss_min = 1e10
      yv_loss_min_idx = -1

      # Training iteration loop
      for it in range(iterations):

        # Select a batch of labeled data
        batch_idx = np.random.permutation(len(x_train[:, 0]))[:batch_size]
        x_batch = x_train[batch_idx, :]
        y_batch = y_train[batch_idx, :]    
    #     print(x_batch.shape)
        # Encode labeled data
    #     x_batch1 = x_batch 
        x_batch = encoder.predict(x_batch)  
    #     x_batch_encoded = encoder.predict(x_batch1)

    #     print(x_batch_encoded.shape)


        # Select a batch of unlabeled data
        batch_u_idx = np.random.permutation(len(x_unlab[:, 0]))[:batch_size]
        xu_batch_ori = x_unlab[batch_u_idx, :]

        # Augment unlabeled data
        xu_batch = list()

        for rep in range(K):      
          # Mask vector generation
          m_batch = mask_generator(p_m, xu_batch_ori)
          # Pretext generator
          _, xu_batch_temp = pretext_generator(m_batch, xu_batch_ori)

          # Encode corrupted samples
          xu_batch_temp = encoder.predict(xu_batch_temp)
          xu_batch = xu_batch + [xu_batch_temp]
        # Convert list to matrix
        xu_batch = np.asarray(xu_batch)

    #     print(x_input.get_shape())
    #     print(x_batch.shape)

    #     print(x_input.name)


        # Train the model
        _, y_loss_curr = sess.run([solver, y_loss], 
                                  feed_dict={x_input: x_batch, y_input: y_batch, 
                                             xu_input: xu_batch})  
        # Current validation loss
        yv_loss_curr = sess.run(y_loss, feed_dict={x_input: x_valid, 
                                                   y_input: y_valid})

        if it % 100 == 0:
          print('Iteration: ' + str(it) + '/' + str(iterations) + 
                ', Current loss: ' + str(np.round(yv_loss_curr, 4)))      

        # Early stopping & Best model save
        if yv_loss_min > yv_loss_curr:
          yv_loss_min = yv_loss_curr
          yv_loss_min_idx = it

          # Saves trained model
          saver.save(sess, class_file_name)

        if yv_loss_min_idx + 100 < it:
          break

      #%% Restores the saved model

      imported_graph = tf1.train.import_meta_graph(class_file_name + '.meta')

#       sess = tf1.Session()
      with tf1.Session(graph=graph) as sess:
            
          imported_graph.restore(sess, class_file_name)

          # Predict on x_test
          y_test_hat = sess.run(y_hat, feed_dict={x_input: x_test})

      return y_test_hat
