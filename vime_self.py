"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

vime_self.py
- Self-supervised learning parts of the VIME framework
- Using unlabeled data to train the encoder
"""
import tensorflow as tf

# Necessary packages
from keras.layers import Input, Dense
from keras.models import Model
from keras import models
from tensorflow.keras.layers import Lambda
from vime_utils import mask_generator, pretext_generator
import numpy as np

from tensorflow.keras.layers import Layer
import tensorflow as tf

class ImputedValuesLayer(Layer):
    def __init__(self, imputed_indices, batch_size, **kwargs):
        self.imputed_indices = imputed_indices
        self.batch_size = batch_size
        super(ImputedValuesLayer, self).__init__(**kwargs)
        
    def call(self, x):
        current_batch_size = tf.shape(x)[0]
        batched_indices = self.imputed_indices % current_batch_size  # Adjust indices based on the batch size
        rows = batched_indices[:, 0]
        cols = batched_indices[:, 1]
        return tf.gather_nd(x, tf.stack([rows, cols], axis=1))

def get_all_activations(model, x_unlab):
    """Retrieve activations from all layers of a model.
    
    Args:
    - model: Keras model object.
    - x_unlab: Input data to get activations for.
    
    Returns:
    - A dictionary where keys are layer names and values are activations for the given data.
    """
    layer_outputs = [layer.output for layer in model.layers if not layer.name.startswith('input')]  # Skip input layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(x_unlab)
    
    if not isinstance(activations, list):  # In case the model has only one layer apart from input
        activations = [activations]
        
    return {layer.name: activation for layer, activation in zip(model.layers[1:], activations)}  # Skip input layer in layer names


def get_embeddings(encoder, x_unlab):
    """Obtain embeddings (latent representations) of unlabeled data.
    
    Args:
    - encoder: Trained encoder model.
    - x_unlab: Unlabeled data.
    
    Returns:
    - embeddings: Latent representations of the input data.
    """
    embeddings = encoder.predict(x_unlab)
    return embeddings

from tensorflow.keras import Input, Model, models
from tensorflow.keras.layers import Dense

def vime_self_fnn(x_unlab, p_m, alpha, parameters):
    """Self-supervised learning part in VIME with FNN architecture.
    
    Args:
      x_unlab: unlabeled feature
      p_m: corruption probability
      alpha: hyper-parameter to control the weights of feature and mask losses
      parameters: epochs, batch_size
      
    Returns:
      encoder: Representation learning block
    """
      
    # Parameters
    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    
    # Build model  
    inputs = Input(shape=(dim,))
    
    # Encoder with FNN architecture
    h = Dense(int(dim), activation='relu')(inputs)
    h = Dense(int(dim/2), activation='relu')(h)  # Additional dense layer
    h = Dense(int(dim/4), activation='relu')(h)  # Another additional dense layer
    
    # Mask estimator
    output_1 = Dense(dim, activation='sigmoid', name = 'mask')(h)  
    
    # Feature estimator
    output_2 = Dense(dim, activation='sigmoid', name = 'feature')(h)
    
    model = Model(inputs = inputs, outputs = [output_1, output_2])
    
    model.compile(optimizer='rmsprop',
                  loss={'mask': 'binary_crossentropy', 
                        'feature': 'mean_squared_error'},
                  loss_weights={'mask': 1, 'feature': alpha})
    
    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
    
    # Fit model on unlabeled data
    model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, 
              epochs = epochs, batch_size= batch_size)
        
    # Extract encoder part
    encoder_layers = model.layers[1:4]  # Extracting the FNN layers
    layer_outputs = [layer.output for layer in encoder_layers]
    encoder = models.Model(inputs=model.input, outputs=layer_outputs[-1])
    embeddings = get_embeddings(encoder, x_unlab)
    all_activations = get_all_activations(encoder, x_unlab)

    return encoder, embeddings, all_activations

from keras.layers import Dense, Input
from keras.models import Model



def get_encoder(x_unlab, architecture='default', p_m=None, alpha=None, parameters=None):
    """
    Flexible function to generate different encoder architectures and train them.

    Args:
    - x_unlab: The unlabeled feature data
    - architecture: The type of architecture ('default' or 'fnn')
    - p_m: corruption probability
    - alpha: Hyper-parameter to control weights of feature and mask losses
    - parameters: Other parameters such as epochs and batch size

    Returns:
    - encoder: The encoder model
    - embeddings: The embeddings from the encoder
    - all_activations: All activations from the encoder
    - encoder_output_dim: The dimensionality of the encoder's output
    """

    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']

    # Common input layer for all architectures
    inputs = Input(shape=(dim,))

    # Depending on the architecture, build the model
    if architecture == 'default':
        h_encoded = Dense(int(dim), activation='relu')(inputs)  # Renamed h to h_encoded
        encoder_output_dim = int(dim)  # Set output dimension for default architecture
    
    elif architecture == 'fnn':
        h = Dense(int(dim), activation='relu')(inputs)
        h = Dense(int(dim/2), activation='relu')(h)
        h_encoded = Dense(int(dim/4), activation='relu')(h)  # This is the new variable for this architecture
        encoder_output_dim = int(dim/4)  # Set output dimension for fnn architecture
  
    elif architecture == 'autoencoder':
        h = Dense(int(dim/2), activation='relu')(inputs)
        h_encoded = Dense(int(dim/4), activation='relu')(h)
        encoder_output_dim = int(dim/4)  # Set output dimension for autoencoder

        # Decoder part
        h_decoder_start = Dense(int(dim/2), activation='relu')(h_encoded)
        h_decoded = Dense(dim, activation='sigmoid')(h_decoder_start)

        # Update the model outputs for autoencoder
        output_1 = h_encoded  # Using encoded representation as primary output
        output_2 = h_decoded  # Decoded output
  
   

    # ... add other architectures as needed

    # Mask estimator
    output_1 = Dense(dim, activation='sigmoid', name='mask')(h_encoded)  
    
    # Feature estimator
    output_2 = Dense(dim, activation='sigmoid', name='feature')(h_encoded)
        
    model = Model(inputs=inputs, outputs=[output_1, output_2])
    model.compile(optimizer='rmsprop',
                  loss={'mask': 'binary_crossentropy', 
                        'feature': 'mean_squared_error'},
                  loss_weights={'mask': 1, 'feature': alpha})

    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)

    # Fit model on unlabeled data
    history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, 
              epochs=epochs, batch_size=batch_size)

    # Extract encoder part for the given architecture
    encoder = Model(inputs=inputs, outputs=h_encoded)

    embeddings = get_embeddings(encoder, x_unlab)
    all_activations = get_all_activations(encoder, x_unlab)

    return encoder, embeddings, all_activations, encoder_output_dim, history

def get_encoder_added_loss(x_unlab, imputed_indices, architecture='default', p_m=None, alpha=None, parameters=None):
    """
    Flexible function to generate different encoder architectures and train them.

    Args:
    - x_unlab: The unlabeled feature data
    - imputed_indices: List of indices where the values were imputed
    - architecture: The type of architecture ('default' or 'fnn')
    - p_m: corruption probability
    - alpha: Hyper-parameter to control weights of feature and mask losses
    - parameters: Other parameters such as epochs and batch size

    Returns:
    - encoder: The encoder model
    - embeddings: The embeddings from the encoder
    - all_activations: All activations from the encoder
    - encoder_output_dim: The dimensionality of the encoder's output
    """

    _, dim = x_unlab.shape
    epochs = parameters['epochs']
    batch_size = parameters['batch_size']

    # Common input layer for all architectures
    inputs = Input(shape=(dim,))

    # Depending on the architecture, build the model
    if architecture == 'default':
        h_encoded = Dense(int(dim), activation='relu')(inputs)  # Renamed h to h_encoded
        encoder_output_dim = int(dim)  # Set output dimension for default architecture
    
    elif architecture == 'fnn':
        h = Dense(int(dim), activation='relu')(inputs)
        h = Dense(int(dim/2), activation='relu')(h)
        h_encoded = Dense(int(dim/4), activation='relu')(h)  # This is the new variable for this architecture
        encoder_output_dim = int(dim/4)  # Set output dimension for fnn architecture
  
    elif architecture == 'autoencoder':
        h = Dense(int(dim/2), activation='relu')(inputs)
        h_encoded = Dense(int(dim/4), activation='relu')(h)
        encoder_output_dim = int(dim/4)  # Set output dimension for autoencoder

        # Decoder part
        h_decoder_start = Dense(int(dim/2), activation='relu')(h_encoded)
        h_decoded = Dense(dim, activation='sigmoid')(h_decoder_start)

    # ... add other architectures as needed

    # Mask estimator
    output_1 = Dense(dim, activation='sigmoid', name='mask')(h_encoded)  
    
    # Feature estimator
    output_2 = Dense(dim, activation='sigmoid', name='feature')(h_encoded)
    
    # Extract the output values corresponding to imputed values
#     imputed_output = Lambda(lambda x: tf.gather(x, imputed_indices, axis=1), name='imputed_output')(output_2)  
    batch_size = parameters['batch_size']
    imputed_output = ImputedValuesLayer(imputed_indices, batch_size, name='imputed_output')(output_2)

#     imputed_output = ImputedValuesLayer(imputed_indices, name='imputed_output')(output_2)

    model = Model(inputs=inputs, outputs=[output_1, output_2, imputed_output])
    model.compile(optimizer='rmsprop',
                  loss={'mask': 'binary_crossentropy', 
                        'feature': 'mean_squared_error',
                        'imputed_output': 'mean_squared_error'},  # New loss for imputed values
                  loss_weights={'mask': 1, 'feature': alpha, 'imputed_output': 0.5*alpha})

    # Generate corrupted samples
    m_unlab = mask_generator(p_m, x_unlab)
    m_label, x_tilde = pretext_generator(m_unlab, x_unlab)

    # Getting actual data at imputed indices
    rows = imputed_indices[:, 0]
    cols = imputed_indices[:, 1]
    actual_data = x_unlab[rows, cols]
    # Fit model on unlabeled data
    # Step 1: Create a placeholder array
    modified_actual_data = np.zeros_like(x_unlab)

    # Step 2: Fill in the positions
    for i in range(len(rows)):
        modified_actual_data[rows[i], cols[i]] = actual_data[i]
    
    history = model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab, 'imputed_output': modified_actual_data}, 
          epochs=epochs, batch_size=batch_size)

    # Extract encoder part for the given architecture
    encoder = Model(inputs=inputs, outputs=h_encoded)

    embeddings = get_embeddings(encoder, x_unlab)
    all_activations = get_all_activations(encoder, x_unlab)

    return encoder, embeddings, all_activations, encoder_output_dim, history



def vime_self (x_unlab, p_m, alpha, parameters):
  """Self-supervised learning part in VIME.
  
  Args:
    x_unlab: unlabeled feature
    p_m: corruption probability
    alpha: hyper-parameter to control the weights of feature and mask losses
    parameters: epochs, batch_size
    
  Returns:
    encoder: Representation learning block
  """
    
  # Parameters
  _, dim = x_unlab.shape
  epochs = parameters['epochs']
  batch_size = parameters['batch_size']
  
  # Build model  
  inputs = Input(shape=(dim,))
  # Encoder  
  h = Dense(int(dim), activation='relu')(inputs)  
  # Mask estimator
  output_1 = Dense(dim, activation='sigmoid', name = 'mask')(h)  
  # Feature estimator
  output_2 = Dense(dim, activation='sigmoid', name = 'feature')(h)
  
  model = Model(inputs = inputs, outputs = [output_1, output_2])
  
  model.compile(optimizer='rmsprop',
                loss={'mask': 'binary_crossentropy', 
                      'feature': 'mean_squared_error'},
                loss_weights={'mask':1, 'feature':alpha})
  
  # Generate corrupted samples
  m_unlab = mask_generator(p_m, x_unlab)
  m_label, x_tilde = pretext_generator(m_unlab, x_unlab)
  
  # Fit model on unlabeled data
  model.fit(x_tilde, {'mask': m_label, 'feature': x_unlab}, 
            epochs = epochs, batch_size= batch_size)
      
  # Extract encoder part
  layer_name = model.layers[1].name
  layer_output = model.get_layer(layer_name).output
  encoder = models.Model(inputs=model.input, outputs=layer_output)
  embeddings = get_embeddings(encoder, x_unlab)
  all_activations = get_all_activations(encoder, x_unlab)

  return encoder, embeddings, all_activations
