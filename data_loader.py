"""VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain (VIME) Codebase.

Reference: Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar, 
"VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain," 
Neural Information Processing Systems (NeurIPS), 2020.
Paper link: TBD
Last updated Date: October 11th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)
-----------------------------

data_loader.py
- Load and preprocess MNIST data (http://yann.lecun.com/exdb/mnist/)
"""

# Necessary packages
import numpy as np
import pandas as pd
from keras.datasets import mnist

import numpy as np
import pandas as pd

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from keras.utils import to_categorical

import pandas as pd
import numpy as np
from keras.utils import to_categorical

import pandas as pd
import numpy as np
from keras.utils import to_categorical

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_excel_data_multi_class(filename, label_data_rate=0.2, test_data_rate=0.2):
    """
    Load multi-class data from an Excel file and prepare it for semi-supervised learning.
    
    Args:
    - filename (str): path to the Excel file.
    - label_data_rate (float): proportion of labeled data.
    - test_data_rate (float): proportion of test data.
    
    Returns:
    - x_label (pd.DataFrame): labeled data features.
    - y_label (np.ndarray): one-hot encoded labeled data target.
    - x_unlab (pd.DataFrame): unlabeled data features.
    - x_test (pd.DataFrame): test data features.
    - y_test (np.ndarray): one-hot encoded test data target.
    """
    # Load data
    data = pd.read_excel(filename)
    
    # Convert 'class' column to binary classification: P (patient) as 1 and H (healthy) as 0
    data['class'] = data['class'].map({'P': 1, 'H': 0})
    
    # Drop the 'ID' column
    data = data.drop(columns=['ID'])
    
    # Separate features and target variable
    X = data.drop(columns=['class'])
    y = data['class']
    
    # Normalize the data
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Compute the number of labeled, unlabeled, and test samples
    num_samples = len(data)
    num_test_samples = int(test_data_rate * num_samples)
    num_labeled_samples = int(label_data_rate * (num_samples - num_test_samples))
    
    # Split data into train and test sets
    x_temp, x_test, y_temp, y_test = train_test_split(X, y, test_size=num_test_samples, stratify=y, random_state=42)
    
    # Further split training data into labeled and unlabeled sets
    x_label, x_unlab, y_label, _ = train_test_split(x_temp, y_temp, train_size=num_labeled_samples, stratify=y_temp, random_state=42)
    
    # Convert the targets to one-hot encoded vectors
    y_label = to_categorical(y_label)
    y_test = to_categorical(y_test)
    
    return x_label, y_label, x_unlab, x_test, y_test



def load_excel_data(filename, label_data_rate=0.2, test_data_rate=0.2):
    """
    Loads data from an Excel file and splits it into labeled, unlabeled, and test sets.
    
    Args:
    - filename: Path to the Excel file.
    - label_data_rate: Ratio of labeled data.
    - test_data_rate: Ratio of test data.
    
    Returns:
    - x_label: Labeled features.
    - y_label: Labeled target.
    - x_unlab: Unlabeled features.
    - x_test: Test features.
    - y_test: Test target.
    """
    
    # Load the Excel data
    data = pd.read_excel(filename)
    
    # Separate features and target
    X = data.drop(columns=['Grade'])
    y = data['Grade']
    
    # Splitting data into training and testing
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    
    test_size = int(len(X) * test_data_rate)
    train_idx, test_idx = idx[test_size:], idx[:test_size]
    
    # Further divide training data into labeled and unlabeled sets
    label_size = int(len(train_idx) * label_data_rate)
    label_idx = train_idx[:label_size]
    unlab_idx = train_idx[label_size:]
    
    # Debug prints
    print(f"Total train data length: {len(train_idx)}")
    print(f"Label data length (from train_idx): {len(label_idx)}")
    print(f"Unlabel data length (from train_idx): {len(unlab_idx)}")
    print(f"Provided label_data_rate: {label_data_rate}")

    x_label, y_label = X.iloc[label_idx], y.iloc[label_idx]
    x_unlab = X.iloc[unlab_idx]
    x_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    return x_label, y_label, x_unlab, x_test, y_test

# def load_excel_data(filename, label_data_rate=0.5, test_data_rate=0.2):
#     """
#     Loads data from an Excel file and splits it into labeled, unlabeled, and test sets.
    
#     Args:
#     - filename: Path to the Excel file.
#     - label_data_rate: Ratio of labeled data.
#     - test_data_rate: Ratio of test data.
    
#     Returns:
#     - x_label: Labeled features.
#     - y_label: Labeled target.
#     - x_unlab: Unlabeled features.
#     - x_test: Test features.
#     - y_test: Test target.
#     """
    
#     # Load the Excel data
#     data = pd.read_excel(filename)
    
#     # Separate features and target
#     X = data.drop(columns=['Grade'])
#     y = data['Grade']
    
#     # Splitting data into training and testing
#     idx = np.arange(len(X))
#     np.random.shuffle(idx)
    
#     test_size = int(len(X) * test_data_rate)
#     train_idx, test_idx = idx[test_size:], idx[:test_size]
    
#     # Further divide training data into labeled and unlabeled sets
#     label_size = int(len(train_idx) * label_data_rate)
#     label_idx = train_idx[:label_size]
#     unlab_idx = train_idx[label_size:]
    
#     x_label, y_label = X.iloc[label_idx], y.iloc[label_idx]
#     x_unlab = X.iloc[unlab_idx]
#     x_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
#     return x_label, y_label, x_unlab, x_test, y_test

# Example usage:
# x_label, y_label, x_unlab, x_test, y_test = load_excel_data('your_file.xlsx')


def load_mnist_data(label_data_rate):
  """MNIST data loading.
  
  Args:
    - label_data_rate: ratio of labeled data
  
  Returns:
    - x_label, y_label: labeled dataset
    - x_unlab: unlabeled dataset
    - x_test, y_test: test dataset
  """
  # Import mnist data
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # One hot encoding for the labels
  y_train = np.asarray(pd.get_dummies(y_train))
  y_test = np.asarray(pd.get_dummies(y_test))

  # Normalize features
  x_train = x_train / 255.0
  x_test = x_test / 255.0
    
  # Treat MNIST data as tabular data with 784 features
  # Shape
  no, dim_x, dim_y = np.shape(x_train)
  test_no, _, _ = np.shape(x_test)
  
  x_train = np.reshape(x_train, [no, dim_x * dim_y])
  x_test = np.reshape(x_test, [test_no, dim_x * dim_y])
  
  # Divide labeled and unlabeled data
  idx = np.random.permutation(len(y_train))
  
  # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
  label_idx = idx[:int(len(idx)*label_data_rate)]
  unlab_idx = idx[int(len(idx)*label_data_rate):]
  
  # Unlabeled data
  x_unlab = x_train[unlab_idx, :]
  
  # Labeled data
  x_label = x_train[label_idx, :]  
  y_label = y_train[label_idx, :]
  
  return x_label, y_label, x_unlab, x_test, y_test