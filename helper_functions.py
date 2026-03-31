import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import zipfile
import os
import datetime

# TENSORBOARD CALLBACKS #
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instance to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback

# PLOT HISTORY CURVES() Loss and accuracy) #
def plot_loss_curves(history):
  """
  Returns separate loss curves(loss and accuracy) for training and validation metrics.

  Args:
    history: TensorFlow model History object 
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

# MODEL CHECKPOINT #
def create_model_checkpoint(dir_name, experiment_name):
    """
    Creates a ModelCheckpoint callback to save the best version of model weights.

    This function is pre-configured with optimized settings for the Food101 
    transfer learning task:
    - monitor="val_accuracy": Saves weights based on peak validation performance.
    - save_best_only=True: Overwrites the file only when a new accuracy record is set.
    - save_weights_only=True: Saves model weights only (.h5), reducing disk usage 
      and increasing efficiency for transfer learning re-loading.
    - verbose=1: Provides clear visual confirmation in logs when a new best model is saved.

    Args:
        dir_name (str): The target directory to store the checkpoint.
        experiment_name (str): The name of the specific experiment (e.g., 'fine_tuned_all_layers').

    Returns:
        tf.keras.callbacks.ModelCheckpoint: A configured callback instance.
    """
    filepath = f"{dir_name}/{experiment_name}.weights.h5"
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    return checkpoint_callback

# CONFUSION MATRIX #
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
  """
  Makes a labelled confusion matrix comparing predictions and ground truth labels.

  This function facilitates deep error analysis by visualizing class-wise performance.
  It supports normalization to show percentages, which is critical for imbalanced 
  datasets or large-scale classification like Food101.

  Args:
    y_true (np.array): Ground truth labels (must be same shape as y_pred).
    y_pred (np.array): Predicted labels (must be same shape as y_true).
    classes (list): Array of class names (e.g., strings). If None, integers are used.
    figsize (tuple): Size of the output figure. Default is (10, 10).
    text_size (int): Size of the font in the matrix cells. Default is 15.
    norm (bool): If True, shows percentages in addition to raw counts.
    savefig (bool): If True, saves the matrix to 'confusion_matrix.png'.

  Returns:
    None: Displays a Matplotlib figure of the confusion matrix.
  """
  # Create the confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] 
  n_classes = cm.shape[0]

  # Plot the figure
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) 
  fig.colorbar(cax)

  # Handle labels
  labels = classes if classes else np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted Label",
         ylabel="True Label",
         xticks=np.arange(n_classes),
         yticks=np.arange(n_classes),
         xticklabels=labels,
         yticklabels=labels)

  # Positioning and rotation for 101 classes
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Color threshold for text visibility
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
               horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  if savefig:
    fig.savefig("confusion_matrix.png")

# LOAD IMAGAES AND PREPARE FOR MODEL #
def load_and_prep_image(filename, img_shape=224, scale=False):
    """
    Reads an image from a file, converts it to a tensor, and reshapes it.

    This function is designed for inference on single images. Note that for 
    EfficientNet architectures, scaling (0-1) is typically handled internally 
    by the model, so the default is scale=False.

    Args:
        filename (str): The path to the target image file.
        img_shape (int): The target height and width (default=224).
        scale (bool): Whether to scale pixel values to the [0, 1] range.

    Returns:
        tf.Tensor: An image tensor of shape (img_shape, img_shape, 3).
    """
    # 1. Read in the image file
    img = tf.io.read_file(filename)

    # 2. Decode the image into a tensor with 3 color channels (RGB)
    # expand_animations=False ensures we don't accidentally load GIFs as 4D tensors
    img = tf.io.decode_image(img, channels=3, expand_animations=False)

    # 3. Resize the image to the target dimensions
    img = tf.image.resize(img, size=[img_shape, img_shape])

    # 4. Rescale if explicitly requested
    if scale:
        return img / 255.
    
    return img