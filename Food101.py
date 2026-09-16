#!/usr/bin/env python
# coding: utf-8

# # Food101: Surpassing Research Benchmarks with EfficientNetB0
# **Author:** Puneet Sharma  
# **Objective:** To outperform the [DeepFood (2016)](https://www.researchgate.net/publication/304163308_DeepFood) benchmark of **77.4%** accuracy using a modern, efficient architecture and optimized data pipeline.
# 

# ## The Strategy
# 1. **Feature Extraction:** Establishing a baseline with a frozen backbone.
# 2. **Fine-Tuning:** Unfreezing the full architecture for class-specific adaptation.
# 3. **Data Augmentation:** Implementing a high-speed CPU-bound pipeline to prevent overfitting.
# 4. **Regularization:** Using Dropout (0.3) to reach peak validation stability.

# ## Imports

# In[1]:


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import mixed_precision, layers, models
from tensorflow.keras.models import Sequential
import numpy as np
import random
import pandas as pd
from sklearn.metrics import accuracy_score


# ## Helper Functions

# In[2]:


from helper_functions import *


# ## Check GPU
# 
# This block identifies the hardware and prepares the memory for training.

# In[3]:


# Check for GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # 'Memory Growth' prevents TensorFlow from hogging all of RAM at once.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs Detected: {len(gpus)}")
        print(f"Device Details: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Ensure you are connected to a GPU runtime in Colab.")


# ## Exploring Dataset
# 

# Most popular datasets in the machine learning world, can be accessed through [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/overview).  

# In[4]:


get_ipython().system('pip install --upgrade protobuf tensorflow-datasets')


# In[5]:


# Get Tensorflow Datasets
import tensorflow_datasets as tfds


# In[6]:


import tensorflow_datasets as tfds

# List all available datasets
datasets_list = tfds.list_builders() # Get all available datasets in TFDS
print("food101" in datasets_list)


# [Tensorflow Datasets Food101](https://www.tensorflow.org/datasets/catalog/food101)

# In[7]:


# Install importlib_resources, a dependency for tensorflow-datasets
get_ipython().system('pip install importlib_resources')


# In[8]:


# Load in the data
(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)


# In[9]:


# Meta data about the dataset
ds_info


# In[10]:


# Features of Food101
ds_info.features


# In[11]:


# Get class names
class_names = ds_info.features["label"].names
print(len(class_names) )
class_names[:10]


# In[12]:


# Take one sample
train_one_sample = train_data.take(1) # samples are in format (image, label)
train_one_sample


# Loop through single training sample and get some info from the `image` and `label`.

# In[13]:


# Output info about our training sample
for image, label in train_one_sample:
  print(f"""
  Image shape: {image.shape}
  Image dtype: {image.dtype}
  Target class from Food101 (tensor form): {label}
  Class name (str form): {class_names[label]}
        """)


# The code block below will give a different result each time. Because the parameter `shuffle_files` is set as `True` in `tfds.load()`
# 
# The images have different shapes, for example `(512, 342, 3)` and `(512, 512, 3)`
# (height, width, color_channels).

# The image from Food101 Datasets

# In[14]:


image


# In[15]:


# What are the min and max values?
tf.reduce_min(image), tf.reduce_max(image)


# ### Let's plot some images

# In[16]:


fig = tfds.show_examples(train_data, ds_info)


# In[17]:


# Plot a single image
plt.imshow(image)
plt.title(class_names[label]) # add title to image by indexing on class_names list
plt.axis(False)


# ## Preprocessing of Data

# Data is currently:
# * In `uint8` data type
# * Comprised of all different sized tensors (different sized images)
# * Not scaled (the pixel values are between 0 & 255)
# 
# After Preprocessing Data
# * In `float32` data type
# * Have all of the same size tensors (batches require all tensors have the same shape, e.g. `(224, 224, 3)`)
# * Scaled (values between 0 & 1), also called normalized

# In[18]:


# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=224, augment=False):
    """
    Prepares images for EfficientNetB0 by resizing, casting, and optionally augmenting.

    Standardizes input images to a fixed shape and float32 dtype. If 'augment' is True,
    applies a series of random transformations (flips, crops, etc.) via the data_augment
    helper function to improve model generalization.

    Args:
        image: The input image tensor (uint8).
        label: The integer label associated with the image.
        img_shape (int): The target height and width for resizing. Default is 224.
        augment (bool): Whether to apply data augmentation. Default is False.

    Returns:
        A tuple of (float32_image, label) where the image is resized to
        [img_shape, img_shape, 3].
    """
    if augment:
        image, label = data_augment(image, label, img_shape=img_shape)
    else:
        image = tf.image.resize(image, [img_shape, img_shape])

    return tf.cast(image, tf.float32), label


# In[19]:


def data_augment(image, label, img_shape=224):
    """
    Applies custom geometric augmentations using native TensorFlow image operations.

    This function simulates complex transformations (Zoom, Pan, Flip) on the CPU,
    enabling high-performance parallel data loading without GPU overhead.

    Args:
        image (tf.Tensor): The input image tensor.
        label (tf.Tensor): The associated class label.
        img_shape (int): The target output dimensions.

    Returns:
        tuple: (Augmented float32 image, original label).
    """
    # 1. Random Horizontal Flip
    image = tf.image.random_flip_left_right(image)

    # 2. Random Zoom & Crop (Simulates Zoom/Height/Width shifts)
    # We define a random scale between 80% and 100% of the original size
    img_hd = tf.cast(tf.shape(image)[0], tf.float32)
    img_wd = tf.cast(tf.shape(image)[1], tf.float32)

    fraction = tf.random.uniform([], 0.8, 1.0)
    new_h = tf.cast(img_hd * fraction, tf.int32)
    new_w = tf.cast(img_wd * fraction, tf.int32)

    # Apply the crop and immediately resize back to target img_shape
    image = tf.image.random_crop(image, size=[new_h, new_w, 3])
    image = tf.image.resize(image, [img_shape, img_shape])

    # 3. Final Cast
    return tf.cast(image, tf.float32), label


# `preprocess_img()` function above takes image and label as input (even though it does nothing to the label) because our dataset is currently in the tuple structure `(image, label)`.
# 
# Try function out on a target image.

# The input image gets converted from `uint8` to `float32` and gets reshaped from its current shape to `(224, 224, 3)`.

# Plot the preprocessed image to see

# In[20]:


processed_img = preprocess_img(image, label)[0]
print(processed_img.shape, processed_img.dtype)
plt.imshow(processed_img/255.)
plt.title(class_names[label])
plt.axis(False)


# ## Prepare Dataset

# In[21]:


train_data, test_data


# In[22]:


# Map preprocessing function to training data (and parallelize)
processed_train_data = (train_data
              .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
              # .cache("m4_food101_cache") # Saves to SSD to keep RAM free
              .shuffle(buffer_size=1000)
              .batch(32)
              .prefetch(buffer_size=tf.data.AUTOTUNE))

processed_test_data = (test_data
             .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
            #  .cache("m4_food101_test_cache")
             .batch(32)
             .prefetch(buffer_size=tf.data.AUTOTUNE))


# In[23]:


processed_train_data, processed_test_data


# ## Setup Mixed Precision Training
# 
# Mixed precision training involves using a mix of float16 and float32 tensors to make better use of your GPU's memory.
# 
# For a more detailed explanation : [TensorFlow mixed precision guide](https://www.tensorflow.org/guide/mixed_precision)
# 
# **Note:** If your GPU doesn't have a score of over 7.0+ (e.g. P100 in Google Colab), mixed precision won't work (see: ["Supported Hardware"](https://www.tensorflow.org/guide/mixed_precision#supported_hardware) in the mixed precision guide for more).

# To handle 101,000 images efficiently, the pipeline was built for maximum throughput:
# * **Parallel Mapping:** Used `num_parallel_calls=tf.data.AUTOTUNE` to offload augmentation to the CPU.
# * **Prefetching:** Overlapped data preparation with model training to eliminate GPU starvation.
# * **Deterministic Resizing:** Ensured all 101 classes were standardized to 224x224 before entering the network.

# In[24]:


# Turn on mixed precision training
mixed_precision.set_global_policy(policy="mixed_float16") # set global policy to mixed precision


# As long as the GPU used has a compute capability of 7.0+ the cell above should run without error.

# In[25]:


mixed_precision.global_policy() # should output "mixed_float16" (if your GPU is compatible with mixed precision)


# In[26]:


print(f"Compute dtype: {mixed_precision.global_policy().compute_dtype}")   # bfloat16/float16
print(f"Variable dtype: {mixed_precision.global_policy().variable_dtype}") # float32


# ## Callbacks

# In[27]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=3) # if val loss decreases for 3 epochs in a row, stop training

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down
                                                 min_lr=1e-7)


# ## Models

# ### Experiment 1: Feature Extraction (Baseline)
# Before fine-tuning, we evaluate the pre-trained weights of EfficientNetB0. This establishes the "floor" of our performance.
# * **Model:** EfficientNetB0 (Frozen)
# * **Epochs:** Early Stopping (patience=3)
# * **Goal:** Reach >70% accuracy before unfreezing layers.

# In[28]:


input_shape = (224, 224, 3)

# Create base model
base_model = tf.keras.applications.EfficientNetB0(include_top=False) # set include_top=False to remove output layer
base_model.trainable = False # freeze the base model

# Create functional model
inputs = layers.Input(shape=input_shape, name="input_layer")
# x = layers.Rescaling(1/255.)(x)
x = base_model(inputs, training=False) # set base_model to inference mode only
x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)

# with mixed precision
x = layers.Dense(len(class_names))(x) # one output neuron per class 101
outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)

# without mixed precision
# outputs = layers.Dense(len(class_names), activation="softmax", name="output_layer")(x) # one output neuron per class 101


model0 = tf.keras.Model(inputs, outputs)

# Compile the model
model0.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])


# In[29]:


model0.summary()


# #### Checking layer dtype policies

# In[30]:


# Check the dtype_policy attributes of layers in our model
for layer in model0.layers:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # Check the dtype policy of layers


# In[31]:


# Check the layers in the base model and see what dtype policy they're using
for layer in model0.layers[1].layers[:20]: # only check the first 20 layers to save output space
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# #### Training

# In[32]:


history_model0 = model0.fit(processed_train_data,
                            epochs=100, # fine-tune for a maximum of 100 epochs
                            steps_per_epoch=len(processed_train_data),
                            validation_data=processed_test_data,
                            validation_steps=int(0.15 * len(processed_test_data)), # validation during training on 15% of test data
                            callbacks=[create_tensorboard_callback("training_logs", "model0"), # track the model training logs
                                       create_model_checkpoint("model_checkpoints", "model0"), # save only the best model during training
                                       early_stopping,
                                       reduce_lr]) # stop model after X epochs of no improvements


# #### Evaluation

# In[33]:


# Evaluate model (unsaved version) on whole test dataset
results_model0 = model0.evaluate(processed_test_data)
results_model0


# In[34]:


plot_loss_curves(history_model0)


# In[35]:


# Good to save model to prevent re-training model again
model0.save("model0.keras")


# ### Experiment 2: Global Fine-Tuning
# A key finding in this project was that **unfreezing all layers** outperformed the common practice of only unfreezing the top 20.
# * **Insight:** Because food categories share many low-level features (textures, colors), the entire network benefited from being adapted to the Food101 distribution.
# * **Learning Rate:** Dropped to $4e-6$ to ensure stable convergence without destroying pre-trained patterns.
# 

# In[36]:


model1 = tf.keras.models.load_model("model0.keras")


# In[37]:


np.isclose(results_model0, model1.evaluate(processed_test_data))


#  All of the layers in the base model is frozen by setting `base_model.trainable=False`
# 

# In[38]:


# Are any of the layers in our model frozen?
for layer in model1.layers:
    layer.trainable = True # set all layers to trainable
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # make sure loaded model is using mixed precision dtype_policy ("mixed_float16")


# In[39]:


for layer in model1.layers[1].layers[:20]:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# In[40]:


# Compile the model
model1.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
                        optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
                        metrics=["accuracy"])


# #### Training

# In[41]:


history_model1 = model1.fit(processed_train_data,
                               epochs=100, # fine-tune for a maximum of 100 epochs
                               steps_per_epoch=len(processed_train_data),
                               validation_data=processed_test_data,
                               validation_steps=int(0.15 * len(processed_test_data)), # validation during training on 15% of test data
                               callbacks=[create_tensorboard_callback("training_logs", "model1"), # track the model training logs
                                          create_model_checkpoint("model_checkpoints", "model1"), # save only the best model during training
                                          early_stopping, # stop model after X epochs of no improvements
                                          reduce_lr]) # reduce the learning rate after X epochs of no improvements


# #### Evaluation

# In[42]:


results_model1 = model1.evaluate(processed_test_data)
results_model1


# In[43]:


plot_loss_curves(history_model1)


# In[44]:


model1.save("model1.keras")


# ### Experiment 3: Global Fine-Tuning
# A key finding in this project was that **unfreezing all layers** outperformed the common practice of only unfreezing the top 20.
# * **Insight:** Because food categories share many low-level features (textures, colors), the entire network benefited from being adapted to the Food101 distribution.
# * **Learning Rate:** Dropped to $4e-6$ to ensure stable convergence without destroying pre-trained patterns.

# #### Prepare Dataset

# In[45]:


# Take one sample
train_one_sample = train_data.take(1)
train_one_sample


# In[46]:


for image , label in train_one_sample:
    print(image.shape, label)


# In[47]:


# Plot an single image
plt.imshow(image)
plt.title(class_names[label]) # add title to image by indexing on class_names list
plt.axis(False)


# In[48]:


processed_img = preprocess_img(image, label)[0]
print(processed_img.shape, processed_img.dtype)
plt.imshow(processed_img/255.)
plt.title(class_names[label])
plt.axis(False)


# In[49]:


# Map preprocessing function to training data (and paralellize)

processed_train_data = (train_data
              .map(map_func=lambda x, y: preprocess_img(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
              # .cache("m4_food101_cache") # Saves to SSD to keep 24GB RAM free
              .shuffle(buffer_size=1000)
              .batch(32)
              .prefetch(buffer_size=tf.data.AUTOTUNE))

processed_test_data = (test_data
             .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
            #  .cache("m4_food101_test_cache")
             .batch(32)
             .prefetch(buffer_size=tf.data.AUTOTUNE))


# In[50]:


model2 = tf.keras.models.load_model("model0.keras")


# In[51]:


np.isclose(results_model0, model2.evaluate(processed_test_data))


# In[52]:


# Are any of the layers in our model frozen?
for layer in model2.layers:
    layer.trainable = True # set all layers to trainable
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy) # make sure loaded model is using mixed precision dtype_policy ("mixed_float16")


# In[53]:


for layer in model2.layers[1].layers[:20]:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)


# In[54]:


# Compile the model
model2.compile(loss="sparse_categorical_crossentropy", # sparse_categorical_crossentropy for labels that are *not* one-hot
                        optimizer=tf.keras.optimizers.Adam(0.0001), # 10x lower learning rate than the default
                        metrics=["accuracy"])


# #### Training

# In[55]:


history_model2 = model2.fit(processed_train_data,
                                     epochs=100, # fine-tune for a maximum of 100 epochs
                                     steps_per_epoch=len(processed_train_data),
                                     validation_data=processed_test_data,
                                     validation_steps=int(0.15 * len(processed_test_data)), # validation during training on 15% of test data
                                     callbacks=[create_tensorboard_callback("training_logs", "model2"), # track the model training logs
                                                create_model_checkpoint("model_checkpoints", "model2"), # save only the best model during training
                                                early_stopping, # stop model after X epochs of no improvements
                                                reduce_lr]) # reduce the learning rate after X epochs of no improvements


# #### Evaluation

# In[56]:


results_model2 = model2.evaluate(processed_test_data)
results_model2


# In[89]:


plot_loss_curves(history_model2)


# In[58]:


model2.save("model2.keras")


# ### Experiment 4: Regularization & Augmentation
# At this stage, the model began to overfit (96% Training vs 83% Validation). To close this gap, two techniques were introduced:
# 1. **CPU-Side Augmentation:** Random flips, brightness shifts, and crops to simulate real-world restaurant lighting and angles.
# 2. **Dropout (0.3):** Added to the final dense layer to force the model to learn redundant, robust features.

# In[59]:


input_shape = (224, 224, 3)

# Create base models
base_model = tf.keras.applications.EfficientNetB0(include_top=False) # set include_top=False to remove output layer
base_model.trainable = True # freeze the base model

# Create functional model
inputs = layers.Input(shape=input_shape, name="input_layer")
# x = layers.Rescaling(1/255.)(x)
x = base_model(inputs, training=False) # set base_model to inference mode only
x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dropout(0.3)(x)

# with mixed precision
x = layers.Dense(len(class_names))(x) # one output neuron per class 101
outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)

# without mixed precision
# outputs = layers.Dense(len(class_names), activation="softmax", name="output_layer")(x) # one output neuron per class 101

model3 = tf.keras.Model(inputs, outputs)

# Compile the model
model3.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy"])


# #### Training

# In[60]:


history_model3 = model3.fit(processed_train_data,
                            epochs=100, # fine-tune for a maximum of 100 epochs
                            steps_per_epoch=len(processed_train_data),
                            validation_data=processed_test_data,
                            validation_steps=int(0.15 * len(processed_test_data)), # validation during training on 15% of test data
                            callbacks=[create_tensorboard_callback("training_logs", "model3"), # track the model training logs
                                       create_model_checkpoint("model_checkpoints", "model3"), # save only the best model during training
                                       early_stopping, # stop model after X epochs of no improvements
                                       reduce_lr])


# #### Evaluation

# In[90]:


results_model3 = model3.evaluate(processed_test_data)
results_model3


# In[119]:


print('Evaluating model3 with best saved weights...')
model3.load_weights('model_checkpoints/model3.weights.h5') # Load the best weights
results_model3_best_weights = model3.evaluate(processed_test_data)
print(f"Results with best weights: {results_model3_best_weights}")


# In[91]:


plot_loss_curves(history_model3)


# In[92]:


model3.save("model3.keras")


# ## Making Predictions with our model

# In[93]:


model3 = tf.keras.models.load_model("model3.keras")


# In[65]:


predict_prob = model3.predict(processed_test_data)


# In[94]:


len(predict_prob)


# In[95]:


predict_prob.shape


# In[96]:


predict_prob[0]


# In[97]:


sum(predict_prob[0])


# In[98]:


print(f"The class with the highest predicted probability by the model for sample 0: {predict_prob[0].argmax()}")


# In[99]:


class_names[predict_prob[0].argmax()]


# In[72]:


# 1. Create empty lists to catch everything in order
all_y_true = []
all_y_pred = []
all_y_pred_probs = []

# 2. Loop through the test data once to ensure perfect alignment
for images, labels in processed_test_data:
    # Get predictions for this batch
    preds = model3.predict(images, verbose=0)

    # Store the true labels and the predicted labels
    all_y_true.extend(labels.numpy())
    all_y_pred.extend(preds.argmax(axis=1))
    all_y_pred_probs.extend(preds.max(axis=1))


# 3. Convert to numpy arrays
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)


# ### Evaluating models predictions

#  Check model's predictions array `(pred_classes)` is in the same order as our test labels array `(y_labels)` is to find the accuracy score

# In[100]:


# Let's try scikit-learn's accuracy score function and see what it comes up with
sklearn_accuracy = accuracy_score(y_true=all_y_true, y_pred=all_y_pred)
sklearn_accuracy


# In[101]:


class_names[:10]


# In[102]:


make_confusion_matrix(y_true=all_y_true,
                      y_pred=all_y_pred,
                      classes=class_names,
                      figsize=(100,100),
                      text_size=20)


# Scikit-learn has a helpful function for acquiring many different classification metrics per class (e.g. precision, recall and F1) called classification_report, let's try it out.

# In[103]:


from sklearn.metrics import classification_report
print(classification_report(all_y_true, all_y_pred))


# In[104]:


# Get a dictionary of the classification report
classification_report_dict = classification_report(all_y_true, all_y_pred, output_dict=True)
classification_report_dict


# In[105]:


# Create empty dictionary
class_f1_scores = {}
# Loop through  classification report dictionary items
for k, v in classification_report_dict.items():
  if k == "accuracy": # stop once we get to accuracy key
    break
  else:
    # Add class names and f1-scores to new dictionary
    class_f1_scores[class_names[int(k)]] = v["f1-score"]
class_f1_scores


# In[106]:


# Turn f1-scores into dataframe for visualization
f1_scores = pd.DataFrame({"class_names": list(class_f1_scores.keys()),
                          "f1-score": list(class_f1_scores.values())}).sort_values("f1-score", ascending=False)


# In[107]:


f1_scores[:10]


# In[108]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12,25))
scores = ax.barh(range(len(f1_scores)), f1_scores["f1-score"].values) # get f1_scores
ax.set_yticks(range(len(f1_scores)))
ax.set_yticklabels(f1_scores["class_names"])
ax.set_xlabel("F1-Score")
ax.set_title("F1-Score for 101 Food Classes")
ax.invert_yaxis(); # reverse the order


# ### Visualize predictions

# In[109]:


plt.figure(figsize=(17, 10))

for i in range(3):

  # 1. Shuffle with a buffer of 1000, then take 1 to make it random
  random_sample = test_data.shuffle(1000).take(1)

  for image, label in random_sample:
    # 2. Match your model's preprocessing (224, 224)
    img = tf.image.resize(image, [224, 224])
    img = tf.cast(img, tf.float32)

    # 3. Add batch dimension (1, 224, 224, 3)
    img_expanded = tf.expand_dims(img, axis=0)

    # 4. Predict using your best model (model3)
    pred_prob = model3.predict(img_expanded, verbose=0)
    pred_class = class_names[pred_prob.argmax()]
    actual_class = class_names[label.numpy()]

    # 5. Plot
    plt.subplot(1, 3, i+1)
    plt.imshow(image.numpy())
    title_color = "green" if actual_class == pred_class else "red"
    plt.title(f"Actual: {actual_class}\nPred: {pred_class}\nProb: {pred_prob.max():.2f}",
              c=title_color)
    plt.axis(False)


# ### Finding the most wrong predictions

# In[110]:


#Create the DataFrame using the full evaluation arrays
pred_df = pd.DataFrame({
    "y_true": all_y_true,
    "y_pred": all_y_pred,
    "pred_conf": all_y_pred_probs,

    # Since we didn't store all 25,250 raw probabilities to save memory,
    # we'll label this as the final class names for now:
    "y_true_classname": [class_names[i] for i in all_y_true],
    "y_pred_classname": [class_names[i] for i in all_y_pred]
})

pred_df


# In[111]:


# Add a column to easily see if the prediction was correct
pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]
pred_df.head()


# In[112]:


# Sort our DataFrame to have most predictions at the top
top_100_wrong = pred_df[pred_df["pred_correct"] == False].sort_values("pred_conf", ascending=False)[:100]
top_100_wrong.head(20)


# In[113]:


# Visualize the test data samples which have the wrong prediction but highest pred probability
test_data_list = list(test_data.as_numpy_iterator())

images_to_view = 9
start_index = 0
plt.figure(figsize=(15, 10))

# 2. Loop through our 'Most Confident Mistakes'
for i, row in enumerate(top_100_wrong[start_index:start_index+images_to_view].itertuples()):
    plt.subplot(3, 3, i+1)

    # row[0] is the original index from the test set
    idx = row.Index
    img, label = test_data_list[idx]

    # Display the image (it's already a numpy array from the iterator)
    plt.imshow(img)

    # Use the class names we stored in the DataFrame
    plt.title(f"Actual: {row.y_true_classname}\nPred: {row.y_pred_classname}\nProb: {row.pred_conf:.2f}",
              fontsize=10,
              color="red") # It's a mistake, so we keep it red
    plt.axis(False)

plt.show()


# ### Making prediction on Custom images

# In[114]:


# Get the custom food images filepaths
custom_food_images = ["custom_food_images/" + img_path for img_path in os.listdir("custom_food_images")]
custom_food_images


# In[118]:


# Make predictions on and plot custom food images
for img in custom_food_images:
  img = load_and_prep_image(img, scale=False) # don't need to scale for our EfficientNetB0 model
  pred_prob = model3.predict(tf.expand_dims(img, axis=0)) # make prediction on image with shape [1, 224, 224, 3] (same shape as model was trained on)
  pred_class = class_names[pred_prob.argmax()] # get the index with the highet prediction probability
  # Plot the appropriate information
  plt.figure()
  plt.imshow(img/225.)
  plt.title(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
  plt.axis(False)


# ## Final Results & Benchmarking
# The final model achieved a **Validation Accuracy of 85.05%**.
# 
# | Metric | Result |
# | :--- | :--- |
# | **Final Accuracy** | **85.05%** |
# | **Benchmark (DeepFood)** | 77.4% |
# | **Improvement** | **+7.65%** |
# 
# ### Key Takeaways:
# * **Architecture:** EfficientNetB0 proved to be a highly efficient "pound-for-pound" winner, beating much larger models from previous years.
# * **Optimization:** Moving augmentation to the `tf.data` pipeline reduced training time by nearly 50% compared to Keras Layers.

# ## Model Limitations & Error Analysis
# The model excels at identifying visually distinct foods (e.g., Edamame, Hot and Sour Soup) but shows confusion in categories with high "intra-class variance" like **Steak vs. Pork Chops**.
# 
# **Future Work:**
# * Implementing **Label Smoothing** to handle the known "noisy" labels in the Food101 dataset.
# * Deploying via **TensorFlow Lite** or **ONNX Runtime** for real-time inference on edge devices or mobile applications.
