#!/usr/bin/env python
# coding: utf-8

# ### A Deep Learning Approach: Identifying Disease in Apple Trees
# Apples are one of the most important temperate fruit crops in the world. Foliar (leaf) diseases pose a major threat to the overall productivity and quality of apple orchards. The current process for disease diagnosis in apple orchards is based on manual scouting by humans, which is time-consuming and expensive.

# #### Objective
# * The main objective of the competition is to develop machine learning-based models to accurately classify a given leaf image from the test dataset to a particular disease category, and to identify an individual disease from multiple disease symptoms on a single leaf image.

# ##### Reading in the CSV file

# In[1]:


import numpy as np
import pandas as pd


# In[222]:


train_data = pd.read_csv("data/train.csv")
train_data.head()


# In[223]:


train_data.info()


# ##### Checking to see if there are any null values in the dataset

# In[224]:


train_data.isnull().any()


# #### We are going to have to make directorys into the labels that are given so that when we use keras it will be a lot easier!
# * We are going to generate data from these images from the train_data directory
# * Only run this once if you download the dataset from kaggle

# In[4]:


# import os
# from shutil import move
# for file in os.listdir("data/train_images"):
#   #print(file)
#     temp = train_data[train_data["image"] == file]
#     label = np.array(temp["labels"])[0]
#     #print(label)
#     if not os.path.exists("data/train_data/"+label):
#         os.makedirs("data/train_data/"+label)
#         #print("train_data/"+label)
#     if not os.path.exists("data/train_data/"+label + "/"+file):
#         move("data/train_images/"+file,"data/train_data/"+label + "/"+file)
#         #print("plant-pathology-2021-fgvc8/train_images/"+file,"train_data/"+label + "/"+file)


# ##### Imports

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# ##### Viewing the data

# In[6]:


import pathlib
data_dir = pathlib.Path("data/train_data/")


# ##### Total Amount of images

# In[7]:


image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


# #### Example of an what an image data we are dealing with

# In[8]:


plant = list(data_dir.glob('complex/*'))
PIL.Image.open(str(plant[0]))


# ##### Parameters for loader

# In[9]:


batch_size = 32
img_height = 180
img_width = 180


# #### 80/20 Train Split

# In[10]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[11]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# ###### Image Labels

# In[12]:


class_names = train_ds.class_names
print(class_names)


# #### Visualization of the images

# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 15))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# ##### Keeps images in memory

# In[14]:


train_ds = train_ds.cache()
val_ds = val_ds.cache()


# ##### Standardize the data 

# In[15]:


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


# ##### Mapping a normalization to the dataset 

# In[16]:


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image)) 


# ##### Data Augmentation layers

# In[21]:


data_augmentation = keras.Sequential()
data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal',input_shape=(img_height,img_width,3)))
data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.2))
data_augmentation.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.2))


# In[22]:


data_augmentation.summary()


# ##### Examples of images being augmented

# In[23]:


plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


# ##### Creating the model

# In[24]:


num_classes = 12
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# #### Compile the model

# In[25]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# #### Model summary
# 
# View all the layers of the network using the model's `summary` method:

# In[26]:


model.summary()


# #### Train the model

# In[27]:


epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


# ##### Save the current model
# * only do this once

# In[29]:


# model.save("model_saved/")


# #### Load model from previous run
# * if you want to perform any predictions on images load the model

# In[34]:


tf.keras.models.load_model("model_saved/")


# ## Visualize training results

# * Plots of loss and accuracy on the training and validation sets.

# In[28]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ##### Figure 1.1
# * We can see that the validation and training set has a good fit

# ### Model Evalation

# ##### Extracting the true labels and predicted labels

# In[225]:


pred = list()
true = list()
pred_name = list()
true_name = list()
iteration_num = list()

counter = 1
for x, y in val_ds:
    #predicted values but for every batch
    predictions = model.predict(x)
    # true values to x
    labels = y.numpy()
    #print(labels)
    for dex in range(len(x)):
        score = tf.nn.softmax(predictions[dex])
        pred_name.append(class_names[np.argmax(score)])
        true_name.append(class_names[y[dex]])
        pred.append(np.argmax(score))
        true.append(labels[dex])
        iteration_num.append(counter)
    counter+=1


# ##### We can see that is 3726 which is our validation set

# In[226]:


print(len(pred),len(true))


# ##### Lets put the lists into a dataframe

# In[227]:


df = pd.DataFrame({"Iteration":iteration_num,"true_label":true_name,"pred_label":pred_name,"true_value":true,"pred_value":pred})
df.head()


# ##### Confusion maxtix for each label

# In[232]:


from sklearn.metrics import multilabel_confusion_matrix
mcm = multilabel_confusion_matrix(df["true_label"],df["pred_label"],labels=class_names)
mcm


# In[233]:


import io
from PIL import Image
import matplotlib.pyplot as plt
counter = 0
label_name = list()
true_neg = list()
false_pos = list()
false_neg = list()
true_pos = list()
for met in mcm:
#     print(class_names[counter])
#     print(met)
    tn, fp, fn, tp = met.ravel()
    true_neg.append(tn)
    false_pos.append(fp)
    false_neg.append(fn)
    true_pos.append(tp)
    label_name.append(class_names[counter])
    counter+=1


# ##### Dataframe of trues and falses

# In[234]:


df_stats = pd.DataFrame({"label_name":label_name,"true_neg":true_neg,"false_pos":false_pos,"false_neg":false_neg,"true_pos":true_pos})
df_stats.head()


# ##### Accuracy 
# * It is the closeness of the measurements to a specific value,
# !["accuracy"](https://wikimedia.org/api/rest_v1/media/math/render/svg/47deb47eb7ac214423d0a6afd05ec0af362fef9b)

# In[235]:


df_stats['accuracy'] = (df_stats.true_pos+df_stats.true_neg)/(df_stats.true_pos + df_stats.true_neg + df_stats.false_pos + df_stats.false_neg)
df_stats


# ##### Precision 
# * It tells you what fraction of predictions as a positive class were actually positive. 
# !["precision"](https://wikimedia.org/api/rest_v1/media/math/render/svg/d854b1544fc77735d575ce0d30e34d7f1eacf707)

# In[236]:


df_stats['precision'] = (df_stats['true_pos'])/(df_stats['true_pos']+df_stats['false_pos'])
df_stats.fillna(0,inplace=True)
df_stats.head()


# ##### Recall 
# * It tells you what fraction of all positive samples were correctly predicted as positive by the classifier. It is also known as True Positive Rate (TPR), Sensitivity, Probability of Detection.  
# !['recall'](https://wikimedia.org/api/rest_v1/media/math/render/svg/f02ea353bf60bfdd9557d2c98fe18c34cd8db835)

# In[237]:


df_stats["tpr"] = df_stats.true_pos/(df_stats.true_pos+df_stats.false_neg)


# In[238]:


df_stats.head()


# ##### Specificity (True Negative rate) 
# * Measures the proportion of negatives that are correctly identified (i.e. the proportion of those who do not have the condition (unaffected) who are correctly identified as not having the condition).
# !['specificity'](https://wikimedia.org/api/rest_v1/media/math/render/svg/8f2c867f0641e498ec8a59de63697a3a45d66b07)

# In[239]:


df_stats['tnr'] = (df_stats.true_neg)/(df_stats.true_neg+df_stats.false_pos)
df_stats.head()


# ##### Type I error (False Negative Rate)
# 
# * The first kind of error is the rejection of a true null hypothesis as the result of a test procedure. This kind of error is called a type I error (false positive) and is sometimes called an error of the first kind.
# !["fnr"](https://wikimedia.org/api/rest_v1/media/math/render/svg/2af486535eb235ed28c3063ed05fd21657b28410)

# In[240]:


df_stats['fnr'] = df_stats.false_neg/(df_stats.false_neg+df_stats.true_pos)
df_stats.head()


# ##### Type II error (False Positive Rate)
# 
# * The second kind of error is the failure to reject a false null hypothesis as the result of a test procedure. This sort of error is called a type II error (false negative) and is also referred to as an error of the second kind.
# ![fpr](https://wikimedia.org/api/rest_v1/media/math/render/svg/422d06161964ca90602ec8712cd211cb0d80da19)

# In[241]:


df_stats['fpr'] = df_stats.false_pos/(df_stats.false_pos+df_stats.true_neg)
df_stats.head()


# ##### F1-score
# * It combines precision and recall into a single measure. Mathematically it’s the harmonic mean of precision and recall. 
# ![f1_score](https://miro.medium.com/max/875/1*wUdjcIb9J9Bq6f2GvX1jSA.png)

# In[242]:


df_stats["f1_score"] = 2 * (df_stats.precision*df_stats.tpr)/(df_stats.precision + df_stats.tpr)
df_stats.fillna(0,inplace=True)
df_stats.head()


# ##### Error rate 
# * Error rate (ERR) is calculated as the number of all incorrect predictions divided by the total number of the dataset. The best error rate is 0.0, whereas the worst is 1.0.
# !["error_rate"](https://s0.wp.com/latex.php?latex=%5Cmathrm%7BERR+%3D+%5Cdisplaystyle+%5Cfrac%7BFP+%2B+FN%7D%7BTP+%2B+TN+%2B+FN+%2B+FP%7D+%3D+%5Cfrac%7BFP+%2B+FN%7D%7BP+%2B+N%7D%7D&bg=ffffff&fg=333333&s=0&c=20201002&zoom=2)

# In[243]:


df_stats["error_rate"] = (df_stats.false_pos + df_stats.false_neg)/(df_stats.true_pos + df_stats.true_neg + df_stats.false_neg + df_stats.false_pos)
df_stats.head()


# ##### Storing each confusion matrix with respect to their class_name in memory

# In[244]:


counter = 0
saved_cmp = list()
for met in mcm:
    counter+=1
    cmp = ConfusionMatrixDisplay(met)
    saved_cmp.append(cmp)


# ### Model Evaluation Visualization

# ##### Dataframe with all the True Positive, True Negative, False Positive, and False Negative for every iteration

# In[245]:


df


# ##### Examining each each iteration's confusion matrix

# In[246]:


samp = df[df["Iteration"] == 1]
cm_samp = confusion_matrix(samp["true_value"], samp["pred_value"])
samp.head()
labs = list()
set_amt = list(set(np.array(samp.true_value)).union(set(np.array(samp.pred_value))))

for i in set_amt:
    labs.append(class_names[i])

cm_samp
cmp_samp = ConfusionMatrixDisplay(cm_samp, display_labels=labs)
fig, ax = plt.subplots(figsize=(8,8))
cmp_samp.plot(ax=ax,xticks_rotation="vertical")


# ##### Final Dataframe with all the statistical classification analysis

# In[247]:


df_stats


# In[ ]:


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# ##### Classification Report
# * We can see that this function from sklearn learn aligns up perfectly with the dataframe above!

# In[248]:


from sklearn.metrics import classification_report
print(classification_report(df['true_label'],df['pred_label']))


# ##### Plotting confusion matrix for each class name

# In[249]:


fig, axs = plt.subplots(4,3)
axs[0,0].set_title(class_names[0])
axs[0,0] = saved_cmp[0].plot(ax= axs[0,0])
axs[0,1].set_title(class_names[1])
axs[0,1] = saved_cmp[1].plot(ax = axs[0,1])
axs[0,2].set_title(class_names[2])
axs[0,2] = saved_cmp[2].plot(ax = axs[0,2])

axs[1,0].set_title(class_names[3])
axs[1,0] = saved_cmp[3].plot(ax = axs[1,0])
axs[1,1].set_title(class_names[4])
axs[1,1] = saved_cmp[4].plot(ax = axs[1,1])
axs[1,2].set_title(class_names[5])
axs[1,2] = saved_cmp[5].plot(ax = axs[1,2])

axs[2,0].set_title(class_names[6])
axs[2,0] = saved_cmp[6].plot(ax = axs[2,0])
axs[2,1].set_title(class_names[7])
axs[2,1] = saved_cmp[7].plot(ax = axs[2,1])
axs[2,2].set_title(class_names[8])
axs[2,2] = saved_cmp[8].plot(ax = axs[2,2])

axs[3,0].set_title(class_names[9])
axs[3,0] = saved_cmp[9].plot(ax = axs[3,0])
axs[3,1].set_title(class_names[10])
axs[3,1] = saved_cmp[10].plot(ax = axs[3,1])
axs[3,2].set_title(class_names[11])
axs[3,2] = saved_cmp[11].plot(ax = axs[3,2])


fig.set_size_inches(12,12)
plt.tight_layout()  


# ##### Overall confusion matrix

# In[250]:


import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

cmp = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
cmp.plot(ax=ax,xticks_rotation="vertical")


# ##### Model prediction with the test images

# In[272]:


test_set = os.listdir("data/test_images/")
for leaf in test_set:

    img = keras.preprocessing.image.load_img("data/test_images/"+leaf, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
)


# In[ ]:




