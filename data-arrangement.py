import os, shutil
import numpy as np

# Create directories of samples
images_dir = r'.\indoorCVPR_09\Images'

train_dir = r'.\indoorCVPR_09\train'
os.mkdir(train_dir)

val_dir = r'.\indoorCVPR_09\val'
os.mkdir(val_dir)

test_dir = r'.\indoorCVPR_09\test'
os.mkdir(test_dir)

# Images belong to 67 classes
categories = os.listdir(images_dir)
print('Number of classes: ', len(categories))
# 67

# Create class directories
for category in categories:
    path = os.path.join(train_dir, category)
    os.mkdir(path)
    path = os.path.join(val_dir, category)
    os.mkdir(path)
    path = os.path.join(test_dir, category)
    os.mkdir(path)
    
# The text files specify which images to use while
# training and which ones to use while testing. We'll
# do one more split in train data to gain validation
# data.

train_fnames = []
with open('TrainImages.txt', 'r') as f:
      train_fnames = [line.strip() for line in f]
for fname in train_fnames:
      src = os.path.join(images_dir, fname)
      dst = os.path.join(train_dir, fname)
      shutil.move(src, dst)
      
test_fnames = []
with open('TestImages.txt', 'r') as f:
      test_fnames = [line.strip() for line in f]
for fname in test_fnames:
      src = os.path.join(images_dir, fname)
      dst = os.path.join(test_dir, fname)
      shutil.move(src, dst)
    
# 10 images per class for validation set
for class_name in os.listdir(train_dir):
    src_dir = os.path.join(train_dir, class_name)
    fnames_for_class = os.listdir(src_dir)[:10]
    for fname in fnames_for_class:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(val_dir, class_name, fname)
        shutil.move(src, dst)
        
# Checking if there's any imbalance in train or
# test set

class_sizes_train = list()
for class_dir in os.listdir(train_dir):
    class_sizes_train.append(len(os.listdir(os.path.join(train_dir, class_dir))))
print('Number of samples in each class in training set: ', class_sizes_train)
# [70, 70, 72, 71, 72, 72, 69, 70, 70, 70, 71, 72, 71, 72, 70, 72, 72, 72, 70, 69, 71, 69, 72, 69, 73, 71, 70, 72, 70, 69, 72, 69, 70, 67, 69, 68, 70, 69, 68, 68, 70, 70, 70, 69, 70, 68, 70, 67, 70, 69, 71, 70, 70, 70, 70, 67, 71, 70, 71, 69, 68, 70, 72, 68, 69, 69, 59]
print('Variance in sizes: ', np.var(class_sizes_train))
# 3.7090666072621965

class_sizes_val = list()
for class_dir in os.listdir(val_dir):
    class_sizes_val.append(len(os.listdir(os.path.join(val_dir, class_dir))))
print('Number of samples in each class in validation set: ', class_sizes_val)
# [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
print('Variance in sizes: ', np.var(class_sizes_val))
# 0.0

class_sizes_test = list()
for class_dir in os.listdir(test_dir):
    class_sizes_test.append(len(os.listdir(os.path.join(test_dir, class_dir))))
print('Number of samples in each class in test set:  ', class_sizes_test)
# [20, 20, 18, 19, 18, 18, 21, 20, 20, 20, 19, 18, 19, 18, 20, 18, 18, 18, 20, 21, 19, 21, 18, 21, 17, 19, 20, 18, 20, 21, 18, 21, 20, 23, 21, 22, 20, 21, 22, 22, 20, 20, 20, 21, 20, 22, 20, 23, 20, 21, 19, 20, 20, 20, 20, 23, 19, 20, 19, 21, 22, 20, 18, 22, 21, 21, 21]
print('Variance in sizes: ', np.var(class_sizes_test))
# 1.9402985074626866

# No dataset skewed, so we won't have to handle such
# problem (we would handle it with
# under-sampling/over-sampling or use data
# augmentation if imbalance had been the case).