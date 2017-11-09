
from __future__ import print_function
import numpy as np
import os

from scipy import ndimage
from six.moves import cPickle as pickle


data_root = '.' # Change me to store data elsewhere


num_classes = 10
np.random.seed(133)

test_folders = ['./notMNIST_small/A', './notMNIST_small/B', './notMNIST_small/C', './notMNIST_small/D',
                './notMNIST_small/E', './notMNIST_small/F', './notMNIST_small/G', './notMNIST_small/H', 
                './notMNIST_small/I', './notMNIST_small/J']
train_folders = ['./notMNIST_large_v2/A', './notMNIST_large_v2/B', './notMNIST_large_v2/C', './notMNIST_large_v2/D',
                 './notMNIST_large_v2/E', './notMNIST_large_v2/F', './notMNIST_large_v2/G', './notMNIST_large_v2/H',
                 './notMNIST_large_v2/I', './notMNIST_large_v2/J']

image_size = 28  # Pixel width and height.
pixel_depth = 255.0

def load_letter(folder, min_num_images):
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (ndimage.imread(image_file).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          #pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
          print(pickle.HIGHEST_PROTOCOL)
          pickle.dump(dataset, f, 2)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def main():
  train_datasets = maybe_pickle(train_folders, 100)
  test_datasets = maybe_pickle(test_folders, 50)
  train_size = 1000
  valid_size = 500
  test_size = 500

  valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, valid_size)
  _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

  print('Training dataset and labels shape:', train_dataset.shape, train_labels.shape)
  print('Validation dataset and labels shape:', valid_dataset.shape, valid_labels.shape)
  print('Testing dataset and labels shape:', test_dataset.shape, test_labels.shape)

  train_dataset, train_labels = randomize(train_dataset, train_labels)
  test_dataset, test_labels = randomize(test_dataset, test_labels)
  valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

  pickle_file = os.path.join(data_root, 'notMNIST.pickle')

  try:
    f = open(pickle_file, 'wb')
    save = {
      'train_dataset': train_dataset,
      'train_labels': train_labels,
      'valid_dataset': valid_dataset,
      'valid_labels': valid_labels,
      'test_dataset': test_dataset,
      'test_labels': test_labels,
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()
  except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

  statinfo = os.stat(pickle_file)
  print('Compressed pickle size:', statinfo.st_size)

if __name__ == '__main__':
  main()


