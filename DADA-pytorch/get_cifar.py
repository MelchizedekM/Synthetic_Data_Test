import numpy as np
import pickle
import os
import tarfile
from urllib.request import urlretrieve
from config import Config
import sys


opt = Config()

def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        filename = url.split('/')[-1]
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(data_dir)

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='latin1')
    images = batch['data'].reshape((-1, 3, 32, 32)).astype('float32')
    labels = np.array(batch['labels'])
    return images, labels

def save_cifar10_numpy(data_dir):
    # Download and extract CIFAR-10 if necessary
    maybe_download_and_extract(data_dir)
    
    # Initialize arrays to store all images and labels
    all_images = np.empty((0, 3, 32, 32), dtype='float32')
    all_labels = np.empty((0,), dtype='int')

    # Load each batch and append to arrays
    for i in range(1, 6):
        images, labels = load_cifar10_batch(os.path.join(data_dir, 'cifar-10-batches-py', f'data_batch_{i}'))
        all_images = np.append(all_images, images, axis=0)
        all_labels = np.append(all_labels, labels, axis=0)
    
    # Load test batch
    test_images, test_labels = load_cifar10_batch(os.path.join(data_dir, 'cifar-10-batches-py', 'test_batch'))
    all_images = np.append(all_images, test_images, axis=0)
    all_labels = np.append(all_labels, test_labels, axis=0)
    
    # Save images and labels to .npy files
    save_path = os.path.join(opt.base_dir, 'gen_png')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'all_images_cf.npy'), all_images)
    np.save(os.path.join(save_path, 'all_labels_cf.npy'), all_labels)
    
    # Generate and save labels info (class_id, index in class)
    num_classes = opt.num_classes
    image_info = []
    for class_id in range(num_classes):
        indices = np.where(all_labels == class_id)[0]
        for idx, global_idx in enumerate(indices):
            image_info.append((class_id, idx))
    np.save(os.path.join(save_path, 'image_info_cf.npy'), image_info)
    
    print(f"All images and labels info saved to {save_path}")

data_dir = Config.data_dir
save_cifar10_numpy(data_dir)
