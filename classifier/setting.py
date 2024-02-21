import os

class opt:
    save_dir = '../wgan_gp/'
    gen_dir = os.path.join(save_dir, 'gen_images')
    train_dir = os.path.join(save_dir, 'train_set')
    epochs_index = 0
    batch_size = 32
    num_classes = 10
    epochs = 10
    classify_index = 1
    split_ratio = 0.8