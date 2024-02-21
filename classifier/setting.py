import os

class opt:
    save_dir = '../wgan_gp/' # model save directory
    gen_dir = os.path.join(save_dir, 'gen_images') # generated images directory
    train_dir = os.path.join(save_dir, 'train_set') # training set directory
    epochs_index = 190 # use generated images from this epoch
    batch_size = 32 # batch size
    num_classes = 10 # number of classes
    epochs = 10 # training epochs
    classify_index = 1 # classifier{i}.py index
    split_ratio = 0 # train/test split ratio