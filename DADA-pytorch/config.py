import os

class Config(object):
    # 获取当前文件所在目录的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # 使用 os.path.join 来构建 data_dir 的路径，使其相对于 base_dir
    data_dir = os.path.join(base_dir, 'data', 'cifar-10-batches-py')
    gpu_id = '0'
    train_batch_size = 100
    test_batch_size = 500
    aug = True
    G_epochs = 500
    epochs = 700
    count = 1000
    save_img = os.path.join(base_dir, 'save_img_G' + str(G_epochs) + "_total" + str(epochs) + "_" + str(count) + "per_class/")
    lr = 0.0003
    fre_print = 1
    seed = 1
    workers = 4
    num_classes = 10
    logs = os.path.join(base_dir, 'logs', str(count) + " per_class")
    save_model = os.path.join(base_dir, 'save_model', str(count) + " per_class")
    gen_epoch = 200
    n_times = 20  # 定义每个类别要生成图像的次数
    save_img_G = False
    save_epoch = 50