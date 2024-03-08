import os
import torchvision
import torchvision.transforms as transforms

def save_cifar10_images_by_class():
    # 直接加载 CIFAR-10 训练数据集，不需要转换
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10-batches-py', train=True, download=True)

    # 定义保存图片的根目录
    root_dir = './train_set'

    # 遍历数据集中的每一张图片和其对应的标签
    for idx, (image, label) in enumerate(trainset):
        # 为每个标签创建一个目录（如果尚不存在）
        label_dir = os.path.join(root_dir, f'label{label}')
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # 定义图片的保存路径
        image_path = os.path.join(label_dir, f'image_{idx}.png')

        # 保存图片
        image.save(image_path)

    print("CIFAR-10 images have been saved by class.")

if __name__ == '__main__':
    save_cifar10_images_by_class()
