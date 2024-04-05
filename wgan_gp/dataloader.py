from torchvision import datasets, transforms
from PIL import Image
import os

# 设置你想要筛选的标签
target_label = 9

# 设置保存图像的目录为相对路径
save_dir = os.path.join("train_set", f"label{target_label}")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 加载MNIST数据集
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# 筛选并保存图像
for idx, (img, label) in enumerate(mnist):
    if label == target_label:
        img_path = os.path.join(save_dir, f"mnist_{idx}_label_{label}.png")
        img = transforms.ToPILImage()(img)
        img.save(img_path)
