import torch
from Nets import _G  # 确保这与你定义Generator的代码一致
import numpy as np
import os
import matplotlib.pyplot as plt

from config import Config
opt = Config()

def save_png_custom(gen_data, save_dir, class_id, iteration, epoch):
    gen_data_np = gen_data.cpu().numpy()
    img_bhwc = np.transpose(gen_data_np, (0, 2, 3, 1))
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.vstack(img_bhwc), aspect="auto")
    plt.savefig(f"{save_dir}\\pngs\\cifar_class_{class_id}_sample_{epoch}_{iteration}.png")
    plt.close()

# 初始化一个足够大的NumPy数组来存储所有生成的图像
# 假设每张图像的尺寸为3x32x32（CIFAR-10的尺寸），可以根据需要调整
n_times = opt.n_times  # 定义每个类别要生成图像的次数
num_classes = opt.num_classes

num_images = num_classes * n_times
all_images = np.empty((num_images, 3, 32, 32), dtype=np.float32)
all_labels = np.empty((num_images,), dtype=int)
image_counter = 0  # 用于跟踪当前保存到大数组中的图像索引

epoch = opt.gen_epoch
model_path = os.path.join(opt.save_model, 'G_epoch_{}.pth'.format(epoch))
z_dim = 100
G = _G(num_classes=num_classes)
state_dict = torch.load(model_path)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
G.load_state_dict(new_state_dict)
G.eval()
if torch.cuda.is_available():
    G = G.cuda()

save_dir = os.path.join(opt.base_dir, 'gen_png')

for class_id in range(num_classes):  # 遍历每个类别
    for n in range(n_times):  # 对每个类别生成n次图像
        noise = torch.randn(1, z_dim).cuda() if torch.cuda.is_available() else torch.randn(1, z_dim)
        labels = torch.full((1,), class_id, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.full((1,), class_id, dtype=torch.long)
        with torch.no_grad():
            generated_image = G(noise, labels)
        if opt.save_img_G:
            save_png_custom(generated_image, save_dir, class_id, n, epoch)

        generated_image= generated_image.cpu().numpy()

        # 保存生成的图像到大数组中
        all_images[image_counter, :, :, :] = generated_image
        # 保存生成图像的label到数组中
        all_labels[image_counter] = class_id
        image_counter += 1



# 保存大数组到.npy文件
np.save(os.path.join(save_dir, 'all_generated_images.npy'), all_images)
np.save(os.path.join(save_dir, 'all_generated_labels.npy'), all_labels)

# 如果需要保存每张图片的类别和在该类别中的索引，可以额外创建一个数组或者使用pickle保存一个包含图片信息的字典
image_info = [(i // n_times, i % n_times) for i in range(num_images)]
np.save(os.path.join(save_dir, 'generated_image_info.npy'), image_info)

