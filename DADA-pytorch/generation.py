import torch
from Nets import _G  # 确保这与你定义Generator的代码一致
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings

from config import Config
from tqdm import tqdm
opt = Config()
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")



def save_png_custom(gen_data, save_dir, class_id, iteration, epoch):
    gen_data_np = gen_data.cpu().numpy()
    img_bhwc = np.transpose(gen_data_np, (0, 2, 3, 1))

    # 归一化图像数据到 [0, 1]
    img_bhwc = (img_bhwc - img_bhwc.min()) / (img_bhwc.max() - img_bhwc.min())

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(np.vstack(img_bhwc), aspect="auto")
    if not os.path.exists(os.path.join(save_dir, f'label{class_id}', f'epoch{epoch}')):
        os.makedirs(os.path.join(save_dir, f'label{class_id}', f'epoch{epoch}'))
    plt.savefig(os.path.join(save_dir, f'label{class_id}', f'epoch{epoch}', f'gen_{class_id}_{iteration}.png'))
    plt.close()


n_times = opt.n_times  
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

save_dir = os.path.join(opt.base_dir, 'gen_images')

for class_id in tqdm(range(num_classes), desc='Class Progress'):
    # 对每个类别生成n次图像
    for n in tqdm(range(n_times), desc=f'Generating for class {class_id}', leave=False):
        noise = torch.randn(1, z_dim).cuda() if torch.cuda.is_available() else torch.randn(1, z_dim)
        labels = torch.full((1,), class_id, dtype=torch.long).cuda() if torch.cuda.is_available() else torch.full((1,), class_id, dtype=torch.long)
        with torch.no_grad():
            generated_image = G(noise, labels)
        
        save_png_custom(generated_image, save_dir, class_id, n, epoch)
