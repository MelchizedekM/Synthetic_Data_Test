import os
import torch
import torch.nn as nn
import torch.optim as optim
import os

from setting import opt
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, gen_root_dir, train_set_root_dir, transform=None, epochs_index=0):
        self.transform = transform
        self.samples = []
        self._load_dataset(gen_root_dir, train_set_root_dir, epochs_index)

    def _load_dataset(self, gen_root_dir, train_set_root_dir, epochs_index):
        for label in os.listdir(gen_root_dir):
            label_dir = os.path.join(gen_root_dir, label)
            epoch_dir = os.path.join(label_dir, f'epoch{epochs_index}')
            if os.path.exists(epoch_dir):
                for img_name in os.listdir(epoch_dir):
                    self.samples.append((os.path.join(epoch_dir, img_name), int(label.replace('label', ''))))
        
        for label in os.listdir(train_set_root_dir):
            train_label_dir = os.path.join(train_set_root_dir, label)
            for img_name in os.listdir(train_label_dir):
                self.samples.append((os.path.join(train_label_dir, img_name), int(label.replace('label', ''))))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_path, model, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 添加batch维度
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

dataset = CustomDataset(root_dir=opt.save_dir, transform=transform, epochs_index=opt.epochs_index)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

num_classes = opt.num_classes
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

resnet.train()
for epoch in range(opt.epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

