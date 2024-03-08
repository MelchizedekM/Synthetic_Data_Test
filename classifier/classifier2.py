import os
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import time
import numpy as np
from torch.utils.data import Subset

from setting import opt
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    def calculate_accuracy(model, data_loader):
        model.eval()  
        correct = 0
        total = 0
        with torch.no_grad(): 
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

    def predict_image(image_path, model, transform):
        model.eval()
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0).to(device)  # 确保图像也在正确的设备上
        with torch.no_grad():
            outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

    class CustomDataset(Dataset):
        def __init__(self, gen_root_dir, train_set_root_dir, transform=None, split='train', split_ratio=0.8, epochs_index=0):
            self.transform = transform
            self.data_files = []
            self.labels = []
            self.split = split
            self.split_ratio = split_ratio
            self.epochs_index = epochs_index
            self._load_dataset( gen_root_dir, train_set_root_dir)

        def _load_dataset(self,  gen_root_dir, train_set_root_dir):
            epochs = self.epochs_index

            if self.split == 'train' and epochs in range(0,1000):
                data_path = os.path.join(gen_root_dir, f'all_generated_images_{epoch}.npy')
                label_path = os.path.join(gen_root_dir, f'all_generated_label_{epoch}.npy')
                
                if os.path.exists(data_path) and os.path.exists(label_path):
                    self.data_files.append(data_path)
                    self.labels.append(label_path)
                else:
                    print(f"Epoch {epoch} data or label file is missing.")

            elif self.split == 'test' and  epochs in range(0,1000):
                print("No generated images for test set.")
            elif epochs == 'false':
                print("No generated images used.")
            else:
                print("Invalid epochs_index type.")

            data_path_train = os.path.join(train_set_root_dir, f'all_images_cf.npy')
            label_path_train = os.path.join(train_set_root_dir, f'all_labels_cf.npy')
            
            if os.path.exists(data_path_train) and os.path.exists(label_path_train):
                self.data_files.append(data_path_train)
                self.labels.append(label_path_train)
            else:
                print(f"Data or label file is missing.")

            split_point = int(len(self.data_files) * self.split_ratio)
            if self.split == 'train':
                self.data_files = self.data_files[:split_point]
                self.labels = self.labels[:split_point]
            else: 
                self.data_files = self.data_files[split_point:]
                self.labels = self.labels[split_point:]

            print(f"Loaded {len(self.data_files)} samples.")

        def __len__(self):
            return len(self.data_files)

        def __getitem__(self, idx):
            data_path = self.data_files[idx]
            label_path = self.labels[idx]

            image = np.load(data_path)
            labels = np.load(label_path)
            
            image_tensor = torch.from_numpy(data).float()
            label_tensor = torch.from_numpy(labels).long() 

            if self.transform:
                image = self.transform(image)

            return image, label


    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomDataset(gen_root_dir=opt.gen_dir,
                                train_set_root_dir=opt.train_dir,
                                transform=transform,
                                epochs_index=opt.epochs_index,
                                split='train',
                                split_ratio=opt.split_ratio)

    test_dataset = CustomDataset(gen_root_dir=opt.gen_dir,
                                train_set_root_dir=opt.train_dir,
                                transform=transform,
                                epochs_index=opt.epochs_index,
                                split='test',
                                split_ratio=opt.split_ratio)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    num_classes = opt.num_classes
    resnet = models.resnet50()
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device) 


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(opt.epochs):
        print(f"Epoch {epoch + 1}/{opt.epochs} begins.")
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{opt.epochs}", leave=True, ncols=100)
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = resnet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{opt.epochs} ends. Loss: {loss.item():.4f}. Time: {train_time:.2f}s.")

        # select 1000 images from the test set to calculate accuracy

        indices = np.random.choice(range(len(test_dataset)), size=1000, replace=False)
        subset = Subset(test_dataset, indices)
        test_loader_1000 = DataLoader(subset, batch_size=opt.batch_size, shuffle=False)

        accuracy = calculate_accuracy(resnet, test_loader_1000)

        print("Training accuracy: ", calculate_accuracy(resnet, train_loader), "%")
        print("Test accuracy: ", accuracy, "%")

        if not os.path.exists(f'Acc._{opt.classify_index}'):
            os.makedirs(f'Acc._{opt.classify_index}')

        if epoch == 0:
            with open(os.path.join(f'Acc._{opt.classify_index}', f'accuracy_epoch{opt.epochs_index}_train{opt.split_ratio}.txt'), 'w') as f:
                f.write(f'Traing Acc. for epoch{epoch+1} is {calculate_accuracy(resnet, train_loader):.2f}%'+ '\n')
                f.write(f'Test Acc. for epoch{epoch+1} is {accuracy:.2f}%'+ '\n')

        with open(os.path.join(f'Acc._{opt.classify_index}', f'accuracy_epoch{opt.epochs_index}_train{opt.split_ratio}.txt'), 'a') as f:
            f.write(f'Traing Acc. for epoch{epoch+1} is {calculate_accuracy(resnet, train_loader):.2f}%'+ '\n')
            f.write(f'Test Acc. for epoch{epoch+1} is {accuracy:.2f}%'+ '\n')
        
        progress_bar.close()

if __name__ == "__main__":
    main()

