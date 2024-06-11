# ML
pytorch_cifar10
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Parameters
batch_size = 64
lr = 0.001
num_epochs = 20

# Data Augmentation and Normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 Dataset
train_dataset = torchvision.datasets.CIFAR10(root='./Dataset/', train=True, transform=transform_train, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./Dataset/', train=False, transform=transform_test, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Display an example image
image, label = next(iter(train_loader))
print(f"Label: {label[0]}")
print(f"Image shape: {image[0].shape}")

image = image[0].numpy().transpose((1, 2, 0))
plt.imshow(image)
plt.axis('off')
plt.show()

# Define the CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(256 * 4 * 4, 10)  # Adjust the input features based on the final spatial dimensions

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

# Initialize the model, loss function, optimizer, and learning rate scheduler
model = CustomCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# Training and validation
num_steps = len(train_loader)
model.train()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    corrects = 0
    
    for step, (imgs, lbls) in enumerate(train_loader):
        out = model(imgs)
        loss_val = loss_fn(out, lbls)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        total_loss += loss_val.item()
        _, predicted = torch.max(out.data, 1)
        corrects += (predicted == lbls).sum().item()
        
        if (step + 1) % 100 == 0:
            print(f'Train, Epoch [{epoch + 1}/{num_epochs}] Step [{step + 1}/{num_steps}] Loss: {total_loss / (step + 1):.4f}')
    
    lr_sch.step()

    # Validation
    model.eval()
    corrects = 0
    with torch.no_grad():
        for step, (imgs, lbls) in enumerate(test_loader):
            out = model(imgs)
            loss_val = loss_fn(out, lbls)
            _, predicted = torch.max(out.data, 1)
            corrects += (predicted == lbls).sum().item()
    
    accuracy = 100. * corrects / len(test_dataset)
    print(f'Validation, Epoch [{epoch + 1}/{num_epochs}] Accuracy: {accuracy:.2f}%')

# Test phase
model.eval()
corrects = 0
num_steps = len(test_loader)
with torch.no_grad():
    for step, (imgs, lbls) in enumerate(test_loader):
        out = model(imgs)
        predicted = torch.argmax(out, 1)
        corrects += torch.sum(predicted == lbls).item()
        print(f'Step [{step + 1}/{num_steps}] Acc: {100. * corrects / ((step + 1) * batch_size):.4f}')

# Save the model
torch.save(model.state_dict(), 'model.pth')
