# model_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*12*32, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# 学習処理
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='.', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 学習ループ（5エポック）
for epoch in range(5):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        
        if batch_idx % 200 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # エポック終了時の精度計算
    train_acc = 100. * correct / total
    print(f'Epoch {epoch+1}: Train Accuracy: {train_acc:.2f}%')
    
    # テストデータでの評価
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            test_loss += criterion(output, y).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(y).sum().item()
    
    test_acc = 100. * correct / len(test_data)
    print(f'Epoch {epoch+1}: Test Accuracy: {test_acc:.2f}%\n')

torch.save(model.state_dict(), "model_cnn.pth")
print("✅ モデル保存完了")
