# app.py
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn

# CNNモデル定義
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

app = FastAPI()

# 静的ファイルの設定
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS許可
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデル準備
model = CNN()
model.load_state_dict(torch.load("model_cnn.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x)  # 画像を反転（白背景→黒背景）
])

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read()))
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probabilities[0][pred].item()
    
    return {
        "prediction": pred,
        "confidence": confidence,
        "probabilities": [probabilities[0][i].item() for i in range(10)]
    }
