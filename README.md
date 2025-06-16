# mnist-canvas

## 概要

このプロジェクトは、手書き数字認識を行うMLモデルを組み込んだWebアプリケーションです。ユーザーはブラウザ上でCanvasに数字を描くことで、リアルタイムにAIによる分類結果を得ることができます。

- バックエンド：FastAPI（Python）
- フロントエンド：HTML / JavaScript（Canvas）
- 機械学習モデル：PyTorchによるCNN（MNISTデータセットで学習済み）
- デプロイ：Docker化済み、Renderにて公開

## 使用技術

- Python 3.10
- FastAPI
- PyTorch
- HTML / JavaScript
- Docker
- Render（PaaSホスティング）

## デモ画面（イメージ）

![mnist-canvas-demo](docs/demo.png)

## 使い方（ローカル実行）

```bash
# ビルド
docker build -t mnist-app .

# 起動
docker run -p 8000:8000 mnist-app
```

ブラウザで [http://localhost:8000](http://localhost:8000) を開き、数字を描画して「判定」ボタンを押してください。

## 公開URL

[mnist-canvas Webアプリ (Render公開)](https://mnist-canvas.onrender.com)

## 作者

Yuto Yamada
