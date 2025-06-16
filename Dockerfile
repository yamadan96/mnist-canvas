# ベースイメージ
FROM python:3.10-slim

# 作業ディレクトリを作成
WORKDIR /app

# 必要ファイルをコピー
COPY . /app

# 依存ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# ポート番号を指定
EXPOSE 8000

# 起動コマンド
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

