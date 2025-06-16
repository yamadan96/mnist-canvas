const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// キャンバスを白で初期化
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

let drawing = false;

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mouseup', () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener('mousemove', (e) => {
  if (drawing) {
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
});

// タッチイベントも追加（モバイル対応）
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  const rect = canvas.getBoundingClientRect();
  const x = touch.clientX - rect.left;
  const y = touch.clientY - rect.top;
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener('touchend', (e) => {
  e.preventDefault();
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (drawing) {
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
  }
});

let hasDrawn = false;

// 描画開始時にヒントを隠す
canvas.addEventListener('mousedown', () => {
  if (!hasDrawn) {
    document.getElementById('canvasHint').style.opacity = '0';
    hasDrawn = true;
  }
});

canvas.addEventListener('touchstart', () => {
  if (!hasDrawn) {
    document.getElementById('canvasHint').style.opacity = '0';
    hasDrawn = true;
  }
});

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // 白で塗りつぶし
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  
  // ヒントを再表示
  document.getElementById('canvasHint').style.opacity = '1';
  hasDrawn = false;
  
  // 結果をリセット
  document.getElementById('resultContainer').style.display = 'none';
  document.getElementById('loading').style.display = 'none';
}

async function predict() {
  // ローディング表示
  document.getElementById('loading').style.display = 'block';
  document.getElementById('resultContainer').style.display = 'none';
  
  try {
    canvas.toBlob(async blob => {
      const form = new FormData();
      form.append("file", blob, "image.png");
      
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: form
      });
      
      const data = await res.json();
      
      // ローディング非表示
      document.getElementById('loading').style.display = 'none';
      
      // 結果表示
      document.getElementById('resultValue').innerText = data.prediction;
      document.getElementById('confidence').innerText = `信頼度: ${(data.confidence * 100).toFixed(1)}%`;
      document.getElementById('resultContainer').style.display = 'block';
      
      // 結果に応じてアニメーション
      const resultValue = document.getElementById('resultValue');
      resultValue.style.transform = 'scale(1.2)';
      setTimeout(() => {
        resultValue.style.transform = 'scale(1)';
      }, 200);
      
    });
  } catch (error) {
    document.getElementById('loading').style.display = 'none';
    alert('予測中にエラーが発生しました。もう一度お試しください。');
  }
}
