# Qwen LINE 訊息翻譯 Bot

這個分支是一個獨立的 LINE 翻譯 Bot，使用本機 Ollama
`qwen3.5:9b`，不會呼叫 Gemini，也不需要 Notion。

翻譯方向沿用本專案早期的 Gemini 版本：

- 中文訊息翻譯成自然英文
- 其他語言翻譯成台灣繁體中文
- 保留網址、名稱、數字、`@mention`、主題標籤、換行與 emoji

## 執行需求

- Python 3.10 以上
- Ollama
- 已下載的 `qwen3.5:9b`
- LINE Messaging API Channel
- 可將本機 `5000` port 暴露為 HTTPS 的服務，例如 ngrok

## 安裝

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

編輯 `.env`，填入：

```dotenv
LINE_CHANNEL_ACCESS_TOKEN=你的 LINE Channel Access Token
LINE_CHANNEL_SECRET=你的 LINE Channel Secret
```

確認 Ollama 看得到模型：

```powershell
ollama list
```

若清單沒有 `qwen3.5:9b`：

```powershell
ollama pull qwen3.5:9b
```

## 啟動

先啟動 Ollama，再啟動 Bot：

```powershell
ollama serve
```

另一個終端：

```powershell
python app.py
```

服務預設位於 `http://127.0.0.1:5000`。

健康檢查：

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

## LINE Webhook

對外提供 HTTPS 網址後，把 LINE Developers Console 的 Webhook URL 設為：

```text
https://你的網域/callback
```

例如使用 ngrok：

```powershell
ngrok http 5000
```

同一個 LINE Channel 同時只能指向一個 Webhook。測試這個分支時若沿用晨報
Bot 的 Channel，需要暫時切換 Webhook；要讓兩個 Bot 同時在線，請使用兩個
LINE Channel。

## 環境變數

| 變數 | 預設值 | 用途 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama API 位址 |
| `OLLAMA_MODEL` | `qwen3.5:9b` | 翻譯模型 |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | 單次翻譯逾時秒數 |
| `OLLAMA_KEEP_ALIVE` | `10m` | 模型留在記憶體的時間 |
| `OLLAMA_THINK` | `false` | 是否啟用思考輸出 |
| `MAX_INPUT_CHARS` | `5000` | 單則輸入長度上限 |
| `CACHE_TTL_SECONDS` | `1800` | 相同翻譯的快取秒數 |
| `CACHE_MAX_SIZE` | `512` | 快取筆數上限 |

## 測試

```powershell
python -m unittest discover -s tests -v
```

測試會模擬 Ollama 回應，不會真的載入 9B 模型。
