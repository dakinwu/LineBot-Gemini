# Line 翻譯機器人

這是一個使用 Line Messaging API 和 Google Gemini API 實作的翻譯機器人。它可以自動檢測輸入的文字語言：
- 如果輸入中文，會翻譯成英文
- 如果輸入其他語言，會翻譯成中文

## 安裝需求

1. Python 3.7 或更新版本
2. Line Messaging API 帳號和頻道
3. Google Cloud 帳號和 API 金鑰

## 設置步驟

1. 安裝相依套件：
   ```bash
   pip install -r requirements.txt
   ```

2. 複製 `.env.example` 到 `.env`：
   ```bash
   cp .env.example .env
   ```

3. 在 `.env` 檔案中填入您的：
   - LINE_CHANNEL_ACCESS_TOKEN
   - LINE_CHANNEL_SECRET
   - GOOGLE_API_KEY

4. 啟動應用程式：
   ```bash
   python app.py
   ```

5. 使用 ngrok 或其他工具將應用程式公開到網際網路

6. 在 Line Developers Console 中設定 Webhook URL：
   - URL 格式：`https://您的網域/callback`

## 使用方式

1. 將機器人加入為好友
2. 傳送任何文字訊息給機器人
3. 機器人會自動檢測語言並進行翻譯

## 注意事項

- 請確保您的 API 金鑰安全，不要公開或上傳到版本控制系統
- 建議在正式環境中使用 HTTPS
- 確保您的伺服器有足夠的記憶體和處理能力 