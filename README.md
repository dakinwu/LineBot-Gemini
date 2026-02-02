# LineBot-Gemini（實際用途版）

這是一個本地運行的 LINE Bot 專案：當使用者貼上 LINE VOOM 文章網址時，程式會用 Playwright 開啟瀏覽器下載該貼文的圖片，接著用 Gemini Vision 進行分析，最後把結果寫入 Notion，並在 LINE 回傳 Notion 連結。

此專案偏向「個人/小團隊工具」，不保證長期穩定，VOOM 網頁結構改版就可能需要調整 selector。

## 主要功能
- 接收 LINE 訊息，抓出 VOOM 連結
- 用 Playwright 開啟瀏覽器並下載 VOOM 圖片
- 將圖片餵給 Gemini Vision 分析並產生文字結果
- 結果寫入 Notion（標題含分析時間），LINE 回傳 Notion 連結
- 內文支援簡易 Markdown 轉 Notion 區塊（標題/清單/粗體）

## 環境需求
- Python 3.8+（建議 3.10 以上）
- 已啟用 LINE Messaging API 的 Channel
- Google Gemini API Key
- Notion Internal Integration Secret（用於建立頁面）
- Windows/macOS/Linux 皆可（本專案目前以 Windows 實測為主）

## 安裝
```bash
pip install -r requirements.txt
```

## 環境變數
建立 `.env`，內容如下：
```
LINE_CHANNEL_ACCESS_TOKEN=你的LINE_ACCESS_TOKEN
LINE_CHANNEL_SECRET=你的LINE_CHANNEL_SECRET
GOOGLE_API_KEY=你的GOOGLE_API_KEY

# 可選
GEMINI_VISION_MODEL=gemini-2.5-flash

# Notion
NOTION_TOKEN=你的NOTION_INTEGRATION_SECRET
NOTION_PARENT_PAGE_URL=https://www.notion.so/1ecb98676e4e806d8480eaefa758945f
```

## 啟動
```bash
python app.py
```

預設會在 `http://0.0.0.0:5000` 啟動。

## Line Webhook 設定
你需要把外部可存取的 URL 指向 `/callback`，例如：
```
https://<your-domain-or-ngrok>/callback
```

本地測試可用 ngrok：
```bash
ngrok http 5000
```

## 使用方式
在 LINE 對話中貼上 VOOM 貼文網址，例如：
```
https://voom.line.me/post/xxxxxxxx
```
或
```
https://linevoom.line.me/post/xxxxxxxx
```

Bot 會建立一個 Notion 頁面，並回覆該頁面連結。

## 單獨使用 VOOM 下載器
你可以直接執行：
```bash
python voom_downloader.py <VOOM 文章網址>
```
下載的圖片會放在 `voom_images/`。

## 注意事項（很現實的部分）
- Playwright 目前使用「有 UI 的瀏覽器模式」；如果你關掉瀏覽器視窗，流程會中斷。
- VOOM DOM 變動很頻繁，`voom_downloader.py` 的 selector 可能失效。
- 如果貼文需要登入才能看，Playwright 會卡住或抓不到圖。
- 下載與分析都需要時間；Notion API 若回傳錯誤，請檢查 Token、權限與父頁面是否已分享給 Integration。
- 內文的 Markdown 只支援基本格式（標題/清單/粗體）。

## 常見問題
- **抓不到圖片**：通常是 selector 失效或貼文被登入限制，請更新 `voom_downloader.py` 的 selector。
- **TargetClosedError**：表示瀏覽器/頁面被關閉或崩潰，重新開啟再試。
- **Gemini 回傳很慢**：視圖片張數與模型而定，可嘗試限制圖片數量或改用較快模型。
- **Notion 沒寫入**：確認 Integration 已加入該頁面、`NOTION_TOKEN` 正確、`NOTION_PARENT_PAGE_URL` 可讀。

## 目錄結構
```
LineBot-Gemini/
├─ app.py                # LINE Bot 主程式
├─ voom_downloader.py    # VOOM 圖片下載器（Playwright）
├─ voom_images/          # 下載後的圖片
├─ requirements.txt
└─ .env                  # 環境變數（自行建立）
```
