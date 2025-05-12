import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import google.generativeai as genai
import pycld2 as cld2
import langid
from dotenv import load_dotenv
# conda activate c:/Users/dakin/Desktop/LGBoT/.conda

# 載入環境變數
load_dotenv()

app = Flask(__name__)

# 設定 Line Bot API
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# 設定 Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

def detect_language(text):
    # try:
    #     isReliable, textBytesFound, details = cld2.detect(text)
    #     if isReliable:
    #         lang_code = details[0][1]  # 例如 'zh-Hant', 'zh-Hans', 'en'
    #         return lang_code.lower()
    try:
        lang, confidence = langid.classify(text)
        return lang  # 返回語言代碼，例如 'zh', 'en'
    except Exception:
        pass
    return "unknown"

def translate_text(text):
    try:
        # 檢測輸入文字的語言
        lang = detect_language(text)
        
        if lang.startswith('zh'):
            # 如果是中文，翻譯成英文
            prompt = f"請將以下中文翻譯成英文，只需要回覆一個版本的翻譯結果：\n{text}"
        else:
            # 如果是其他語言，翻譯成中文
            prompt = f"請將以下文字翻譯成繁體中文，只需要回覆一個版本的翻譯結果：\n{text}"
        
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"翻譯時發生錯誤：{str(e)}"

@app.route("/callback", methods=['POST'])
def callback():
    # 取得 X-Line-Signature header 的值
    signature = request.headers['X-Line-Signature']

    # 取得請求內容
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # 取得使用者傳送的訊息
    user_message = event.message.text
    
    # 進行翻譯
    translated_text = translate_text(user_message)
    
    # 回傳翻譯結果
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=translated_text)
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000) 