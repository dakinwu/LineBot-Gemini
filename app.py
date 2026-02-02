import os
import re
import mimetypes
import subprocess
import sys
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage as LineTextMessage,
    PushMessageRequest,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Line Bot API (v3)
configuration = Configuration(access_token=os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))

# Reuse LINE API client (keep Flask single-thread or single worker for safety)
_line_api_client = ApiClient(configuration)
_line_bot_api = MessagingApi(_line_api_client)

# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
VISION_MODEL_NAME = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
model = genai.GenerativeModel(VISION_MODEL_NAME)

VOOM_IMAGES_DIR = "voom_images"
MAX_VOOM_IMAGES = None

def _clear_voom_images():
    os.makedirs(VOOM_IMAGES_DIR, exist_ok=True)
    for name in os.listdir(VOOM_IMAGES_DIR):
        path = os.path.join(VOOM_IMAGES_DIR, name)
        if os.path.isfile(path):
            os.remove(path)

def _extract_first_url(text):
    match = re.search(r"(https?://\S+)", text)
    if not match:
        return None
    url = match.group(1)
    return url.rstrip(").,;，。】》>」")

def _download_voom_images(url):
    _clear_voom_images()
    cmd = [sys.executable, "voom_downloader.py", url]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result

def _load_voom_images():
    if not os.path.isdir(VOOM_IMAGES_DIR):
        return []
    files = [
        os.path.join(VOOM_IMAGES_DIR, name)
        for name in os.listdir(VOOM_IMAGES_DIR)
        if os.path.isfile(os.path.join(VOOM_IMAGES_DIR, name))
    ]
    files.sort()
    if MAX_VOOM_IMAGES is None:
        return files
    return files[:MAX_VOOM_IMAGES]

def _image_part(path):
    mime_type, _ = mimetypes.guess_type(path)
    if not mime_type:
        mime_type = "image/jpeg"
    with open(path, "rb") as f:
        data = f.read()
    return {"mime_type": mime_type, "data": data}

def analyze_voom_images(image_paths):
    if not image_paths:
        return "????????????? VOOM ?????????"

    image_labels = [f"圖{i+1}: {os.path.basename(p)}" for i, p in enumerate(image_paths)]
    prompt = (
        "你是『專業投資研究助理（Sell-side 等級）』，"
        "任務是根據 LINE VOOM 提供的多張投資晨報圖片，"
        "進行結構化整理、客觀解讀與研究分析。\n\n"

        "【最高原則（務必遵守）】\n"
        "1️⃣ 僅能使用圖片中「明確可辨識」的文字、數字、表格與圖表資訊。\n"
        "2️⃣ 嚴禁使用任何金融常識、歷史經驗、市場慣性、過往循環或一般推論。\n"
        "3️⃣ 不得補充圖片未出現的事件、政策、財報、產業背景或宏觀因素。\n"
        "4️⃣ 若圖片無法支持判斷，請明確寫出：「圖片未顯示明確趨勢或方向性訊號」。\n"
        "5️⃣ 所有結論必須可回溯至圖片內容，並於段落結尾標註圖號（如：[圖12]）。\n\n"

        "【資訊使用優先順序】\n"
        "請依下列順序判斷可信度：\n"
        "（1）數字表格 ＞（2）K 線與技術圖 ＞（3）成交量 ＞（4）圖片內文字說明。\n"
        "若文字敘述與數據不一致，以數據為準。\n\n"

        "【任務流程（請嚴格依序執行，不得跳步）】\n"

        "Step 1｜圖片主題相關性判斷\n"
        "- 判斷整組圖片是否與股票、投資、金融市場直接相關。\n"
        "- 若非直接相關，請說明原因並停止分析。\n\n"

        "Step 2｜整體晨報重點摘要（2–4 句）\n"
        "- 說明圖片整體在傳達的市場狀態。\n"
        "- 聚焦於：指數表現、資金動向、族群強弱、市場氣氛。\n\n"

        "Step 3｜多方與空方訊號整理（不得推論）\n"
        "【多方訊號】\n"
        "- 條列圖片中支持偏多的『事實性現象』\n"
        "- 不可使用預期、可能、將會等語句\n"
        "- 每一點皆須標註圖號\n\n"

        "【空方或壓力因素】\n"
        "- 條列圖片中出現的下跌、轉弱、賣超、量縮等現象\n"
        "- 不可延伸原因\n\n"

        "Step 4｜仍需觀察的不確定因素\n"
        "- 僅限圖片中尚未形成明確方向的資訊\n"
        "- 例如：法人分歧、量能變化、族群輪動尚未確認等\n\n"

        "Step 5｜時間尺度判斷\n"
        "- 判斷圖片內容較偏向：短線 / 中線 / 長線觀察\n"
        "- 若僅為單日資料，請明確標示為『短線觀察為主』\n\n"

        "Step 6｜整體操作態度（擇一）\n"
        "- 保守 / 中性 / 積極\n"
        "- 僅能根據圖片內容支持，不得使用盤後或未來推論。\n\n"

        "【輸出格式規範】\n"
        "- 使用繁體中文\n"
        "- 條列清楚、標題分明\n"
        "- 不使用情緒性、鼓動性、預測性語言\n"
        "- 不出現『我認為』『我推測』『可能會』等主觀表述\n\n"

        "【圖片順序對照】\n"
        + "\\n".join(image_labels)
    )
    
    parts = [prompt]
    for path in image_paths:
        parts.append(_image_part(path))

    response = model.generate_content(parts)
    return response.text.strip()

def _sentence_split(text):
    # Keep punctuation as sentence endings; include newline as a boundary.
    parts = re.split(r"(?<=[。！？!?]|\n)", text)
    return [p for p in parts if p]


def split_text_for_line(text, limit=4900):
    sentences = _sentence_split(text)
    chunks = []
    current = ""
    for s in sentences:
        add = s
        if len(current) + len(add) > limit:
            if current:
                chunks.append(current)
                current = ""
            # If a single sentence is too long, hard-split it
            while len(add) > limit:
                chunks.append(add[:limit])
                add = add[limit:]
        current += add
    if current:
        chunks.append(current)
    return chunks


@app.route("/callback", methods=['POST'])
def callback():
    """LINE Webhook callback."""
    # Get X-Line-Signature header
    signature = request.headers.get('X-Line-Signature', '')

    # Get request body text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.warning("Invalid signature. Please check your channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """Handle incoming text messages."""
    # Get incoming user message text
    user_message = event.message.text.strip()
    print(f"[debug] User message raw: {event.message.text!r}", flush=True)
    print(f"[debug] User message stripped: {user_message!r}", flush=True)
    url = _extract_first_url(user_message)
    print(f"[debug] Extracted URL: {url!r}", flush=True)

    if not url or not ("voom.line.me" in url or "linevoom.line.me" in url):
        reply_text = "??? LINE VOOM ??????? https://voom.line.me/post/... ? https://linevoom.line.me/post/...??"
    else:
        try:
            result = _download_voom_images(url)
            if result.returncode != 0:
                reply_text = (
                    "?????????? VOOM ????????????\n"
                    f"???{(result.stderr or result.stdout).strip()}"
                )
            else:
                image_paths = _load_voom_images()
                if not image_paths:
                    reply_text = "????????????? VOOM ?????????"
                else:
                    reply_text = analyze_voom_images(image_paths)
        except Exception as e:
            reply_text = f"?????{str(e)}"

    chunks = split_text_for_line(reply_text, limit=4900)
    messages = [LineTextMessage(text=chunk) for chunk in chunks]

    # LINE reply allows only a limited number of messages; push the rest if needed.
    reply_batch = messages[:5]
    _line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=reply_batch,
        )
    )

    remaining = messages[5:]
    if remaining:
        source = event.source
        target_id = getattr(source, "user_id", None) or getattr(source, "group_id", None) or getattr(source, "room_id", None)
        if target_id:
            for i in range(0, len(remaining), 5):
                batch = remaining[i:i + 5]
                _line_bot_api.push_message(
                    PushMessageRequest(
                        to=target_id,
                        messages=batch,
                    )
                )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
