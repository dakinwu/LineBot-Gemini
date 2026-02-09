from datetime import datetime
import logging
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
import google.generativeai as genai
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage as LineTextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import requests

from prompts import after_hours_report_prompt, morning_report_prompt
import uvicorn

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure Line Bot API (v3)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# Reuse LINE API client (keep Flask single-thread or single worker for safety)
_line_api_client = ApiClient(configuration)
_line_bot_api = MessagingApi(_line_api_client)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
VISION_MODEL_NAME = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")
model = genai.GenerativeModel(VISION_MODEL_NAME)

VOOM_IMAGES_DIR = "voom_images"
MAX_VOOM_IMAGES = None
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_PARENT_PAGE_MORNING = os.getenv("NOTION_PARENT_PAGE_MORNING_URL")
NOTION_PARENT_PAGE_AFTER_HOURS = os.getenv("NOTION_PARENT_PAGE_AFTER_HOURS_URL")

NOTION_VERSION = "2022-06-28"
NOTION_BLOCK_LIMIT = 100
NOTION_APPEND_BATCH_SIZE = 50
NOTION_RETRY_STATUSES = {429, 502, 503, 504}
NOTION_MAX_RETRIES = 3
NOTION_RETRY_BASE_DELAY = 1.0

MODE_PREFIX_MAP = {
    "1": "morning",
    "2": "after_hours",
}
_PREFIX_RE = re.compile(
    r"^\s*(?:\[|【|（|\()?\s*(?P<prefix>[12])\s*(?:\]|】|）|\))?\s*(?:[:：\-—\s]+)"
)


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


def _detect_report_mode(text):
    if not text:
        return "morning", text
    match = _PREFIX_RE.match(text)
    if not match:
        return "morning", text
    prefix = match.group("prefix")
    mode = MODE_PREFIX_MAP.get(prefix, "morning")
    return mode, text[match.end():].lstrip()


def _extract_notion_page_id(value):
    if not value:
        return None
    match = re.search(r"([0-9a-fA-F]{32})", value)
    if not match:
        return None
    raw = match.group(1).lower()
    return f"{raw[0:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:32]}"


def _notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _notion_request(method, url, payload):
    for attempt in range(NOTION_MAX_RETRIES + 1):
        resp = requests.request(
            method,
            url,
            headers=_notion_headers(),
            json=payload,
            timeout=30,
        )
        if resp.status_code < 400:
            return resp
        if resp.status_code not in NOTION_RETRY_STATUSES:
            return resp
        if attempt >= NOTION_MAX_RETRIES:
            return resp
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = NOTION_RETRY_BASE_DELAY * (2 ** attempt)
        else:
            delay = NOTION_RETRY_BASE_DELAY * (2 ** attempt)
        time.sleep(delay)
    return resp


def _format_exception(err):
    if err is None:
        return "未知錯誤"
    err_str = str(err).strip()
    if err_str:
        return f"{type(err).__name__}: {err_str}"
    return f"{type(err).__name__}: 未知錯誤"


def _chunk_text(text, limit=1800):
    if not text:
        return []
    chunks = []
    current = ""
    for ch in text:
        if len(current) + 1 > limit:
            chunks.append(current)
            current = ""
        current += ch
    if current:
        chunks.append(current)
    return chunks


def _rich_text_from_markdown(text):
    parts = []
    pattern = re.compile(r"\*\*(.+?)\*\*")
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            parts.append({"type": "text", "text": {"content": text[last:match.start()]}})
        bold_text = match.group(1)
        if bold_text:
            parts.append({
                "type": "text",
                "text": {"content": bold_text},
                "annotations": {"bold": True},
            })
        last = match.end()
    if last < len(text):
        parts.append({"type": "text", "text": {"content": text[last:]}})
    return parts or [{"type": "text", "text": {"content": ""}}]


def _text_blocks_from_content(content):
    blocks = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": []},
            })
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        number_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        bullet_match = re.match(r"^[-*•]\s+(.+)$", stripped)

        if heading_match:
            raw_level = len(heading_match.group(1))
            level = raw_level if raw_level <= 2 else 2
            text = heading_match.group(2)
            block_type = f"heading_{level}"
            for chunk in _chunk_text(text):
                blocks.append({
                    "object": "block",
                    "type": block_type,
                    block_type: {
                        "rich_text": _rich_text_from_markdown(chunk),
                    },
                })
            continue

        if number_match:
            text = number_match.group(2)
            for chunk in _chunk_text(text):
                blocks.append({
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": _rich_text_from_markdown(chunk),
                    },
                })
            continue

        if bullet_match:
            text = bullet_match.group(1)
            for chunk in _chunk_text(text):
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": _rich_text_from_markdown(chunk),
                    },
                })
            continue

        for chunk in _chunk_text(stripped):
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": _rich_text_from_markdown(chunk),
                },
            })
    return blocks


def _create_notion_page(title, content, voom_url, parent_page):
    if not NOTION_TOKEN:
        raise ValueError("NOTION_TOKEN 未設定")
    if not parent_page:
        raise ValueError("NOTION_PARENT_PAGE 未設定")
    parent_id = _extract_notion_page_id(parent_page)
    if not parent_id:
        raise ValueError("NOTION_PARENT_PAGE_URL/ID 未設定或格式不正確")

    header_blocks = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {"type": "text", "text": {"content": "VOOM 連結: "}},
                    {
                        "type": "text",
                        "text": {"content": voom_url, "link": {"url": voom_url}},
                    },
                ],
            },
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": []},
        },
    ]

    all_children = header_blocks + _text_blocks_from_content(content)
    initial_children = all_children[:NOTION_BLOCK_LIMIT]

    payload = {
        "parent": {"page_id": parent_id},
        "properties": {
            "title": {
                "title": [{"type": "text", "text": {"content": title}}],
            }
        },
        "children": initial_children,
    }

    resp = _notion_request(
        "POST",
        "https://api.notion.com/v1/pages",
        payload,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Notion API 錯誤 {resp.status_code}: {resp.text}")
    data = resp.json()
    page_id = data.get("id")
    page_url = data.get("url")

    remaining = all_children[NOTION_BLOCK_LIMIT:]
    while remaining:
        batch = remaining[:NOTION_APPEND_BATCH_SIZE]
        remaining = remaining[NOTION_APPEND_BATCH_SIZE:]
        append_resp = _notion_request(
            "POST",
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            {"children": batch},
        )
        if append_resp.status_code >= 400:
            raise RuntimeError(
                f"Notion API 錯誤 {append_resp.status_code}: {append_resp.text}"
            )

    return page_url


def _download_voom_images(url):
    _clear_voom_images()
    cmd = [sys.executable, "voom_downloader.py", url]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )


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


def _analysis_prompt_template(mode):
    return after_hours_report_prompt if mode == "after_hours" else morning_report_prompt


def _analysis_parent_page(mode):
    return NOTION_PARENT_PAGE_AFTER_HOURS if mode == "after_hours" else NOTION_PARENT_PAGE_MORNING


def _analysis_title(mode, analyzed_at):
    return f"{analyzed_at} 盤後整理" if mode == "after_hours" else f"{analyzed_at} 晨報整理"


def analyze_voom_images(image_paths, prompt_template):
    if not image_paths:
        return "找不到圖片，無法分析 VOOM 貼文。"

    image_labels = [f"圖{i+1}: {os.path.basename(p)}" for i, p in enumerate(image_paths)]
    prompt = prompt_template.replace("{image_labels}", "\n".join(image_labels))

    parts = [prompt]
    for path in image_paths:
        parts.append(_image_part(path))

    response = model.generate_content(parts)
    try:
        return response.text.strip()
    except Exception:
        texts = []
        try:
            for part in response.parts:
                if getattr(part, "text", None):
                    texts.append(part.text)
        except Exception:
            texts = []
        if not texts:
            try:
                for cand in response.candidates or []:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in content.parts or []:
                        if getattr(part, "text", None):
                            texts.append(part.text)
            except Exception:
                texts = []
        return "\n".join(texts).strip()


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
            while len(add) > limit:
                chunks.append(add[:limit])
                add = add[limit:]
        current += add
    if current:
        chunks.append(current)
    return chunks


def _push_text(target_id, text):
    if not target_id:
        return
    chunks = split_text_for_line(text, limit=4900)
    messages = [LineTextMessage(text=chunk) for chunk in chunks]
    for i in range(0, len(messages), 5):
        batch = messages[i:i + 5]
        _line_bot_api.push_message(
            PushMessageRequest(
                to=target_id,
                messages=batch,
            )
        )


def _reply_with_optional_push(reply_token, target_id, text):
    chunks = split_text_for_line(text, limit=4900)
    messages = [LineTextMessage(text=chunk) for chunk in chunks]
    reply_batch = messages[:5]
    _line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=reply_token,
            messages=reply_batch,
        )
    )
    remaining = messages[5:]
    if remaining and target_id:
        for i in range(0, len(remaining), 5):
            batch = remaining[i:i + 5]
            _line_bot_api.push_message(
                PushMessageRequest(
                    to=target_id,
                    messages=batch,
                )
            )


def _process_voom_sync(url, mode):
    result = _download_voom_images(url)
    if result.returncode != 0:
        raise RuntimeError(
            "下載 VOOM 圖片失敗。\n"
            f"錯誤訊息：{(result.stderr or result.stdout).strip()}"
        )
    images = _load_voom_images()
    if not images:
        raise RuntimeError("找不到圖片，無法分析 VOOM 貼文。")

    prompt = _analysis_prompt_template(mode)
    analysis_text = analyze_voom_images(images, prompt)

    analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = _analysis_title(mode, analyzed_at)
    parent_page = _analysis_parent_page(mode)
    notion_url = _create_notion_page(title, analysis_text, url, parent_page)
    return notion_url


def process_voom_background(url, mode, target_id):
    try:
        _push_text(target_id, "🔍 正在分析 VOOM 圖片…")
        notion_url = _process_voom_sync(url, mode)
        _push_text(target_id, f"✅ 分析完成\n{notion_url}")
    except Exception as e:
        err_msg = _format_exception(e)
        print(f"[error] {err_msg}\n{traceback.format_exc()}", flush=True)
        _push_text(target_id, f"❌ 分析失敗：{err_msg}")


@app.post("/callback")
async def callback(request: Request):
    """LINE Webhook callback."""
    signature = request.headers.get("X-Line-Signature", "")

    body_bytes = await request.body()
    body = body_bytes.decode("utf-8")
    logger.info("Request body: %s", body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        logger.warning("Invalid signature. Please check your channel secret.")
        raise HTTPException(status_code=400, detail="Invalid signature.")

    return PlainTextResponse("OK")


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """Handle incoming text messages."""
    user_message = event.message.text.strip()
    print(f"[debug] User message raw: {event.message.text!r}", flush=True)
    print(f"[debug] User message stripped: {user_message!r}", flush=True)
    mode, content_text = _detect_report_mode(user_message)
    print(f"[debug] Report mode: {mode!r}", flush=True)
    url = _extract_first_url(content_text)
    print(f"[debug] Extracted URL: {url!r}", flush=True)

    source = event.source
    target_id = (
        getattr(source, "user_id", None)
        or getattr(source, "group_id", None)
        or getattr(source, "room_id", None)
    )

    if not url or not ("voom.line.me" in url or "linevoom.line.me" in url):
        reply_text = (
            "請提供 LINE VOOM 文章網址，例如 https://voom.line.me/post/... 或 https://linevoom.line.me/post/...\n"
            "可在訊息前綴輸入「1」或「2」切換報告類型（1=晨報，2=盤後報告）。"
        )
        _reply_with_optional_push(event.reply_token, target_id, reply_text)
        return

    if target_id:
        reply_text = "📥 已收到 VOOM，開始分析，完成後會通知你"
        _reply_with_optional_push(event.reply_token, target_id, reply_text)
        threading.Thread(
            target=process_voom_background,
            args=(url, mode, target_id),
            daemon=True,
        ).start()
        return

    # Fallback: no target_id, process synchronously and reply once
    try:
        notion_url = _process_voom_sync(url, mode)
        _reply_with_optional_push(event.reply_token, None, f"已建立 Notion 頁面：{notion_url}")
    except Exception as e:
        err_msg = _format_exception(e)
        print(f"[error] {err_msg}\n{traceback.format_exc()}", flush=True)
        _reply_with_optional_push(event.reply_token, None, f"處理失敗：{err_msg}")


if __name__ == "__main__":

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=False)
