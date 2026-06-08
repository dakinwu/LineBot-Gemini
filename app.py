from collections import deque
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
from google.generativeai import client as genai_client
from google.generativeai.types import generation_types
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
import uvicorn

from market_data import build_market_context_text
from prompts import after_hours_report_prompt, extract_facts_prompt, morning_report_prompt

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


def _get_env_int(name, default=None):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %r", name, raw, default)
        return default


def _get_env_float(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %r", name, raw, default)
        return default

# Configure Line Bot API (v3)
configuration = Configuration(access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# Reuse LINE API client (keep Flask single-thread or single worker for safety)
_line_api_client = ApiClient(configuration)
_line_bot_api = MessagingApi(_line_api_client)

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
VISION_MODEL_NAME = os.getenv("GEMINI_VISION_MODEL", "gemini-3-flash-preview")
# Multimodal calls (many images) often need more than the default gRPC deadline.
GEMINI_TIMEOUT_SECONDS = _get_env_float("GEMINI_TIMEOUT_SECONDS", 300.0)
GEMINI_IMAGE_BATCH_SIZE = max(1, _get_env_int("GEMINI_IMAGE_BATCH_SIZE", 3))
# Text-only synthesis (batch merge, morning report) can be large; allow a separate ceiling.
GEMINI_SYNTHESIS_TIMEOUT_SECONDS = _get_env_float(
    "GEMINI_SYNTHESIS_TIMEOUT_SECONDS",
    max(GEMINI_TIMEOUT_SECONDS, 600.0),
)
GEMINI_RATE_LIMIT_MAX_CALLS = max(1, _get_env_int("GEMINI_RATE_LIMIT_MAX_CALLS", 5))
GEMINI_RATE_LIMIT_WINDOW_SECONDS = max(
    1.0,
    _get_env_float("GEMINI_RATE_LIMIT_WINDOW_SECONDS", 60.0),
)
GEMINI_RATE_LIMIT_SAFETY_SECONDS = max(
    0.0,
    _get_env_float("GEMINI_RATE_LIMIT_SAFETY_SECONDS", 0.25),
)
# If > 0: when image count >= this, skip the single all-images request and use batches first
# (avoids wasted retries when Gemini often times out on large multimodal payloads).
GEMINI_DIRECT_BATCH_MIN_IMAGES = max(0, _get_env_int("GEMINI_DIRECT_BATCH_MIN_IMAGES", 0))
model = genai.GenerativeModel(VISION_MODEL_NAME)
_gemini_rate_lock = threading.Lock()
_gemini_call_starts = deque()

VOOM_IMAGES_DIR = "voom_images"
MAX_VOOM_IMAGES = _get_env_int("MAX_VOOM_IMAGES")
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
NOTION_PARENT_PAGE_MORNING = os.getenv("NOTION_PARENT_PAGE_MORNING_URL")
NOTION_PARENT_PAGE_AFTER_HOURS = os.getenv("NOTION_PARENT_PAGE_AFTER_HOURS_URL")

NOTION_VERSION = "2022-06-28"
NOTION_BLOCK_LIMIT = 100
NOTION_APPEND_BATCH_SIZE = 50

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
    resp = requests.request(
        method,
        url,
        headers=_notion_headers(),
        json=payload,
        timeout=30,
    )
    if resp.status_code >= 400:
        snippet = (resp.text or "")[:800]
        logger.error(
            "Notion API 失敗：%s %s — %s",
            method,
            resp.status_code,
            snippet,
        )
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
    if not content or not content.strip():
        raise ValueError("分析結果是空的，已停止建立 Notion 頁面")
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
            "PATCH",
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


def _analysis_prompt(prompt_template, image_paths, start_index=0):
    image_labels = [
        f"Image {i + start_index + 1}: {os.path.basename(path)}"
        for i, path in enumerate(image_paths)
    ]
    prompt = prompt_template.replace("{image_labels}", "\n".join(image_labels))
    return prompt, image_labels


def _batched_image_paths(image_paths, batch_size):
    for start in range(0, len(image_paths), batch_size):
        yield start, image_paths[start:start + batch_size]


def _extract_generation_text(response):
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


def _generation_debug_summary(response):
    details = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback:
        details.append(f"prompt_feedback={prompt_feedback}")
    try:
        for idx, cand in enumerate(getattr(response, "candidates", None) or [], start=1):
            finish_reason = getattr(cand, "finish_reason", None)
            safety_ratings = getattr(cand, "safety_ratings", None)
            details.append(
                f"candidate {idx}: finish_reason={finish_reason}, "
                f"safety_ratings={safety_ratings}"
            )
    except Exception as err:
        details.append(f"candidate_debug_error={type(err).__name__}: {err}")
    return "; ".join(details) if details else "no response details available"


def _extract_generation_text_or_raise(response, context):
    text = _extract_generation_text(response)
    if text:
        return text
    raise RuntimeError(
        f"Gemini 沒有回傳可寫入 Notion 的分析文字（{context}）。"
        f"回應資訊：{_generation_debug_summary(response)}"
    )


def _wait_for_gemini_rate_slot():
    while True:
        with _gemini_rate_lock:
            now = time.monotonic()
            window_start = now - GEMINI_RATE_LIMIT_WINDOW_SECONDS
            while _gemini_call_starts and _gemini_call_starts[0] <= window_start:
                _gemini_call_starts.popleft()

            if len(_gemini_call_starts) < GEMINI_RATE_LIMIT_MAX_CALLS:
                _gemini_call_starts.append(now)
                return

            oldest = _gemini_call_starts[0]
            wait_seconds = (
                GEMINI_RATE_LIMIT_WINDOW_SECONDS
                - (now - oldest)
                + GEMINI_RATE_LIMIT_SAFETY_SECONDS
            )

        wait_seconds = max(wait_seconds, 0.1)
        logger.info(
            "Gemini rate limit reached (%s calls / %.1fs); waiting %.1fs",
            GEMINI_RATE_LIMIT_MAX_CALLS,
            GEMINI_RATE_LIMIT_WINDOW_SECONDS,
            wait_seconds,
        )
        time.sleep(wait_seconds)


def _generate_gemini_response(parts, timeout_seconds=None):
    deadline = (
        GEMINI_TIMEOUT_SECONDS if timeout_seconds is None else float(timeout_seconds)
    )
    request = model._prepare_request(
        contents=parts,
        generation_config=None,
        safety_settings=None,
        tools=None,
        tool_config=None,
    )
    if model._client is None:
        model._client = genai_client.get_default_generative_client()

    try:
        _wait_for_gemini_rate_slot()
        raw_response = model._client.generate_content(
            request=request,
            retry=None,
            timeout=deadline,
        )
        return generation_types.GenerateContentResponse.from_response(raw_response)
    except Exception as err:
        logger.error("Gemini 請求失敗：%s", _format_exception(err))
        raise


def _build_batch_synthesis_prompt(reports, full_prompt):
    return (
        "The following are partial outputs from multiple image batches of the same LINE VOOM post.\n"
        "Combine them into one single output that follows the original instructions.\n"
        "Do not mention batching, missing images, or the analysis process.\n"
        "Remove duplication and keep only grounded details.\n\n"
        f"Original instructions:\n{full_prompt}\n\n"
        "Partial analyses:\n"
        f"{chr(10).join(reports)}"
    )


def _synthesize_batch_reports(batch_reports, full_prompt):
    """Apply original instructions to merged batch outputs."""
    synthesis_prompt = _build_batch_synthesis_prompt(list(batch_reports), full_prompt)
    response = _generate_gemini_response(
        [synthesis_prompt],
        timeout_seconds=GEMINI_SYNTHESIS_TIMEOUT_SECONDS,
    )
    return _extract_generation_text_or_raise(response, "batch synthesis")


def _analyze_voom_images_in_batches(image_paths, prompt_template, full_prompt):
    batch_reports = []
    for batch_number, (start_index, batch_paths) in enumerate(
        _batched_image_paths(image_paths, GEMINI_IMAGE_BATCH_SIZE),
        start=1,
    ):
        batch_prompt, batch_labels = _analysis_prompt(
            prompt_template,
            batch_paths,
            start_index=start_index,
        )
        batch_prompt = (
            f"{batch_prompt}\n\n"
            "You are only receiving this subset of images from the same LINE VOOM post. "
            "Analyze only these images and do not invent details from images that are not shown."
        )
        batch_parts = [batch_prompt]
        for path in batch_paths:
            batch_parts.append(_image_part(path))
        batch_response = _generate_gemini_response(batch_parts)
        batch_text = _extract_generation_text_or_raise(
            batch_response,
            f"batch {batch_number}",
        )
        batch_reports.append(
            f"Batch {batch_number} ({', '.join(batch_labels)}):\n{batch_text}"
        )

    if len(batch_reports) == 1:
        return batch_reports[0]

    return _synthesize_batch_reports(batch_reports, full_prompt)


def _analyze_voom_images_core(image_paths, prompt_template):
    if not image_paths:
        return "No VOOM images found."

    prompt, _ = _analysis_prompt(prompt_template, image_paths)

    direct_batch = (
        GEMINI_DIRECT_BATCH_MIN_IMAGES > 0
        and len(image_paths) >= GEMINI_DIRECT_BATCH_MIN_IMAGES
        and len(image_paths) > 1
        and GEMINI_IMAGE_BATCH_SIZE < len(image_paths)
    )
    if direct_batch:
        logger.info(
            "Skipping single multimodal request (%s images >= GEMINI_DIRECT_BATCH_MIN_IMAGES=%s); "
            "using batches of %s",
            len(image_paths),
            GEMINI_DIRECT_BATCH_MIN_IMAGES,
            GEMINI_IMAGE_BATCH_SIZE,
        )
        return _analyze_voom_images_in_batches(image_paths, prompt_template, prompt)

    parts = [prompt]
    for path in image_paths:
        parts.append(_image_part(path))

    response = _generate_gemini_response(parts)
    return _extract_generation_text_or_raise(response, "full image analysis")


def analyze_voom_images(image_paths, prompt_template):
    if not image_paths:
        return "找不到圖片，無法分析 VOOM 貼文。"

    image_labels = [f"圖{i+1}: {os.path.basename(p)}" for i, p in enumerate(image_paths)]
    prompt = prompt_template.replace("{image_labels}", "\n".join(image_labels))

    parts = [prompt]
    for path in image_paths:
        parts.append(_image_part(path))

    response = _generate_gemini_response(parts)
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


# Override the legacy implementation above with the timeout-aware path.
analyze_voom_images = _analyze_voom_images_core


def _morning_prompt_from_facts(facts_text):
    return morning_report_prompt.replace("{facts_table}", facts_text.strip())


def _after_hours_prompt_from_facts(facts_text):
    return after_hours_report_prompt.replace("{facts_table}", facts_text.strip())


def _facts_with_market_context(facts_text, mode):
    facts_text = (facts_text or "").strip()
    try:
        market_context = build_market_context_text(mode=mode)
    except Exception as err:
        logger.exception("FinMind market context fetch failed")
        market_context = (
            "## 程式補充資料（FinMind API）\n"
            f"- 資料狀態：FinMind 擷取失敗，{_format_exception(err)}。"
        )

    market_context = (market_context or "").strip()
    if not market_context:
        return facts_text
    if not facts_text:
        return market_context
    return f"{facts_text}\n\n{market_context}"


def analyze_morning_voom_images(image_paths):
    facts_text = analyze_voom_images(image_paths, extract_facts_prompt)
    logger.info("Gemini extracted facts text length: %s", len(facts_text or ""))
    facts_text = _facts_with_market_context(facts_text, "morning")
    logger.info("Facts text with FinMind context length: %s", len(facts_text or ""))

    synthesis_prompt = _morning_prompt_from_facts(facts_text)
    synthesis_response = _generate_gemini_response(
        [synthesis_prompt],
        timeout_seconds=GEMINI_SYNTHESIS_TIMEOUT_SECONDS,
    )
    return _extract_generation_text_or_raise(
        synthesis_response,
        "morning report synthesis",
    )


def analyze_after_hours_voom_images(image_paths):
    facts_text = analyze_voom_images(image_paths, extract_facts_prompt)
    logger.info("Gemini extracted facts text length: %s", len(facts_text or ""))
    facts_text = _facts_with_market_context(facts_text, "after_hours")
    logger.info("Facts text with FinMind context length: %s", len(facts_text or ""))

    synthesis_prompt = _after_hours_prompt_from_facts(facts_text)
    synthesis_response = _generate_gemini_response(
        [synthesis_prompt],
        timeout_seconds=GEMINI_SYNTHESIS_TIMEOUT_SECONDS,
    )
    return _extract_generation_text_or_raise(
        synthesis_response,
        "after hours report synthesis",
    )


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
    logger.info("Downloaded %s VOOM images for analysis", len(images))

    if mode == "morning":
        analysis_text = analyze_morning_voom_images(images)
    else:
        analysis_text = analyze_after_hours_voom_images(images)
    logger.info("Gemini analysis text length: %s", len(analysis_text or ""))

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
