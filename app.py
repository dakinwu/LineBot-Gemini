import os
import time
from threading import Lock
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage as LineTextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import google.generativeai as genai
import langid
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
model = genai.GenerativeModel('gemini-2.5-flash')

# Simple in-memory cache with TTL
CACHE_TTL_SECONDS = 30 * 60
_cache_lock = Lock()
_cache = {}  # (lang, text) -> (timestamp, translated_text)
_cache_max_size = 512


def _cache_get(key):
    now = time.time()
    with _cache_lock:
        item = _cache.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts > CACHE_TTL_SECONDS:
            _cache.pop(key, None)
            return None
        return value


def _cache_set(key, value):
    now = time.time()
    with _cache_lock:
        if len(_cache) >= _cache_max_size:
            # Drop the oldest item to cap memory
            oldest_key = min(_cache.items(), key=lambda kv: kv[1][0])[0]
            _cache.pop(oldest_key, None)
        _cache[key] = (now, value)



def detect_language(text):
    """Detect the language of the input text."""
    try:
        lang, confidence = langid.classify(text)
        return lang  # Example values: 'zh', 'en'
    except Exception:
        pass
    return "unknown"


def translate_text(text):
    """Translate text using Gemini."""
    try:
        # Detect input language
        lang = detect_language(text)

        cache_key = (lang, text)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        if lang.startswith('zh'):
            # Translate to English if input is Chinese
            prompt = (
                "Translate into modern, casual English. "
                "Return a single best translation only. "
                "No lists, no numbering, no explanations. "
                "Keep meaning, tone, and punctuation.\n"
                f"{text}"
            )
        else:
            # Translate to Chinese if input is not Chinese
            prompt = (
                "Translate into modern, casual Traditional Chinese. "
                "Return a single best translation only. "
                "No lists, no numbering, no explanations. "
                "Keep meaning, tone, and punctuation.\n"
                f"{text}"
            )

        response = model.generate_content(prompt)
        translated = response.text.strip()
        _cache_set(cache_key, translated)
        return translated

    except Exception as e:
        return f"Translation error: {str(e)}"


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
    user_message = event.message.text

    # Translate text
    translated_text = translate_text(user_message)

    # Send translated reply
    _line_bot_api.reply_message(
        ReplyMessageRequest(
            reply_token=event.reply_token,
            messages=[LineTextMessage(text=translated_text)],
        )
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
