import unittest
from unittest.mock import Mock, patch

import app


class TranslationTests(unittest.TestCase):
    def setUp(self):
        app._cache_clear()

    def test_chinese_targets_english(self):
        source, target, description = app.translation_target("今天天氣很好")

        self.assertTrue(source.startswith("zh"))
        self.assertEqual(target, "en")
        self.assertIn("English", description)

    def test_english_targets_traditional_chinese(self):
        source, target, description = app.translation_target("The weather is great today.")

        self.assertEqual(source, "en")
        self.assertEqual(target, "zh-TW")
        self.assertIn("Traditional Chinese", description)

    def test_japanese_targets_traditional_chinese(self):
        source, target, _ = app.translation_target("今日はいい天気です")

        self.assertEqual(source, "ja")
        self.assertEqual(target, "zh-TW")

    @patch("app.requests.post")
    def test_translate_uses_qwen_and_caches_result(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "message": {"role": "assistant", "content": "Good morning!"}
        }
        mock_post.return_value = response

        first = app.translate_text("早安")
        second = app.translate_text("早安")

        self.assertEqual(first, "Good morning!")
        self.assertEqual(second, "Good morning!")
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], app.OLLAMA_MODEL)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["messages"][-1]["content"], "早安")

    def test_long_line_reply_is_split(self):
        text = "a" * (app.LINE_MESSAGE_CHARS + 10)

        chunks = app._split_line_messages(text)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= app.LINE_MESSAGE_CHARS for chunk in chunks))

    def test_empty_message_is_rejected(self):
        with self.assertRaises(app.TranslationError):
            app.translate_text("   ")


if __name__ == "__main__":
    unittest.main()
