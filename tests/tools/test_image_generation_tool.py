"""Tests for tools/image_generation_tool.py."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.image_generation_tool import (
    _clean_env_secret,
    check_image_generation_requirements,
    image_generate_tool,
)


class TestImageGenerationRequirements:
    def test_accepts_openai_when_fal_missing(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        assert check_image_generation_requirements() is True

    def test_clean_env_secret_rejects_control_char_garbage(self, monkeypatch):
        monkeypatch.setenv("FAL_KEY", "\x1b")
        assert _clean_env_secret("FAL_KEY") == ""


class TestImageGenerateTool:
    def test_falls_back_to_openai_and_saves_local_image(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        fake_png = base64.b64encode(b"fake-image-bytes").decode("ascii")
        mock_response = SimpleNamespace(data=[SimpleNamespace(b64_json=fake_png)])
        mock_client = MagicMock()
        mock_client.images.generate.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(
                image_generate_tool(
                    prompt="Dog in the cosmos",
                    aspect_ratio="square",
                    output_format="png",
                )
            )

        assert result["success"] is True
        output_path = tmp_path / "generated-images" / next(iter((tmp_path / "generated-images").iterdir())).name
        assert result["image"] == str(output_path)
        assert output_path.read_bytes() == b"fake-image-bytes"
        mock_client.images.generate.assert_called_once_with(
            model="gpt-image-1.5",
            prompt="Dog in the cosmos",
            size="1024x1024",
            quality="high",
        )

    def test_openai_url_response_is_returned_directly(self, monkeypatch):
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

        mock_response = SimpleNamespace(data=[SimpleNamespace(url="https://example.com/dog.png")])
        mock_client = MagicMock()
        mock_client.images.generate.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            result = json.loads(image_generate_tool(prompt="Dog in the cosmos"))

        assert result == {"success": True, "image": "https://example.com/dog.png"}
