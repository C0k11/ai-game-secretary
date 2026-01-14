import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class OpenAIChatConfig:
    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout_s: int = 60
    temperature: float = 0.0
    max_tokens: int = 512
    enforce_json: bool = True


class OpenAIChatClient:
    def __init__(self, cfg: OpenAIChatConfig):
        self.cfg = cfg

    def _ollama_chat(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/api/chat"
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature,
            },
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "")

        if not self.cfg.enforce_json:
            try:
                return self._parse_json_content(content)
            except json.JSONDecodeError:
                return {"raw": content}

        return self._parse_json_content(content)

    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        s = content.strip()
        if s.startswith("```"):
            parts = s.split("```")
            if len(parts) >= 3:
                s = parts[1]
                if "\n" in s:
                    s = s.split("\n", 1)[1]
                s = s.strip()

        l = s.find("{")
        r = s.rfind("}")
        if l != -1 and r != -1 and r > l:
            return json.loads(s[l : r + 1])

        raise json.JSONDecodeError("Unable to parse JSON", content, 0)

    def chat(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        url = self.cfg.base_url.rstrip("/") + "/v1/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"

        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.enforce_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
        except requests.RequestException:
            return self._ollama_chat(messages)

        if resp.status_code in (404, 405):
            return self._ollama_chat(messages)

        if not resp.ok and self.cfg.enforce_json:
            payload.pop("response_format", None)
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.cfg.timeout_s)
            except requests.RequestException:
                return self._ollama_chat(messages)

        if resp.status_code in (404, 405):
            return self._ollama_chat(messages)

        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        if not self.cfg.enforce_json:
            try:
                return self._parse_json_content(content)
            except json.JSONDecodeError:
                return {"raw": content}

        return self._parse_json_content(content)
