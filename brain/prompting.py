import json
from typing import Any, Dict, List, Sequence

from .openai_client import ChatMessage


SYSTEM_PROMPT = "你是一个碧蓝档案(Blue Archive)的高手玩家辅助AI。你的目标是通关活动关卡。请根据当前的屏幕信息（由视觉模块提供），输出下一步的 Action。只允许输出 JSON 对象，禁止输出额外文本。Action 格式: {\"action\":\"click\",\"target\":[x,y],\"reason\":\"...\"} 或 {\"action\":\"swipe\",\"from\":[x,y],\"to\":[x,y],\"duration_ms\":500,\"reason\":\"...\"} 或 {\"action\":\"wait\",\"duration_ms\":800,\"reason\":\"...\"} 或 {\"action\":\"back\",\"reason\":\"...\"}."


def build_screen_state(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    texts: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []

    for it in items:
        t = it.get("type")
        if t == "ocr":
            texts.append({"text": it.get("label"), "bbox": it.get("bbox")})
        else:
            objects.append({"label": it.get("label"), "bbox": it.get("bbox"), "query": it.get("query")})

    return {"texts": texts, "objects": objects}


def build_messages(screen_items: Sequence[Dict[str, Any]]) -> List[ChatMessage]:
    state = build_screen_state(screen_items)
    user = "Current Screen Context:\n" + json.dumps(state, ensure_ascii=False)
    user += "\nUser: 现在该做什么？"
    return [ChatMessage(role="system", content=SYSTEM_PROMPT), ChatMessage(role="user", content=user)]
