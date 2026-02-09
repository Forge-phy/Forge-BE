"""
Forge 대화 관리 - 멀티턴 대화 및 맥락 유지
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
from enum import Enum


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """대화 메시지"""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ConversationContext:
    """대화 컨텍스트 - 현재 상태 저장"""
    current_config: Optional[dict] = None
    last_config: Optional[dict] = None
    intent_history: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)  # 추출된 엔티티
    modification_count: int = 0

    def update_config(self, new_config: dict):
        self.last_config = self.current_config
        self.current_config = new_config
        self.modification_count += 1

    def add_entity(self, key: str, value: Any):
        self.entities[key] = value

    def get_summary(self) -> str:
        """현재 상태 요약"""
        if not self.current_config:
            return "설정 없음"

        env = self.current_config.get("environment", {})
        robots = self.current_config.get("robots", [])

        summary_parts = []
        if env:
            summary_parts.append(f"{env.get('type', 'unknown')} {env.get('width', 0)}x{env.get('length', 0)}m")

        for robot in robots:
            summary_parts.append(f"{robot.get('type', 'robot')} {robot.get('count', 0)}대")

        return ", ".join(summary_parts) if summary_parts else "설정 없음"


class ConversationManager:
    """대화 세션 관리자"""

    def __init__(self, session_id: Optional[str] = None, max_history: int = 20):
        self.session_id = session_id or self._generate_session_id()
        self.messages: List[Message] = []
        self.context = ConversationContext()
        self.max_history = max_history
        self.created_at = datetime.now()

    def _generate_session_id(self) -> str:
        return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:12]

    def add_message(self, role: MessageRole, content: str, metadata: Dict = None) -> Message:
        """메시지 추가"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)

        # 최대 히스토리 유지
        if len(self.messages) > self.max_history:
            # 시스템 메시지는 유지
            system_msgs = [m for m in self.messages if m.role == MessageRole.SYSTEM]
            other_msgs = [m for m in self.messages if m.role != MessageRole.SYSTEM]
            self.messages = system_msgs + other_msgs[-(self.max_history - len(system_msgs)):]

        return msg

    def add_user_message(self, content: str, metadata: Dict = None) -> Message:
        return self.add_message(MessageRole.USER, content, metadata)

    def add_assistant_message(self, content: str, metadata: Dict = None) -> Message:
        return self.add_message(MessageRole.ASSISTANT, content, metadata)

    def add_system_message(self, content: str) -> Message:
        return self.add_message(MessageRole.SYSTEM, content)

    def get_history_for_prompt(self, max_tokens: int = 4000) -> str:
        """프롬프트용 대화 히스토리 생성"""
        history_parts = []
        total_length = 0

        # 최근 메시지부터 역순으로
        for msg in reversed(self.messages):
            if msg.role == MessageRole.SYSTEM:
                continue

            msg_text = f"[{msg.role.value}]: {msg.content}"
            msg_length = len(msg_text)

            if total_length + msg_length > max_tokens * 4:  # 대략적인 토큰 추정
                break

            history_parts.insert(0, msg_text)
            total_length += msg_length

        return "\n".join(history_parts)

    def get_context_summary(self) -> str:
        """현재 컨텍스트 요약"""
        parts = []

        if self.context.current_config:
            parts.append(f"현재 설정: {self.context.get_summary()}")

        if self.context.entities:
            entities_str = ", ".join(f"{k}={v}" for k, v in self.context.entities.items())
            parts.append(f"인식된 정보: {entities_str}")

        if self.context.modification_count > 0:
            parts.append(f"수정 횟수: {self.context.modification_count}회")

        return "\n".join(parts) if parts else "새 대화"

    def update_config(self, config: dict):
        """설정 업데이트"""
        self.context.update_config(config)

    def get_current_config(self) -> Optional[dict]:
        return self.context.current_config

    def get_last_config(self) -> Optional[dict]:
        return self.context.last_config

    def detect_intent(self, user_input: str) -> str:
        """사용자 의도 감지"""
        user_lower = user_input.lower()

        # 수정 의도
        modify_keywords = ["수정", "변경", "바꿔", "올려", "내려", "추가", "제거", "삭제", "더", "빼"]
        if any(kw in user_lower for kw in modify_keywords):
            return "modify"

        # 분석 요청
        analyze_keywords = ["분석", "결과", "어떻게", "왜", "문제", "원인"]
        if any(kw in user_lower for kw in analyze_keywords):
            return "analyze"

        # 정보 요청
        info_keywords = ["알려", "설명", "뭐야", "무엇"]
        if any(kw in user_lower for kw in info_keywords):
            return "info"

        # 새 환경 생성
        create_keywords = ["만들어", "생성", "설정", "창고", "공장", "환경"]
        if any(kw in user_lower for kw in create_keywords):
            return "create"

        return "unknown"

    def to_dict(self) -> dict:
        """직렬화"""
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "context": {
                "current_config": self.context.current_config,
                "last_config": self.context.last_config,
                "intent_history": self.context.intent_history,
                "entities": self.context.entities,
                "modification_count": self.context.modification_count
            },
            "created_at": self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationManager":
        """역직렬화"""
        manager = cls(session_id=data["session_id"])

        for msg_data in data["messages"]:
            manager.messages.append(Message(
                role=MessageRole(msg_data["role"]),
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                metadata=msg_data.get("metadata", {})
            ))

        ctx_data = data.get("context", {})
        manager.context.current_config = ctx_data.get("current_config")
        manager.context.last_config = ctx_data.get("last_config")
        manager.context.intent_history = ctx_data.get("intent_history", [])
        manager.context.entities = ctx_data.get("entities", {})
        manager.context.modification_count = ctx_data.get("modification_count", 0)

        return manager


# 세션 저장소
class SessionStore:
    """세션 저장소"""

    def __init__(self):
        self._sessions: Dict[str, ConversationManager] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> ConversationManager:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        session = ConversationManager(session_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[ConversationManager]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())


# 글로벌 세션 스토어
session_store = SessionStore()
