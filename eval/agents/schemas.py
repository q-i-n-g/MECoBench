from dataclasses import dataclass, field


class ActionDecodeError(Exception):
    def __init__(self, message: str, memory: str | None = None):
        super().__init__(message)
        self.memory = memory


@dataclass
class CommEvent:
    sender: int
    receiver: str | int | list[int] | tuple[int, ...] | set[int]
    content: str
    event_type: str = "message"

    def as_dialogue(self, sender_name: str, content: str) -> dict[str, str]:
        return {"role": sender_name, "content": content}

    def is_visible_to(self, agent_id: int) -> bool:
        if self.sender == agent_id:
            return True
        if self.receiver == "all":
            return True
        if isinstance(self.receiver, int):
            return self.receiver == agent_id
        if isinstance(self.receiver, (list, tuple, set)):
            return agent_id in self.receiver
        return False


@dataclass
class CommPhaseResult:
    transcript: list[CommEvent] = field(default_factory=list)
    round_transcripts: list[list[CommEvent]] = field(default_factory=list)
    inbox_per_agent: dict[int, list[dict[str, str]]] = field(default_factory=dict)
    action_dialogue_per_agent: dict[int, list[dict[str, str]]] = field(default_factory=dict)
    assignments: dict[int, str | None] = field(default_factory=dict)
    stop_reason: str | None = None


@dataclass
class ActionMemoryDecision:
    action: str
    memory: str
    raw_output: str
    action_thinking: str = ""
    memory_thinking: str = ""
