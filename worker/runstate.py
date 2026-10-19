from dataclasses import dataclass
from typing import Optional

@dataclass
class RunState:
    current_step: int = 0
    stop_reason: Optional[str] = None
