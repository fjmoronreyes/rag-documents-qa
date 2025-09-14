from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class Chunk:
    chunk_id: str
    content: str
    source: str
    page: int
    page_label: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
