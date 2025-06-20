from dataclasses import dataclass

@dataclass(frozen=True, order=True)
class DiffKey:
    """Identifier for pairwise score difference"""
    pit: str
    model_a: str
    model_b: str

    def label(self) -> str:
        return f"{self.model_a}-{self.model_b}"

    def __str__(self) -> str:
        return f"{self.pit}:{self.model_a}-{self.model_b}"
