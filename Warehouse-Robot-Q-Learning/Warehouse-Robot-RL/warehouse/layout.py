"""Single source of truth for the 2D and 3D warehouse map."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Layout:
    width: int = 16
    height: int = 12
    start: tuple[int, int] = (1, 1)
    goal: tuple[int, int] = (14, 10)

    @property
    def shelves(self) -> set[tuple[int, int]]:
        return {(x, y) for x in (5, 6, 10, 11)
                for y in (*range(2, 5), *range(7, 10))}

    def blocked(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return x < 0 or y < 0 or x >= self.width or y >= self.height or cell in self.shelves

LAYOUT = Layout()
