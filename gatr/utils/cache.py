# gatr/utils/cache.py
import functools
from typing import Any, Callable

class gatr_cache(dict):
    """Serves as a torch.compile-compatible replacement for @functools.cache()."""
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def __missing__(self, item: Any) -> Any:
        result = self.fn(*item)
        self[item] = result
        return result

    def __call__(self, *args: Any) -> Any:
        return self[args]
