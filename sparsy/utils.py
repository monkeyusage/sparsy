from typing import Iterable, Iterator, TypeVar
from itertools import islice

T = TypeVar("T")
def chunked_iterable(iterable: Iterable[T], size:int) -> Iterator[tuple[T, ...]]:
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk