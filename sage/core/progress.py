"""Printed progress of long commands: steps ahead, step times, bars.

Bars write to stdout, refreshed once a minute when it is not a terminal
(a job log), so a log keeps steps and bars in order.
"""

import sys
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, Optional, Sequence

from tqdm import tqdm

LOG_REFRESH_S = 60.0


def print_steps(steps: Sequence[str], after: Optional[str] = None) -> None:
    """
    Print the numbered steps ahead, a closing line, then a blank line.

    Parameters
    ----------
    steps : sequence of str
        Steps, each a sentence.
    after : str, optional
        Line printed after the steps.
    """
    for index, text in enumerate(steps, 1):
        print(f"  {index}. {text}")
    if after:
        print(after)
    print(flush=True)


@contextmanager
def step(index: int, steps: Sequence[str]) -> Iterator[None]:
    """
    Print step ``index`` of ``steps``, run the block, then its time.

    Parameters
    ----------
    index : int
        Step number, from 1.
    steps : sequence of str
        Every step, each a sentence.
    """
    print(f"[{index}/{len(steps)}] {steps[index - 1]}", flush=True)
    started = time.monotonic()
    yield
    print(f"  done in {time.monotonic() - started:.1f} s", flush=True)


def bar(
    iterable: Optional[Iterable] = None,
    total: Optional[int] = None,
    initial: int = 0,
    unit: str = "it",
) -> tqdm:
    """
    A progress bar on stdout.

    Parameters
    ----------
    iterable : iterable, optional
        Items to iterate over.
    total : int, optional
        Items in all; taken from ``iterable`` when it has a length.
    initial : int
        Items done before.
    unit : str
        Name of one item.

    Returns
    -------
    tqdm.tqdm
        The bar; iterate over it or call ``update``.
    """
    interval = 0.1 if sys.stdout.isatty() else LOG_REFRESH_S
    return tqdm(
        iterable,
        total=total,
        initial=initial,
        unit=unit,
        file=sys.stdout,
        mininterval=interval,
        dynamic_ncols=True,
    )
