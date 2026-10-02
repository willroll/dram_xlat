"""Load the probe's CSV dataset into numpy arrays."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Timing:
    addr_a: np.ndarray   # uint64[N]
    addr_b: np.ndarray   # uint64[N]
    latency: np.ndarray  # float[N]

    def __len__(self) -> int:
        return int(self.addr_a.size)

    @property
    def delta(self) -> np.ndarray:
        """a XOR b for every pair (uint64)."""
        return self.addr_a ^ self.addr_b


def load_csv(path: str) -> Timing:
    """Read 'addr_a,addr_b,latency' rows (header optional)."""
    a, b, lat = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#" or line.startswith("addr_a"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            try:
                a.append(int(parts[0]))
                b.append(int(parts[1]))
                lat.append(float(parts[2]))
            except ValueError:
                continue
    return Timing(np.array(a, dtype=np.uint64),
                  np.array(b, dtype=np.uint64),
                  np.array(lat, dtype=float))
