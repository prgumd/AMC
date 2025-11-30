###############################################################################
#
# Buffer a signal and return samples using time ranges
#
# History:
# 01-03-24 - Levi Burner - Created file
#
###############################################################################

from bisect import bisect, bisect_left
from collections.abc import Iterable
from typing import Callable, Generic, Iterable, Optional, Tuple, TypeVar, overload

def linear_interp(t, sample_data):
    ts, xs = sample_data
    alpha = (t - ts[0]) / (ts[1] - ts[0])
    return alpha * (xs[1] - xs[0]) + xs[0]

def linear_interp_iterable(t, sample_data):
    ts, xs = sample_data
    alpha = (t - ts[0]) / (ts[1] - ts[0])
    return [alpha * (x1 - x0) + x0 for x0, x1 in zip(xs[0], xs[1])]

T = TypeVar("T")
class SampleBuffer(Generic[T]):
    def __init__(self, sample_close: Optional[Callable]=None, sample_interp: Optional[Callable]=None):
        self.t0 = None
        self.ts: list[float] = []
        self.samples: list[T] = []
        self.sample_close = sample_close
        self.sample_interp = sample_interp

    def __len__(self) -> int:
        return len(self.ts)

    def append(self, t: float, sample: T) -> None:
        if self.t0:
            t = t - self.t0

        if len(self.ts) > 0:
            assert self.ts[-1] < t, f"Current f: {t}, last t: {self.ts[-1]}"

        self.ts.append(t)
        self.samples.append(sample)

    @overload
    def get(self, t0: float) -> Tuple[float, T]: ...

    @overload 
    def get(self, t0: float, tf: float) -> Tuple[list[float], list[T]]: ...

    def get(self, t0: float, tf: Optional[float]=None) -> Tuple[float, T] | Tuple[list[float], list[T]]:
        if tf is not None:
            assert tf > t0

            left_idx  = bisect_left(self.ts, t0)
            right_idx = bisect(self.ts, tf, lo=left_idx)

            return self.ts[left_idx:right_idx], self.samples[left_idx:right_idx]
        else:
            left_idx = bisect_left(self.ts, t0)

            if left_idx == len(self.ts):
                return self.ts[-1], self.samples[-1]
            elif left_idx == 0:
                return self.ts[0], self.samples[0]
            elif self.sample_interp is not None:
                
                return t0, self.sample_interp(t0, self[left_idx-1:left_idx+1])
            else:
                return self.ts[left_idx-1], self.samples[left_idx-1]

    @overload
    def __getitem__(self, key: int) -> Tuple[float, T]:
        ...

    @overload
    def __getitem__(self, key: slice) -> Tuple[list[float], list[T]]:
        ...

    def __getitem__(self, key) -> Tuple[float, T] | Tuple[list[float], list[T]]:
        return self.ts[key], self.samples[key]

    def __setitem__(self, key: int, value: Tuple[float, T]) -> None:
        self.ts[key] = value[0]
        self.samples[key] = value[1]

    def __delitem__(self, key: int) -> None:
        del self.ts[key]
        del self.samples[key]

    def trim(self, t0: float) -> None:
        left_idx  = bisect_left(self.ts, t0)

        if self.sample_close:
            for i in range(left_idx):
                self.sample_close(self.ts[i], self.samples[i])

        self.ts = self.ts[left_idx:]
        self.samples = self.samples[left_idx:]

    def set_t0(self, t0: Optional[float]=None) -> None:
        if t0 is None and self.t0 is None:
            t0_inc = self.ts[-1]
            self.t0 = t0_inc
        elif t0 is None and self.t0 is not None:
            t0_inc = self.ts[-1]
            self.t0 += t0_inc
        elif t0 is not None and self.t0 is None:
            t0_inc = t0
            self.t0 = t0
        elif t0 is not None and self.t0 is not None:
            t0_inc = t0 - self.t0
            self.t0 = t0

        self.ts = [x - t0_inc for x in self.ts]


    def set_tf(self, tf: float) -> float:
        if len(self.ts) > 0:
            tf_inc = tf - self.ts[-1]
            self.ts = [x + tf_inc for x in self.ts]
            
        else:
            tf_inc = tf
        if self.t0 is None:
            self.t0 = -tf_inc
        else:
            self.t0 -= tf_inc
        return tf_inc

    def inc_t(self, t_inc: float) -> None:
            self.ts = [x + t_inc for x in self.ts]

    def samples_since(self, t: float) -> int:
        left_idx = bisect_left(self.ts, t)
        return (len(self.ts) - 1) - left_idx
