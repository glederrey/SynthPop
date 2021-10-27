import numpy as np
import tensorflow as tf

from tensorpack import Callback
from tensorpack.utils import logger

# See paper "Persistent Homology in Data Science" from S. Huber
# link: https://link.springer.com/chapter/10.1007%2F978-3-658-32182-6_13
# link-code: https://www.sthu.org/blog/13-perstopology-peakdetection/index.html

class Peak:
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return float("inf") if self.died is None else seq[self.born] - seq[self.died]


def get_persistent_homology(seq):
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key=lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = (idx > 0 and idxtopeak[idx - 1] is not None)
        rgtdone = (idx < len(seq) - 1 and idxtopeak[idx + 1] is not None)
        il = idxtopeak[idx - 1] if lftdone else None
        ir = idxtopeak[idx + 1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks) - 1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    # This is optional convenience
    pers = np.array([p.get_persistence(seq) for p in peaks])

    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)


class ClipCallback(Callback):
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            n = v.op.name
            if not n.startswith('discrim/'):
                continue
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()
