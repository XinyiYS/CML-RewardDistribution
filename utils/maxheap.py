import heapq


class MaxHeapObj(object):
    def __init__(self, val): self.val = val
    def __lt__(self, other): return self.val > other.val
    def __eq__(self, other): return self.val == other.val
    def __str__(self): return str(self.val)


class MinHeap(object):
    def __init__(self): self.h = []
    def heappush(self, x): heapq.heappush(self.h, x)
    def heappop(self): return heapq.heappop(self.h)
    def __getitem__(self, i): return self.h[i]
    def __len__(self): return len(self.h)


class MaxHeap(MinHeap):
    def heappush(self, x): heapq.heappush(self.h, MaxHeapObj(x))
    def heappop(self): return heapq.heappop(self.h).val
    def __getitem__(self, i): return self.h[i].val
