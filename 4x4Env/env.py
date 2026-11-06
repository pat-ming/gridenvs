import torch

class graph: # 4x4 grid
    def __init__(self, values: list[list[float | int]]):
        self.grid = torch.tensor(values)

    def __repr__(self):
        return repr(self.grid)

    def get(self, r: int, c: int):
        return self[r, c]


example = [
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]
]

print(graph(example))