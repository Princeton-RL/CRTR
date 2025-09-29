import os
import numpy as np
import torch
import copy
import cloudpickle
from pathlib import Path

def get_mapping():    
    return  {'o':0, 'y':1, 'r':2, 'b':3, 'g':4, 'w':5}

GOAL = 'yyyyyyyyybbbbbbbbbrrrrrrrrrgggggggggooooooooowwwwwwwww'

NUM_GOAL = []
for char in GOAL:
    NUM_GOAL.append(get_mapping()[char])
    
NUM_GOAL = torch.tensor(NUM_GOAL)

reverse_actions = torch.tensor([3, 6, 7, 0, 11, 9, 1, 2, 10, 5, 8, 4], dtype=int)

perms = torch.from_numpy(np.array([[[15, 16, 17, 42, 43, 44, 33, 34, 35, 24, 25, 26, 45, 46, 47, 48, 49, 50, 51, 52, 53], [42, 43, 44, 33, 34, 35, 24, 25, 26, 15, 16, 17, 51, 48, 45, 52, 49, 46, 53, 50, 47]],
    [[9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 0, 1, 2, 3, 4, 5, 6, 7, 8] , [18, 19, 20, 27, 28, 29, 36, 37, 38, 9, 10, 11, 6, 3, 0, 7, 4, 1, 8, 5, 2]],
    [[0, 3, 6, 38, 41, 44, 45, 48, 51, 18, 21, 24, 9, 10, 11, 12, 13, 14, 15, 16, 17] , [44, 41, 38, 51, 48, 45, 18, 21, 24, 0, 3, 6, 15, 12, 9, 16, 13, 10, 17, 14, 11]],
    [[15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53] , [24, 25, 26, 33, 34, 35, 42, 43, 44, 15, 16, 17, 47, 50, 53, 46, 49, 52, 45, 48, 51]],
    [[0, 1, 2, 29, 32, 35, 51, 52, 53, 9, 12, 15, 36, 37, 38, 39, 40, 41, 42, 43, 44] , [29, 32, 35, 53, 52, 51, 9, 12, 15, 2, 1, 0, 42, 39, 36, 43, 40, 37, 44, 41, 38]],
    [[2, 5, 8, 20, 23, 26, 47, 50, 53, 36, 39, 42, 27, 28, 29, 30, 31, 32, 33, 34, 35] , [20, 23, 26, 47, 50, 53, 42, 39, 36, 8, 5, 2, 33, 30, 27, 34, 31, 28, 35, 32, 29]],
    [[9, 10, 11, 36, 37, 38, 27, 28, 29, 18, 19, 20] + list(range(9)) , [36, 37, 38, 27, 28, 29, 18, 19, 20, 9, 10, 11, 2, 5, 8, 1, 4, 7, 0, 3, 6]],
    [[0, 3, 6, 18, 21, 24, 45, 48, 51, 38, 41, 44] + list(range(9, 18)), [18, 21, 24, 45, 48, 51, 44, 41, 38, 6, 3, 0, 11, 14, 17, 10, 13, 16, 9, 12, 15]],
    [[6, 7, 8, 27, 30, 33, 45, 46, 47, 11, 14, 17] + list(range(18, 27)) , [27, 30, 33, 47, 46, 45, 11, 14, 17, 8, 7, 6, 20, 23, 26, 19, 22, 25, 18, 21, 24]],
    [[2, 5, 8, 36, 39, 42, 47, 50, 53, 20, 23, 26] + list(range(27, 36)), [42, 39, 36, 53, 50, 47, 20, 23, 26, 2, 5, 8, 29, 32, 35, 28, 31, 34, 27, 30, 33]],
    [[6, 7, 8, 11, 14, 17, 45, 46, 47, 27, 30, 33] + list(range(18, 27)) , [17, 14, 11, 45, 46, 47, 33, 30, 27, 6, 7, 8, 24, 21, 18, 25, 22, 19, 26, 23, 20]],
    [[0, 1, 2, 9, 12, 15, 51, 52, 53, 29, 32, 35] + list(range(36, 45)), [15, 12, 9, 51, 52, 53, 35, 32, 29, 0, 1, 2, 38, 41, 44, 37, 40, 43, 36, 39, 42]],
    [list(range(21)), list(range(21))]])) # last is an identity transform

def get_dataset_stats(directories):
    num_traj = 0
    file_paths = []
    assert all(map(lambda x: os.path.isdir(x), directories)) or not any(map(lambda x: os.path.isdir(x), directories))
    for directory in directories:
        if os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.pkl'):
                    num_traj += 1
                    file_paths.append((directory, filename))
        
        else:
            arr = joblib.load(directory)
            num_traj += len(arr)

    file_paths = sorted(file_paths)
    return num_traj, file_paths

class GenDataset():
    def __init__(self, length = 20):
        
        self.length = length
        
        self.NUM_GOAL = copy.deepcopy(NUM_GOAL)
        
        self.perms = copy.deepcopy(perms)
        
        self.reverse_actions = copy.deepcopy(reverse_actions)
        
        
    def _get_trajs(self, n_traj):
        new_data = torch.zeros((n_traj, self.length, 54))

        current_data = self.NUM_GOAL.unsqueeze(0).repeat(n_traj, 1)
        all_actions = torch.full((n_traj, self.length), -1, dtype=int)
        
        for i in range(self.length):
            new_data[:, self.length - i - 1] = current_data
            actions = torch.randint(high=len(perms) - 1, size=(n_traj,))
            all_actions[:, self.length-i-1] = actions


            to_apply = self.perms[actions]
            permuted_values = torch.gather(current_data, 1, to_apply[:, 1])
            current_data.scatter_(1, to_apply[:, 0], permuted_values)

        return new_data, all_actions
            

def main():
    import sys
    import os
    assert len(sys.argv) == 3, f"Usage: python {sys.argv[0]} length save_path"
    length=int(sys.argv[1])
    save_path = Path(sys.argv[2])  
    save_path.mkdir(parents=True, exist_ok=True) 
    
    os.makedirs(save_path, exist_ok=True)
    n_trajs = 5000000
    bs = 2048 * 128
    
    dataset = GenDataset(length=length)
    
    done = 0
    i = -1
    print("n iter", int(n_trajs/bs))
    while True:
        i += 1
        trajs, actions = dataset._get_trajs(bs)
        with open(save_path / f"part_{i}.pkl", 'wb') as f:
            cloudpickle.dump(trajs.to(int), f)
        
        done += len(trajs)
        del trajs
        
        if done > n_trajs:
            break


        

if __name__ == "__main__":
    main()

