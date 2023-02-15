import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, label_dict, stack_size, stride, transform=None, size=(200,200)):
        super().__init__()
        self.root_dir = root_dir
        self.label_dict = label_dict
        self.stack_size = stack_size
        self.stride = stride
        self.transform = transform
        self.size = size # New line
        self.folder_list = sorted([folder for folder in list(label_dict.keys()) if folder in os.listdir(root_dir)])
        self.stack_numbers = [int((len(os.listdir(os.path.join(root_dir, folder))) - stack_size) / stride) for folder in self.folder_list]

    def __len__(self):
        return sum(self.stack_numbers)

    def __getitem__(self, idx):
        i = 0

        while idx > self.stack_numbers[i]:
            idx -= self.stack_numbers[i]
            i += 1

        active_dir = os.path.join(self.root_dir, self.folder_list[i])
        label = self.label_dict[self.folder_list[i]]
        active_contents = sorted(os.listdir(active_dir))

        stack = []
        for i in range(idx*self.stride, (idx * self.stride) + self.stack_size):
            filepath = os.path.join(active_dir, active_contents[i])
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.size) # New line
            image = np.asarray(image)
            if self.transform:
                image = self.transform(image)
            image = np.expand_dims(image, axis=3)
            stack.append(image)
        stack = np.concatenate(stack, axis=3)

        return stack, label



def SpectrogramDataLoader(
    root_dir,
    label_dict,
    stack_size=25,
    stride=5,
    transform=None,
    batch_size=2,
    shuffle=True,
    num_workers=0,
):
    dataset = SpectrogramDataset(root_dir, label_dict, stack_size, stride, transform)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return dataloader
