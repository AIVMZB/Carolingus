from torch.utils.data import Dataset


class WordDataset(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
    
    def __len__(self) -> int: ...

    def __getitem__(self, index): ...
