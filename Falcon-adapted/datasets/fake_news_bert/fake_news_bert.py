import torch
from torch.utils.data import Dataset

class FakeNewsBertData(Dataset):
    def __init__(self, file_path, two_views=False):
        self.file_path = file_path
        self.two_views = two_views
        print(f"Loading fake news data from {self.file_path}")
        
        self.data = torch.load(self.file_path)
                
        self.ids = self.data["ids"]
        self.embeddings = self.data["embeddings"]
        self.labels = self.data["labels"]
        
        self.num_coarse = int(torch.max(self.labels).item()) + 1
        self.num_fine = 1  # placeholder for FALCON's unsupervised fine-grained discovery
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, item):
        x = self.embeddings[item]
        y = self.labels[item]
        dummy_fine = 0
        
        return {
            "index": item,
            "inputs": x if not self.two_views else [x, x],
            "fine_label": dummy_fine,
            "coarse_label": y,
        }
        
    def get_graph(self):
        M = torch.zeros((self.num_fine, self.num_coarse))
        for coarse in self.labels:
            M[0, coarse] = 1
        assert torch.sum(M) == self.num_coarse
        return M.numpy()