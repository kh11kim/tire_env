from pathlib import Path
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class OccPlacementPair:
    occ: np.ndarray
    placements: np.ndarray

    def save(self, path: str):
        data = {
            "occ": self.occ,
            "placements": self.placements
        }
        np.savez_compressed(path, **data)
    
    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        return cls(data["occ"], data["placements"])
    
if __name__ == "__main__":
    data_dir = Path("./data/train_data")
    data = list(data_dir.glob("*.npz"))
    for d in data:
        pair = OccPlacementPair.load(d)
        occ = pair.occ
        placements = pair.placements

        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        placements[placements!=1] = 0
        placements = placements.sum(axis=-1)
        ax[0].imshow(occ[0])
        ax[1].imshow(placements)