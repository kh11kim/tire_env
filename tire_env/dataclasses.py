import numpy as np
from typing import Optional
from dataclasses import dataclass
import yaml
from pathlib import Path

# coner_rad = np.array(
#     [0.0384, 0.0429, 0.0329, 0.0412, 0.0430, 0.0499, 0.0309, 0.0529, 0.0217]
# ) * 2
nparray_to_list = lambda x: x.tolist() if isinstance(x, np.ndarray) else x

@dataclass
class TireInfo:
    name: str
    tire_type: int
    inner_radius: float
    outer_radius: float
    width: float
    corner_radius: Optional[float] = None
    path: Optional[str] = None

    def __post_init__(self):
        self.name = str(self.name)
        self.tire_type = int(self.tire_type)
        self.inner_radius = float(self.inner_radius)
        self.outer_radius = float(self.outer_radius)
        self.width = float(self.width)
        self.corner_radius = float(self.corner_radius)
        self.path = str(self.path)

    def save(self, path: str):
        attrs = [
            "name",
            "tire_type", 
            "inner_radius", 
            "outer_radius", 
            "width", 
            "corner_radius"
        ]
        with open(path, "w") as f:
            data = {attr: getattr(self, attr) for attr in attrs}
            yaml.dump(data, f)
    
    @classmethod
    def load(self, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        data['path'] = str(path)
        return TireInfo(**data)
    
    def get_mesh_path(self, col=False):
        if col:
            mesh_path = Path(self.path).parent / "col.obj"
        else:
            mesh_path = Path(self.path).parent / "visual.obj"
        return mesh_path
    
    @property
    def mesh_path(self):
        return self.get_mesh_path()
    
    @property
    def col_mesh_path(self):
        return self.get_mesh_path(col=True)

@dataclass
class OccPlacementPairV2:
    occ: np.ndarray
    placeable: np.ndarray # x, theta
    stable: np.ndarray # x, theta
    unstable: np.ndarray # x, theta

    def save(self, path: str):
        data = {
            "occ": self.occ.astype(bool),
            "placeable": self.placeable.astype(bool),
            "stable": self.stable.astype(bool),
            "unstable": self.unstable.astype(bool)
        }
        np.savez_compressed(path, **data)
    
    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        return cls(
            occ=data["occ"], 
            placeable=data["placeable"],
            stable=data["stable"],
            unstable=data["unstable"]
        )