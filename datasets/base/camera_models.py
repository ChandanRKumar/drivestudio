from dataclasses import dataclass
import torch

@dataclass
class Camera:
    cam_id: int
    cam_name: str
    original_size: torch.Tensor
    load_size: torch.Tensor
    intrinsics: torch.Tensor
    distortions: torch.Tensor
    cam_to_worlds: torch.Tensor
    camera_type: str = "pinhole"

@dataclass
class FisheyeCamera(Camera):
    camera_type: str = "fisheye"
