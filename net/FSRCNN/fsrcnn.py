import os
import torch
import torch.nn as nn
import numpy as np

FSRCNN_PATH = {
    2: "net/FSRCNN/models/fsrcnn_x2.pth",
    3: "net/FSRCNN/models/fsrcnn_x3.pth",
    4: "net/FSRCNN/models/fsrcnn_x4.pth",
}

class FSRCNN:
    def __init__(self, scale_factor: int, weights_path: str, normalize: str = "auto", return_uint8: bool = True):
        """
        Args:
            scale_factor: 2, 3 o 4
            weights_path: ruta al checkpoint entrenado con el mismo scale_factor
            normalize: "auto" | "255" | "none"
                - "auto": if uint8 -> /255; if float and max<=1.5 -> assume [0,1]; else /np.max
                - "255": always divide by 255
                - "none": no normalization (assumes input already matches training)
            return_uint8: if True returns uint8 in [0,255], else float32 in [0,1]
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale_factor = int(scale_factor)
        self.normalize = normalize
        self.return_uint8 = return_uint8

        self.model = _FSRCNN(self.scale_factor).to(self.device)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"FSRCNN weights not found: {weights_path}")
        # strict=True to catch mismatch between scale/arch and weights
        state = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        torch.backends.cudnn.benchmark = True

    @classmethod
    def from_scale(cls, scale_factor: int, normalize: str = "auto", return_uint8: bool = True) -> "FSRCNN":
        if scale_factor not in FSRCNN_PATH:
            raise ValueError(f"Unsupported scale_factor={scale_factor}. Allowed: {list(FSRCNN_PATH.keys())}")
        return cls(scale_factor, FSRCNN_PATH[scale_factor], normalize=normalize, return_uint8=return_uint8)

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Apply FSRCNN on a single-channel image.
        Input: 2D numpy array (H, W), dtype uint8 or float.
        Output: upscaled image with factor 'scale_factor'.
        """
        if img.ndim != 2:
            raise ValueError(f"Expected 2D gray image, got shape {img.shape}")
        arr = np.ascontiguousarray(img)

        # Normalization strategy
        arr = arr.astype(np.float32, copy=False)
        if self.normalize == "255":
            x = arr / 255.0
        elif self.normalize == "none":
            x = arr
        else:  # "auto"
            if img.dtype == np.uint8:
                x = arr / 255.0
            else:
                m = float(arr.max()) if arr.size > 0 else 1.0
                if m <= 1.5:  # looks already [0,1]
                    x = arr
                else:
                    x = arr / (m + 1e-8)

        # (1,1,H,W)
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            y_t = self.model(x_t)

        y = y_t.squeeze(0).squeeze(0).detach().cpu().numpy()
        y = np.clip(y, 0.0, 1.0)

        if self.return_uint8:
            return (y * 255.0 + 0.5).astype(np.uint8)
        else:
            return y.astype(np.float32)


class _FSRCNN(nn.Module):
    def __init__(self, scale_factor: int, num_channels: int = 1, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        if scale_factor not in (2, 3, 4):
            raise ValueError(f"scale_factor must be 2, 3 or 4, got {scale_factor}")
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d)
        )
        mid = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            mid += [nn.Conv2d(s, s, kernel_size=3, padding=1), nn.PReLU(s)]
        mid += [nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)]
        self.mid_part = nn.Sequential(*mid)
        self.last_part = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9,
            stride=scale_factor, padding=9 // 2, output_padding=scale_factor - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
