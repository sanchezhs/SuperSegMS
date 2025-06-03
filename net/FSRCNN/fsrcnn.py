import torch
import torch.nn as nn
import numpy as np

FSRCNN_PATH = {
    2: "net/FSRCNN/models/fsrcnn_x2.pth",
    3: "net/FSRCNN/models/fsrcnn_x3.pth",
    4: "net/FSRCNN/models/fsrcnn_x4.pth",
}

# https://github.com/yjn870/FSRCNN-pytorch
class FSRCNN:
    def __init__(self, scale_factor: int, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _FSRCNN(scale_factor).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Applies the FSRCNN model to a grayscale image and returns the upscaled result.
        """
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)
        img_tensor = img_tensor / (img_tensor.max() + 1e-8)

        with torch.no_grad():
            sr_tensor = self.model(img_tensor)

        sr_img = sr_tensor.squeeze(0).squeeze(0).cpu().numpy()
        sr_img = np.clip(sr_img, 0, 1) * 255
        return sr_img.astype(np.uint8)


class _FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(_FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2), nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend(
                [nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)]
            )
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1,
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
