"""Random patch masking for the FLAME MAE branch."""

import torch


class RandomPatchMasker:
    def __init__(self, patch_size: int = 4, mask_ratio: float = 0.8) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not 0 <= mask_ratio <= 1:
            raise ValueError("mask_ratio must be between 0 and 1")
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __call__(self, images: torch.Tensor):
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        _, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("patch_size must divide both image dimensions")
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        num_patches = grid_h * grid_w
        num_masked = round(num_patches * self.mask_ratio)
        patch_mask = torch.zeros(
            (images.size(0), num_patches), dtype=torch.bool, device=images.device
        )
        for row in patch_mask:
            if num_masked:
                row[torch.randperm(num_patches, device=images.device)[:num_masked]] = True
        mask = patch_mask.view(-1, grid_h, grid_w)
        mask = mask.repeat_interleave(self.patch_size, dim=1).repeat_interleave(
            self.patch_size, dim=2
        )
        return images * (~mask).unsqueeze(1).to(images.dtype), mask
