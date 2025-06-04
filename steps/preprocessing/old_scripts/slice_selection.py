import numpy as np


def get_centered_lesion_block(
    flair_data, mask_data, block_size=5, min_area_threshold=0
) -> tuple[list[np.ndarray], list[int]]:
    """
    Returns a block of slices centered on the slice with the largest lesion.

    Args:
        flair_data: 3D FLAIR array (H x W x Z)
        mask_data: 3D binary mask array (H x W x Z)
        block_size: odd number of slices to include (e.g., 3, 5, 7)
        min_area_threshold: minimum lesion pixel threshold

    Returns:
        block of FLAIR slices (list of 2D arrays)
        selected indices
    """
    assert block_size % 2 == 1, "block_size debe ser impar"
    assert flair_data.shape == mask_data.shape, "Dimensiones no coinciden"

    lesion_areas = [
        (i, np.sum(mask_data[:, :, i] > 0)) for i in range(mask_data.shape[2])
    ]
    lesion_areas = [t for t in lesion_areas if t[1] > min_area_threshold]

    if not lesion_areas:
        center_idx = mask_data.shape[2] // 2
    else:
        center_idx = max(lesion_areas, key=lambda x: x[1])[0]

    half = block_size // 2
    start = max(0, center_idx - half)
    end = min(mask_data.shape[2], center_idx + half + 1)

    # Padding if needed (when at borders)
    slices = []
    indices = []
    for i in range(start, end):
        slices.append(flair_data[:, :, i])
        indices.append(i)

    # If not enough slices, pad with zeros (optional)
    while len(slices) < block_size:
        slices.append(np.zeros_like(flair_data[:, :, 0]))
        indices.append(-1)

    return slices, indices


def get_all_lesion_slices(
    flair_data: np.ndarray,
    mask_data: np.ndarray,
    only_lesion_slices: bool = False
) -> tuple[list[np.ndarray], list[int]]:
    """
    Returns all slices of a volume or only those containing lesions.

    Args:
        flair_data: 3D FLAIR array (H x W x Z)
        mask_data: 3D binary mask array (H x W x Z)
        only_lesion_slices: if True, returns only slices with any lesion

    Returns:
        List of FLAIR slices (2D arrays)
        List of selected indices
    """
    assert flair_data.shape == mask_data.shape, "Dimensiones no coinciden"

    selected_slices = []
    indices = []

    for i in range(mask_data.shape[2]):
        has_lesion = np.any(mask_data[:, :, i] > 0)
        if only_lesion_slices and not has_lesion:
            continue
        selected_slices.append(flair_data[:, :, i])
        indices.append(i)

    return selected_slices, indices
