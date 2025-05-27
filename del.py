import pandas as pd
import matplotlib.pyplot as plt

# # Cargar los datos desde el CSV
file_paths = [
    "results/yolo/yolo_all/train/results.csv",
    "results/yolo/yolo_all_100/train/results.csv",
    "results/yolo/yolo_all_100_x2/train/results.csv",
    "results/yolo/yolo_top5_slice_x0/train/results.csv",
    "results/yolo/yolo_top5_slice_x2/train/results.csv",
]

df_map = {
    k: pd.read_csv(v) for k, v in zip(
        [
            "yolo_all", 
            "yolo_all_100", 
            "yolo_all_100_x2", 
            "yolo_top5_slice_x0", 
            "yolo_top5_slice_x2"
        ],
        file_paths
    )
}

# df = df_map["yolo_all"]
# df = df[df["val/seg_loss"].notna()]
# plt.figure(figsize=(10, 6))
# plt.plot(df["epoch"], df["train/seg_loss"], label="Train Loss")
# plt.plot(df["epoch"], df["val/seg_loss"], label="Val Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title(f"Train and Validation Loss Curve - yolo_all_100_x2")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(f"results/yolo/yolo_100_x2_loss_per_epoch.png")
# plt.close()  # Close the figure to free memory


# # Crear y guardar una figura para cada archivo
for key, df in df_map.items():
    df = df[df["val/seg_loss"].notna()]
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train/seg_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val/seg_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train and Validation Loss Curve - {key}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"results/yolo/{key}_loss_per_epoch.png")
    plt.close()  # Close the figure to free memory
