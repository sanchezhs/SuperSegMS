from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from schemas.pipeline_schemas import SegmentationMetrics


class MetricsViz:
    """Metrics for evaluating segmentation performance."""
    def __init__(self,
            metrics: SegmentationMetrics,
        ):
        self.iou = metrics.iou
        self.dice_score = metrics.dice_score
        self.precision = metrics.precision
        self.recall = metrics.recall
        self.specificity = metrics.specificity
        self.inference_time = metrics.inference_time

    def __str__(self):
        return (
            f"IoU: {self.iou:.4f}, Dice: {self.dice_score:.4f}, "
            f"Precision: {self.precision:.4f}, Recall: {self.recall:.4f}, "
            f"Specificity: {self.specificity}, "
            f"Inference Time: {self.inference_time if self.inference_time else 'N/A'}s"
        )

    def __repr__(self):
        return (
            f"SegmentationMetrics(iou={self.iou:.4f}, "
            f"dice_score={self.dice_score:.4f}, "
            f"precision={self.precision:.4f}, "
            f"recall={self.recall:.4f}, "
            f"specificity={self.specificity}, "
            f"inference_time={self.inference_time if self.inference_time else 'N/A'})"
        )

    def write_to_file(self, path: Path) -> None:
        """Write metrics to a file.
        Args:
            path: Path to the file where metrics will be saved.
        """
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))

    def plot_metrics_bar(self, 
                        save_path: Optional[Path] = None, 
                        title: str = "Segmentation Metrics",
                        figsize: tuple = (10, 6),
                        color_palette: str = "viridis") -> None:
        """
        Create a bar plot of segmentation metrics.
        
        Args:
            save_path: Path to save the plot. If None, plot is displayed.
            title: Title for the plot.
            figsize: Figure size as (width, height).
            color_palette: Color palette for the bars.
        """
        # Prepare data excluding inference_time for main metrics
        metrics_data = {
            'IoU': self.iou,
            'Dice Score': self.dice_score,
            'Precision': self.precision,
            'Recall': self.recall,
            'Specificity': self.specificity
        }
        
        # Create the plot
        plt.figure(figsize=figsize)
        colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(metrics_data)))
        
        bars = plt.bar(metrics_data.keys(), metrics_data.values(), color=colors)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_data.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add inference time as text if available
        if self.inference_time is not None:
            plt.figtext(0.02, 0.02, f'Inference Time: {self.inference_time:.4f}s', 
                       fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics bar plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_metrics_radar(self, 
                          save_path: Optional[Path] = None,
                          title: str = "Segmentation Metrics Radar Chart",
                          figsize: tuple = (8, 8)) -> None:
        """
        Create a radar chart of segmentation metrics.
        
        Args:
            save_path: Path to save the plot. If None, plot is displayed.
            title: Title for the plot.
            figsize: Figure size as (width, height).
        """
        # Prepare data excluding inference_time
        metrics_names = ['IoU', 'Dice Score', 'Precision', 'Recall', 'Specificity']
        metrics_values = [self.iou, self.dice_score, self.precision, self.recall, self.specificity]
        
        # Number of metrics
        N = len(metrics_names)
        
        # Compute angles for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add first value at the end to complete the circle
        metrics_values += metrics_values[:1]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Plot the radar chart
        ax.plot(angles, metrics_values, 'o-', linewidth=2, label='Metrics', color='#1f77b4')
        ax.fill(angles, metrics_values, alpha=0.25, color='#1f77b4')
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_names, fontsize=11)
        
        # Set y-axis limits and labels
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)
        
        # Add values on the plot
        for angle, value, name in zip(angles[:-1], metrics_values[:-1], metrics_names):
            ax.text(angle, value + 0.05, f'{value:.3f}', 
                   horizontalalignment='center', fontweight='bold', fontsize=9)
        
        plt.title(title, size=14, fontweight='bold', pad=20)
        
        # Add inference time as text if available
        if self.inference_time is not None:
            plt.figtext(0.02, 0.02, f'Inference Time: {self.inference_time:.4f}s', 
                       fontsize=10, style='italic')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics radar chart saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def plot_confusion_matrix_style(self,
                                   save_path: Optional[Path] = None,
                                   title: str = "Segmentation Performance Matrix",
                                   figsize: tuple = (8, 6)) -> None:
        """
        Create a confusion matrix style visualization showing metrics.
        
        Args:
            save_path: Path to save the plot. If None, plot is displayed.
            title: Title for the plot.
            figsize: Figure size as (width, height).
        """
        # Create a 2x3 matrix for metrics display
        metrics_matrix = np.array([
            [self.precision, self.recall, self.dice_score],
            [self.iou, self.specificity, 
             self.inference_time if self.inference_time else 0]
        ])
        
        labels = np.array([
            ['Precision', 'Recall', 'Dice Score'],
            ['IoU', 'Specificity', 'Inf. Time (s)']
        ])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        _ = ax.imshow(metrics_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(['Metric 1', 'Metric 2', 'Metric 3'])
        ax.set_yticklabels(['Row 1', 'Row 2'])
        
        # Add text annotations
        for i in range(2):
            for j in range(3):
                if labels[i, j] == 'Inf. Time (s)' and self.inference_time:
                    text = f'{labels[i, j]}\n{metrics_matrix[i, j]:.4f}'
                elif labels[i, j] == 'Specificity' and self.specificity is None:
                    text = f'{labels[i, j]}\nN/A'
                elif labels[i, j] == 'Inf. Time (s)' and self.inference_time is None:
                    text = f'{labels[i, j]}\nN/A'
                else:
                    text = f'{labels[i, j]}\n{metrics_matrix[i, j]:.3f}'
                
                ax.text(j, i, text, ha="center", va="center", 
                       color="black", fontweight='bold', fontsize=10)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics matrix plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    @classmethod
    def plot_comparison(cls,
                       metrics_list: List['SegmentationMetrics'],
                       labels: List[str],
                       save_path: Optional[Path] = None,
                       title: str = "Metrics Comparison",
                       plot_type: str = "bar",
                       figsize: tuple = (12, 8)) -> None:
        """
        Compare multiple SegmentationMetrics instances.
        
        Args:
            metrics_list: List of SegmentationMetrics instances to compare.
            labels: Labels for each metrics instance.
            save_path: Path to save the plot. If None, plot is displayed.
            title: Title for the plot.
            plot_type: Type of plot ('bar', 'line', 'radar').
            figsize: Figure size as (width, height).
        """
        if len(metrics_list) != len(labels):
            raise ValueError("Number of metrics and labels must match")
        
        # Prepare data
        metric_names = ['IoU', 'Dice Score', 'Precision', 'Recall', 'Specificity']
        data = {name: [] for name in metric_names}
        
        # Collect data
        for metrics in metrics_list:
            data['IoU'].append(metrics.iou)
            data['Dice Score'].append(metrics.dice_score)
            data['Precision'].append(metrics.precision)
            data['Recall'].append(metrics.recall)
            data['Specificity'].append(metrics.specificity)
        
        if plot_type == "bar":
            cls._plot_comparison_bar(data, labels, metric_names, title, figsize, save_path)
        elif plot_type == "line":
            cls._plot_comparison_line(data, labels, metric_names, title, figsize, save_path)
        elif plot_type == "radar":
            cls._plot_comparison_radar(metrics_list, labels, title, figsize, save_path)
        else:
            raise ValueError("plot_type must be 'bar', 'line', or 'radar'")

    @staticmethod
    def _plot_comparison_bar(data: Dict, labels: List[str], metric_names: List[str],
                            title: str, figsize: tuple, save_path: Optional[Path]) -> None:
        """Create bar plot comparison."""
        x = np.arange(len(metric_names))
        width = 0.8 / len(labels)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, label in enumerate(labels):
            values = [data[metric][i] for metric in metric_names]
            bars = ax.bar(x + i * width, values, width, label=label, alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison bar plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    @staticmethod
    def _plot_comparison_line(data: Dict, labels: List[str], metric_names: List[str],
                             title: str, figsize: tuple, save_path: Optional[Path]) -> None:
        """Create line plot comparison."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, label in enumerate(labels):
            values = [data[metric][i] for metric in metric_names]
            ax.plot(metric_names, values, marker='o', linewidth=2, 
                   markersize=8, label=label)
            
            # Add value labels
            for j, value in enumerate(values):
                ax.text(j, value + 0.02, f'{value:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison line plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    @staticmethod
    def _plot_comparison_radar(metrics_list: List['SegmentationMetrics'], 
                              labels: List[str], title: str, figsize: tuple, 
                              save_path: Optional[Path]) -> None:
        """Create radar plot comparison."""
        # Prepare data
        metric_names = ['IoU', 'Dice Score', 'Precision', 'Recall', 'Specificity']
        
        N = len(metric_names)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(metrics_list)))
        
        for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
            values = [metrics.iou, metrics.dice_score, metrics.precision, metrics.recall, metrics.specificity]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.title(title, size=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison radar plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

    def save_all_plots(self, 
                      base_path: Path, 
                      prefix: str = "metrics",
                      title_prefix: str = "Model") -> None:
        """
        Save all available plot types for the metrics.
        
        Args:
            base_path: Base directory to save plots.
            prefix: Prefix for filenames.
            title_prefix: Prefix for plot titles.
        """
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Save bar plot
        self.plot_metrics_bar(
            save_path=base_path / f"{prefix}_bar.png",
            title=f"{title_prefix} - Segmentation Metrics"
        )
        
        # Save radar chart
        self.plot_metrics_radar(
            save_path=base_path / f"{prefix}_radar.png",
            title=f"{title_prefix} - Metrics Radar Chart"
        )
        
        # Save matrix plot
        self.plot_confusion_matrix_style(
            save_path=base_path / f"{prefix}_matrix.png",
            title=f"{title_prefix} - Performance Matrix"
        )
        
        logger.info(f"All plots saved in: {base_path}")
