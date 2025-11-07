"""
Callbacks for training and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import tensorboard.plugins.projector as projector
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, classification_report
)
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import psutil
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# Disable NVML-based GPU monitoring by default
pynvml = None
HAS_NVML = False

# Only try to import pynvml if we really need it
if torch.cuda.is_available():
    try:
        import pynvml
        pynvml.nvmlInit()
        HAS_NVML = True
    except (ImportError, pynvml.NVMLError):
        # Silently disable NVML - not critical for training
        pynvml = None
        HAS_NVML = False


class BaseCallback:
    """
    Base class for callbacks.
    """

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class TensorBoardLogger(BaseCallback):
    """
    Optimized TensorBoard logger with reduced overhead and better GPU utilization.
    Features:
    - Efficient model visualization and monitoring
    - Reduced logging overhead with configurable intervals
    - Memory-efficient activation logging
    - Hardware monitoring (CPU/GPU utilization, memory usage)
    - Gradient/weight histogram logging
    """
    def __init__(self, 
                 log_dir: str = "runs", 
                 num_classes: int = 10,
                 model: Optional[torch.nn.Module] = None,
                 input_shape: Tuple[int, ...] = (1, 28, 28),
                 log_interval: int = 10,
                 hparams: Optional[Dict[str, Any]] = None,
                 histogram_freq: int = 0,
                 log_gradients: bool = False,
                 log_weights: bool = True,
                 profile_batch: Optional[int] = None):
        """
        Initialize the optimized TensorBoard logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            num_classes: Number of classes in the classification task
            model: PyTorch model for visualization
            input_shape: Shape of input tensor (C, H, W)
            log_interval: Log metrics every N batches
            hparams: Hyperparameters dictionary for logging
            histogram_freq: Frequency (in epochs) at which to log weight histograms (0 to disable)
            log_gradients: Whether to log gradient histograms
            log_weights: Whether to log weight histograms
            profile_batch: Batch to profile (None to disable)
        """
        self.writer = SummaryWriter(log_dir, flush_secs=30)  # Flush every 30 seconds
        self.num_classes = num_classes
        self.class_names = [str(i) for i in range(num_classes)]
        self.log_interval = max(1, log_interval)  # Ensure at least 1
        self.hparams = hparams or {}
        self.histogram_freq = max(0, histogram_freq)
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.profile_batch = profile_batch
        
        # Model and input shape for visualization
        self.model = model
        self.input_shape = input_shape
        
        # Store predictions and targets for epoch-level metrics (using lists for memory efficiency)
        self.train_outputs = []
        self.train_targets = []
        self.val_outputs = []
        self.val_targets = []
        self.current_phase = 'train'
        
        # Hardware monitoring
        self.gpu = torch.cuda.is_available()
        
        # Track metrics over time (using list for memory efficiency)
        self.metrics_history = defaultdict(list)
        
        # Performance optimization flags
        self._should_log_histograms = False
        self._last_log_time = 0
        
        # For activation logging
        self.handles = []
        self.activations = {}
        
        # Register hooks for activation visualization if needed
        if model is not None:
            self._log_model_architecture()
            if histogram_freq > 0:
                self._register_hooks()
        
        # Initialize hardware monitoring
        self._init_hardware_monitoring()

    def _init_hardware_monitoring(self):
        """Initialize hardware monitoring setup."""
        self.gpu = torch.cuda.is_available()
        if not self.gpu:
            print("CUDA not available, GPU monitoring will be disabled.")

    def log_scalar(self, tag, scalar_value, global_step=None):
        """Log a scalar value to TensorBoard."""
        if hasattr(self, 'writer'):
            self.writer.add_scalar(tag, scalar_value, global_step)
            
    def on_epoch_begin(self, epoch, logs=None):
        # Reset storage at the beginning of each epoch
        self.train_outputs = []
        self.train_targets = []
        self.val_outputs = []
        self.val_targets = []
        self.current_phase = None

        # Only log histograms at specified frequency to reduce overhead
        self._should_log_histograms = (self.histogram_freq > 0 and 
                                     epoch % self.histogram_freq == 0)
        
        # Log hardware metrics at the start of each epoch
        self._log_hardware_metrics(epoch)
        
        # Log learning rate schedule at the beginning of training
        if epoch == 0 and hasattr(self, 'writer'):
            self._log_learning_rate_schedule()

    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        # Register hooks for all convolutional and linear layers
        for name, layer in self.model.named_children():
            if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, 
                               torch.nn.ReLU, torch.nn.MaxPool2d)):
                self.handles.append(
                    layer.register_forward_hook(get_activation(name))
                )

    def _log_model_architecture(self):
        """Log model architecture and graph to TensorBoard."""
        if self.model is None:
            return
            
        try:
            # Skip slow graph tracing - just log text summary
            # Note: writer.add_graph() is very slow, especially on GPU
            
            # Log model summary as text (simple string representation)
            model_summary = str(self.model)
            self.writer.add_text('Model/Architecture', f'```\n{model_summary}\n```')
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            param_info = f"Total Parameters: {total_params:,}\nTrainable Parameters: {trainable_params:,}"
            self.writer.add_text('Model/Parameters', param_info)
            
        except Exception as e:
            print(f"Could not log model architecture: {e}")

    def _log_hardware_metrics(self, step: int):
        """Log hardware utilization metrics."""
        try:
            # Log CPU utilization
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.writer.add_scalar('Hardware/CPU_Usage_Percent', cpu_percent, step)
            self.writer.add_scalar('Hardware/Memory_Usage_Percent', memory.percent, step)
            self.writer.add_scalar('Hardware/Memory_Used_GB', memory.used / (1024**3), step)
            
            # Log basic GPU metrics if available (without NVML)
            if self.gpu and torch.cuda.is_available():
                try:
                    # Get basic GPU metrics without NVML
                    gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    gpu_mem_cached = torch.cuda.memory_reserved() / (1024 ** 3)   # GB
                    
                    self.writer.add_scalar('Hardware/GPU_Memory_Allocated_GB', gpu_mem_alloc, step)
                    self.writer.add_scalar('Hardware/GPU_Memory_Cached_GB', gpu_mem_cached, step)
                    
                    # If NVML is available, try to get more detailed metrics
                    if HAS_NVML and pynvml is not None:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            
                            gpu_load = util.gpu
                            gpu_mem = (mem_info.used / mem_info.total) * 100
                            
                            self.writer.add_scalar('Hardware/GPU_Usage_Percent', gpu_load, step)
                            self.writer.add_scalar('Hardware/GPU_Memory_Utilization_Percent', gpu_mem, step)
                        except Exception as e:
                            if hasattr(pynvml, 'nvmlShutdown'):
                                pynvml.nvmlShutdown()
                            # Don't modify the global HAS_NVML here as it's used for initialization
                            pass
                            
                except Exception as e:
                    print(f"Error logging GPU metrics: {e}")
        except Exception as e:
            print(f"Failed to log hardware metrics: {e}")

    def _log_metrics(self, outputs, targets, prefix, epoch):
        if not outputs or not targets:
            return
            
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        
        # Convert logits to probabilities and predictions
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Calculate metrics
        accuracy = accuracy_score(targets, preds)
        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = f1_score(targets, preds, average='macro')
        
        # Log scalar metrics
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Loss': torch.tensor(0.0)  # Will be updated from training loop
        }
        
        # Update metrics history
        for name, value in metrics.items():
            self.metrics_history[f'{prefix}_{name}'].append(value)
            self.writer.add_scalar(f'{name}/{prefix}', value, epoch)
            
        # Log per-class metrics
        if self.num_classes <= 20:  # Avoid cluttering for too many classes
            class_report = classification_report(
                targets, preds, target_names=self.class_names, output_dict=True, zero_division=0
            )
            
            # Log per-class precision, recall, f1
            for i, class_name in enumerate(self.class_names):
                if class_name in class_report:
                    for metric in ['precision', 'recall', 'f1-score']:
                        self.writer.add_scalar(
                            f'Class/{class_name}_{metric}',
                            class_report[class_name][metric],
                            epoch
                        )
        
        # Log confusion matrix
        self._log_confusion_matrix(preds, targets, prefix, epoch)
        
        # Log ROC and PR curves
        if self.num_classes == 2:
            self._plot_roc_curve(probs, targets, prefix, epoch)
            self._plot_precision_recall_curve(probs, targets, prefix, epoch)
        else:
            self._plot_multiclass_roc(probs, targets, prefix, epoch)
            self._plot_multiclass_pr_curve(probs, targets, prefix, epoch)
        
        # Log predictions distribution
        self._plot_prediction_distribution(probs, targets, prefix, epoch)
    
    def _log_confusion_matrix(self, preds, targets, prefix, epoch):
        """Log confusion matrix."""
        try:
            cm = confusion_matrix(targets, preds)
            
            # Normalize the confusion matrix
            cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Count')
            
            # Add labels
            tick_marks = np.arange(len(self.class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_yticklabels(self.class_names)
            
            # Add text annotations
            thresh = 0.5
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                           ha="center", va="center", fontsize=8,
                           color="white" if cm_norm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Confusion Matrix - {prefix.capitalize()}')
            
            # Save to TensorBoard
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
            plt.close(fig)
            buf.seek(0)
            
            image = Image.open(buf)
            self.writer.add_image(
                f'Confusion_Matrix/{prefix}', 
                np.array(image), 
                global_step=epoch,
                dataformats='HWC'
            )
            
        except Exception as e:
            print(f"Error logging confusion matrix: {e}")
            
    def _plot_roc_curve(self, probs, targets, prefix, epoch):
        """Plot ROC curve for binary classification."""
        try:
            fpr, tpr, _ = roc_curve(targets, probs[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {prefix.capitalize()}')
            plt.legend(loc="lower right")
            
            # Save to TensorBoard
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            image = Image.open(buf)
            self.writer.add_image(
                f'ROC/{prefix}',
                np.array(image),
                global_step=epoch,
                dataformats='HWC'
            )
            
        except Exception as e:
            print(f"Error plotting ROC curve: {e}")
    
    def _plot_precision_recall_curve(self, probs, targets, prefix, epoch):
        """Plot precision-recall curve for binary classification."""
        precision, recall, _ = precision_recall_curve(targets, probs[:, 1])
        avg_precision = average_precision_score(targets, probs[:, 1])
        
        fig = plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where='post', label=f'Precision-Recall (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        self.writer.add_figure(f'Precision_Recall_Curve/{prefix}', fig, epoch)
        plt.close(fig)
    
    def _plot_multiclass_roc(self, probs, targets, prefix, epoch):
        """Plot ROC curves for multiclass classification (one-vs-rest)."""
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve((targets == i).int(), probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(torch.nn.functional.one_hot(targets, num_classes=self.num_classes).numpy().ravel(), 
                                                probs.numpy().ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot all ROC curves
        fig = plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, color in zip(range(self.num_classes), colors):
            if i < len(colors):  # In case we have more classes than colors
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass ROC Curves')
        plt.legend(loc="lower right")
        
        self.writer.add_figure(f'ROC_Curve/{prefix}', fig, epoch)
        plt.close(fig)
    
    def _plot_multiclass_pr_curve(self, probs, targets, prefix, epoch):
        """Plot precision-recall curves for multiclass classification."""
        # Compute PR curve and PR area for each class
        precision = dict()
        recall = dict()
        avg_precision = dict()
        
        # Convert to one-hot encoding for precision_recall_curve
        y_test = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).numpy()
        
        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], probs[:, i].numpy())
            avg_precision[i] = average_precision_score(y_test[:, i], probs[:, i].numpy())
        
        # Compute micro-average PR curve and PR area
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test.ravel(), probs.numpy().ravel())
        avg_precision["micro"] = average_precision_score(y_test, probs.numpy(), average="micro")
        
        # Plot all PR curves
        fig = plt.figure(figsize=(10, 8))
        plt.step(recall["micro"], precision["micro"], where='post',
                 label=f'micro-average PR (AP = {avg_precision["micro"]:.2f})')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, color in zip(range(self.num_classes), colors):
            if i < len(colors):  # In case we have more classes than colors
                plt.step(recall[i], precision[i], where='post', color=color,
                         label=f'Class {i} (AP = {avg_precision[i]:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Multiclass Precision-Recall Curves')
        plt.legend(loc='lower left')
        
        self.writer.add_figure(f'Precision_Recall_Curve/{prefix}', fig, epoch)
        plt.close(fig)
    
    def _plot_prediction_distribution(self, probs, targets, prefix, epoch):
        """Plot distribution of prediction probabilities."""
        fig = plt.figure(figsize=(12, 6))
        
        # For each class, plot the distribution of prediction probabilities
        for i in range(self.num_classes):
            class_probs = probs[targets == i][:, i].numpy()
            if len(class_probs) > 0:
                plt.hist(class_probs, bins=20, alpha=0.5, 
                         label=f'Class {i} (n={len(class_probs)})',
                         range=(0, 1))
        
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Prediction Probability Distribution by Class')
        plt.legend()
        
        # Convert figure to image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Add to tensorboard
        image = Image.open(buf)
        self.writer.add_image(f'{prefix}_prediction_distribution', 
                            np.array(image), 
                            global_step=epoch,
                            dataformats='HWC')
        
        # Flush to ensure all events are written to disk
        self.writer.flush()

    def _log_weights_and_gradients(self, epoch: int):
        """Log weight and gradient distributions/histograms."""
        if self.model is None:
            return
            
        try:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # Log weight distributions
                    self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                    
                    # Log gradient distributions (if gradients exist)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                        
                        # Log gradient norms
                        grad_norm = param.grad.norm().item()
                        self.writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, epoch)
                    
                    # Log weight statistics
                    self.writer.add_scalar(f'Weight_Stats/{name}_mean', param.data.mean().item(), epoch)
                    self.writer.add_scalar(f'Weight_Stats/{name}_std', param.data.std().item(), epoch)
                    self.writer.add_scalar(f'Weight_Stats/{name}_min', param.data.min().item(), epoch)
                    self.writer.add_scalar(f'Weight_Stats/{name}_max', param.data.max().item(), epoch)
                    
        except Exception as e:
            if not hasattr(self, '_weight_logging_warning_shown'):
                print(f"Could not log weights/gradients: {e}")
                self._weight_logging_warning_shown = True
        
    def _log_embeddings(self, epoch):
        """Log embeddings to TensorBoard Projector with proper visualization."""
        if not hasattr(self, 'writer') or not hasattr(self.model, 'get_embeddings'):
            return
            
        try:
            # Use the same images we created the sprite with
            if not hasattr(self, '_embedding_indices') or not hasattr(self, 'val_loader'):
                return
                
            # Get the model's device
            device = next(self.model.parameters()).device
            
            # Get the same subset of data we used for the sprite
            dataset = self.val_loader.dataset
            if hasattr(dataset, 'indices') and dataset.indices is not None:
                # Handle SubsetRandomSampler
                indices = dataset.indices
                subset = torch.utils.data.Subset(dataset.dataset, [indices[i] for i in self._embedding_indices])
            else:
                subset = torch.utils.data.Subset(dataset, self._embedding_indices)
                
            loader = torch.utils.data.DataLoader(
                subset,
                batch_size=len(self._embedding_indices),
                shuffle=False
            )
            
            # Get the data
            try:
                inputs, labels = next(iter(loader))
                inputs = inputs.to(device)
                
                # Get embeddings
                with torch.no_grad():
                    self.model.eval()
                    embeddings = self.model.get_embeddings(inputs)
                
                # Ensure we have the sprite images
                if not hasattr(self, '_sprite_path') or not os.path.exists(self._sprite_path):
                    self._create_sprite_image()
                
                # Log embeddings with proper metadata and sprite image
                self.writer.add_embedding(
                    mat=embeddings,
                    metadata=labels.tolist(),
                    label_img=self._embedding_images,
                    global_step=epoch,
                    tag='mnist_embeddings',
                    metadata_header=['label']
                )
                
                # Also save the embeddings and labels for potential later use
                torch.save({
                    'embeddings': embeddings.cpu(),
                    'labels': labels.cpu(),
                    'images': self._embedding_images.cpu()
                }, os.path.join(self.writer.log_dir, f'embeddings_epoch_{epoch}.pt'))
                
            except StopIteration:
                pass
                
        except Exception as e:
            import traceback
            print(f"Warning: Could not log embeddings: {e}")
            print(traceback.format_exc())
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
            
        # Log training metrics
        if self.train_outputs and self.train_targets:
            self._log_metrics(self.train_outputs, self.train_targets, 'train', epoch)
            
        # Log validation metrics
        if self.val_outputs and self.val_targets:
            self._log_metrics(self.val_outputs, self.val_targets, 'val', epoch)
            
        # Log losses if available in logs
        if 'train_loss' in logs:
            self.writer.add_scalar('Loss/train', logs['train_loss'], epoch)
        if 'val_loss' in logs:
            self.writer.add_scalar('Loss/val', logs['val_loss'], epoch)
        
        # Log weight and gradient distributions
        self._log_weights_and_gradients(epoch)
        
        # Log embeddings
        self._log_embeddings(epoch)
            
        # Ensure all data is written to disk
        self.writer.flush()

    def _log_learning_rate_schedule(self):
        """Log the learning rate schedule to TensorBoard."""
        if not hasattr(self, 'hparams') or 'learning_rate' not in self.hparams:
            return
            
        try:
            # Get the initial learning rate from hparams
            initial_lr = self.hparams['learning_rate']
            if isinstance(initial_lr, (list, tuple)):
                initial_lr = initial_lr[0]  # Take first learning rate if it's a list
            initial_lr = float(initial_lr)
            
            # Get number of epochs, default to 10 if not specified
            epochs = int(self.hparams.get('epochs', 10))
            
            # Create a figure for the learning rate schedule
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot the learning rate schedule
            x = np.linspace(0, epochs, 1000)
            y = []
            
            # Calculate the learning rate at each point
            for step in x:
                # 5-epoch warmup
                if step < 5:
                    # Linear warmup
                    y.append(initial_lr * (step / 5))
                else:
                    # Cosine decay
                    progress = (step - 5) / max(1, epochs - 5)  # Avoid division by zero
                    y.append(0.5 * initial_lr * (1 + np.cos(np.pi * min(progress, 1.0))))
            
            # Plot the schedule
            ax.plot(x, y, linewidth=2.5, label=f'Learning Rate (max: {initial_lr:.2e})')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, pad=20)
            ax.grid(True, alpha=0.3)
            
            # Customize the plot
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_ylim(bottom=0)  # Start y-axis from 0
            
            # Add warmup indicator
            ax.axvline(x=5, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.text(5.1, initial_lr * 0.1, 'Warmup Ends', 
                   rotation=90, verticalalignment='bottom',
                   color='r', fontsize=10, alpha=0.8)
            
            # Add legend and adjust layout
            ax.legend(loc='upper right', fontsize=10)
            plt.tight_layout()
            
            # Save to TensorBoard
            self.writer.add_figure('learning_rate/schedule', fig, close=True)
            
            # Also log the final learning rate as a scalar for HPARAMS
            final_lr = y[-1]
            self.writer.add_scalar('hparams/final_learning_rate', final_lr, 0)
            
        except Exception as e:
            print(f"Warning: Could not log learning rate schedule: {e}")

    def _log_hparams_metrics(self, final_metrics):
        """Log hyperparameters and metrics to TensorBoard HPARAMS."""
        try:
            # Clean up hparams to only include scalar values
            hparams = {}
            for k, v in self.hparams.items():
                # Convert non-scalar values to strings
                if isinstance(v, (int, float, str, bool, torch.Tensor)):
                    if torch.is_tensor(v):
                        hparams[k] = v.item() if v.numel() == 1 else str(v.tolist())
                    else:
                        hparams[k] = v
                else:
                    hparams[k] = str(v)
            
            # Log to TensorBoard HPARAMS
            self.writer.add_hparams(
                hparam_dict=hparams,
                metric_dict=final_metrics,
                run_name=os.path.basename(self.writer.log_dir)
            )
            
            # Save hparams and metrics to JSON files
            hparams_path = os.path.join(self.writer.log_dir, 'hparams.json')
            with open(hparams_path, 'w') as f:
                import json
                json.dump(hparams, f, indent=2, default=str)  # Handle non-serializable types
            
            metrics_path = os.path.join(self.writer.log_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            # Log as text for easy viewing in TensorBoard
            def dict_to_markdown(d, indent=0):
                markdown = []
                for k, v in d.items():
                    if isinstance(v, dict):
                        markdown.append(f"{'  '*indent}- **{k}**:")
                        markdown.append(dict_to_markdown(v, indent+1))
                    else:
                        markdown.append(f"{'  '*indent}- **{k}**: {v}")
                return '\n'.join(markdown)
            
            # Log hyperparameters
            hparams_md = f"## Hyperparameters\n{dict_to_markdown(hparams)}"
            self.writer.add_text('Hyperparameters', hparams_md)
            
            # Log metrics
            metrics_md = f"## Metrics\n{dict_to_markdown(final_metrics)}"
            self.writer.add_text('Metrics', metrics_md)
            
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
    
    def _setup_embeddings(self):
        """Setup embeddings for visualization in TensorBoard Projector."""
        if not hasattr(self, 'writer') or not hasattr(self.model, 'get_embeddings'):
            return
            
        # Create a sprite image for MNIST digits (28x28 grayscale)
        self._create_sprite_image()
        
        # Configure the projector
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        
        # The embedding variable will be populated during training
        embedding.tensor_name = 'embeddings'
        embedding.metadata_path = os.path.join(self.writer.log_dir, 'metadata.tsv')
        
        if hasattr(self, '_sprite_path'):
            embedding.sprite.image_path = os.path.basename(self._sprite_path)
            embedding.sprite.single_image_dim.extend([28, 28])
        
        # Save the config
        projector.visualize_embeddings(self.writer, config)
    
    def _create_sprite_image(self):
        """Create a sprite image for MNIST digits with proper formatting."""
        from torchvision.datasets import MNIST
        from torchvision import transforms
        import torchvision.utils as vutils
        import numpy as np
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Define transformations - we'll denormalize the images for better visualization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # Add padding to make digits more visible in the sprite
            transforms.Pad(2, fill=0, padding_mode='constant')
        ])
        
        # Load MNIST dataset
        dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        
        # Get balanced samples per class (10 classes, 10 samples each)
        samples_per_class = 10
        indices = []
        for class_idx in range(10):
            class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx][:samples_per_class]
            indices.extend(class_indices)
        
        # Create subset with balanced classes
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=len(indices), shuffle=False)
        
        # Get images and labels
        images, labels = next(iter(loader))
        
        # Denormalize images for better visualization
        mean = torch.tensor([0.1307]).view(1, 1, 1, 1)
        std = torch.tensor([0.3081]).view(1, 1, 1, 1)
        images_denorm = images * std + mean
        
        # Create sprite image with a grid of images (10x10 by default)
        sprite_image = vutils.make_grid(
            images_denorm, 
            nrow=10,  # 10 images per row
            padding=2,
            normalize=True,
            scale_each=True,
            pad_value=1.0  # White padding between images
        )
        
        # Save sprite image
        self._sprite_path = os.path.join(self.writer.log_dir, 'sprite.png')
        vutils.save_image(sprite_image, self._sprite_path, nrow=10, padding=2)
        
        # Save metadata with both index and label
        metadata_path = os.path.join(self.writer.log_dir, 'metadata.tsv')
        with open(metadata_path, 'w') as f:
            f.write('Index\tLabel\tName\n')
            for i, label in enumerate(labels):
                f.write(f'{i}\t{label}\t{label}\n')
                
        # Save the indices for later use in _log_embeddings
        self._embedding_indices = indices
        
        # Save the denormalized images for visualization
        self._embedding_images = images_denorm
    
    def on_train_begin(self, logs=None):
        """Called once at the beginning of training."""
        # Initialize metrics history
        self.metrics_history = defaultdict(list)
        
        # Log model architecture (fast - no graph tracing)
        if self.model is not None:
            self._log_model_architecture()
            
        # Skip embedding setup at start - it's slow and not needed immediately
        # Embeddings will be created lazily when first needed
            
        # Log hyperparameters
        if self.hparams:
            self._log_hparams_metrics({})
            
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if not hasattr(self, 'hparams') or not hasattr(self, 'metrics_history'):
            return
            
        try:
            # Get final metrics
            final_metrics = {
                'hparam/final_train_loss': float(self.metrics_history.get('train_loss', [0])[-1]),
                'hparam/final_val_loss': float(self.metrics_history.get('val_loss', [0])[-1]),
                'hparam/final_val_accuracy': float(self.metrics_history.get('val_accuracy', [0])[-1]),
                'hparam/best_val_accuracy': float(max(self.metrics_history.get('val_accuracy', [0]))),
                'hparam/final_epoch': len(self.metrics_history.get('train_loss', [0]))
            }
            # Add any additional metrics from callbacks
            if logs is not None:
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        final_metrics[f'hparam/{k}'] = v
            
            # Log to HPARAMS
            self._log_hparams_metrics(final_metrics)
            
            # Log final metrics as scalars for the last epoch
            for k, v in final_metrics.items():
                if k.startswith('hparam/'):
                    self.writer.add_scalar(k, v, 0)  # Step 0 for final metrics
            
            metrics_str = '\n'.join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
            self.writer.add_text('Final Metrics', f"```\n{metrics_str}\n```")
            
        except Exception as e:
            print(f"Warning: Error during training completion: {e}")
            
        finally:
            # Always close the writer
            if hasattr(self, 'writer'):
                self.writer.close()
