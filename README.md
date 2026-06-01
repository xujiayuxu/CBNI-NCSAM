# CBNI-NCSAM

This project contains the improved experimental implementation of NCSAM (Noise-Compensated Sharpness-Aware Minimization). The main training entry point is `train_noise_CBN.py`. It implements Cluster-Based Noise Inversion (CBNI): mini-batch features are clustered with KMeans, temporary noisy labels are sampled according to cluster-level prediction uncertainty, the resulting labels are used to estimate the noise gradient, and the SAM second-step perturbation is corrected with that estimate.

The hyperparameter settings follow the experimental configuration in `/home/xjy/下载/CAC_NCSAM_improve-1.pdf`.

## Code Structure

```text
.
├── train_noise_CBN.py          # Main training script
├── sam.py                      # SAM optimizer wrapper
├── data/
│   ├── cifar_fsam.py           # Cutout-based CIFAR/noisy-CIFAR loaders used by train_noise_CBN.py
│   ├── noise_ind_datasets.py   # CIFAR-10/100 noisy-data loaders
│   ├── noisecifar.py           # CIFAR dataset definitions
│   ├── randaugment.py          # FixMatch-style weak/strong augmentations
│   └── utils.py                # Noisy-label generation utilities
├── model/
│   ├── presnet.py              # PreResNet-34 backbone
│   ├── resnet.py               # Standard ResNet and PreActResNet alternatives
│   └── smooth_cross_entropy.py # Smoothed cross-entropy loss
├── utility/
│   ├── scheduler.py            # Cosine learning-rate scheduler
│   ├── save_file.py            # Result archiving
│   ├── log.py                  # Evaluation logging
│   ├── loading_bar.py          # Console progress-bar renderer used by utility/log.py
│   ├── time_record.py          # Per-stage wall-clock timing accumulator
│   └── bypass_bn.py            # BN-statistics control during SAM updates
└── other/
    ├── accuracy.txt
    └── whole_train_time.txt
```

## Added Code Files

The current training entry point imports several support files in addition to the core NCSAM implementation:

| File | Purpose |
| --- | --- |
| `data/cifar_fsam.py` | Provides `Cutout`, CIFAR noisy-label dataset wrappers, and `get_datasets_cutout(args)` for `CIFAR10_noise` / `CIFAR100_noise` runs. It also contains hooks for Food101N, Tiny-ImageNet, Animal-10N, and Clothing1M loaders; those external loader modules are not included in this repository. |
| `model/resnet.py` | Provides standard `ResNet18/34/50/101/152`, `ResNet18_reg`, and `PreResNet18/34` alternatives. Its `ResNet.forward(..., return_features=True)` path returns pooled feature vectors for methods that need feature extraction. The default training script still uses `model/presnet.py`. |
| `utility/time_record.py` | Defines `TIME_RECORD`, a lightweight timer that accumulates elapsed time between named `start` / `end` events. `train_noise_CBN.py` uses it for per-epoch stage timing. |
| `utility/loading_bar.py` | Defines `LoadingBar`, the Unicode progress bar used by `utility/log.py` for console training output. |

## Environment

The current machine reports the following main environment:

```text
Python        3.9.23
PyTorch       2.6.0+cu118
TorchVision   0.21.0+cu118
NumPy         1.26.3
scikit-learn  1.6.1
SciPy         1.13.1
CUDA runtime  11.8
```

The code depends on `torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `Pillow`, and `tqdm`. Install the core dependencies with:

```bash
pip install torch torchvision numpy scipy scikit-learn pillow tqdm
```

## Data Path

The current code uses the following fixed data root in `train_noise_CBN.py`:

```python
root='/home/xjy/code/Label_noise_experiment/data'
```

Before running training, make sure this directory contains CIFAR-10 or CIFAR-100. If your data is stored elsewhere, update the `root` argument passed to `cifar_dataloader(...)` in `train_noise_CBN.py`.

## Default Training Flow

Training runs for 200 epochs:

1. The first 50 epochs use standard SGD warmup.
2. CBNI-NCSAM is enabled after epoch 50.
3. Each batch first computes the normal gradient and the first SAM perturbation step.
4. When CBNI is enabled, mini-batch features are extracted from the model and clustered with KMeans.
5. Each cluster estimates a cluster-level inversion rate from the top-2 logit gap.
6. The noise gradient `gn` is computed only on samples whose temporary label changes.
7. The SAM perturbation is corrected with `learning_rate * noise_grad_k * scale * gn`, then the second SAM step is applied.

## Key Hyperparameters

Code defaults:

| Argument | Default | Description |
| --- | ---: | --- |
| `--datasets` | `cifar-100` | Dataset, supports `cifar-10` / `cifar-100` |
| `--model` | `Presnet34` | Current backbone is PreResNet-34 |
| `--epochs` | `200` | Total training epochs |
| `--warmup_epochs` | `50` | SGD warmup epochs |
| `--batch_size` | `128` | Batch size |
| `--learning_rate` | `0.05` | Initial learning rate |
| `--momentum` | `0.9` | SGD momentum |
| `--weight_decay` | `0.001` | Weight decay |
| `--rho` | `0.05` | SAM perturbation radius |
| `--label_smoothing` | `0.1` | Label smoothing used for noise-gradient estimation |
| `--noise_type` | `ins` | Instance-dependent noise |
| `--noise_ratio` | `0.2` | Dataset noise ratio |
| `--flip_ratio` | `0.2` | Maximum CBNI inversion rate, corresponding to `rmax` in the paper |
| `--num_clusters` | `3` | Number of KMeans clusters; the paper's default experiments use `K=5` |
| `--noise_grad_k` | `5.0` | Noise-gradient compensation coefficient |

The PDF denotes the compensation strength as `γ`. In the current code, the effective value is:

```text
γ = learning_rate * noise_grad_k
```

With the default `learning_rate=0.05`:

```text
noise_grad_k = γ / 0.05
```

## PDF Experimental Configuration

Table I in `CAC_NCSAM_improve-1.pdf` gives the following main CBNI-NCSAM hyperparameters:

| Dataset | Noise Ratio | K | rmax / `--flip_ratio` | γ | Corresponding `--noise_grad_k` |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 20% | 5 | 0.20 | 0.10 | 2 |
| CIFAR-10 | 40% | 5 | 0.40 | 0.25 | 5 |
| CIFAR-10 | 60% | 5 | 0.40 | 1.25 | 25 |
| CIFAR-100 | 20% | 5 | 0.20 | 0.25 | 5 |
| CIFAR-100 | 40% | 5 | 0.40 | 0.75 | 15 |
| CIFAR-100 | 60% | 5 | 0.40 | 1.50 | 30 |

General training settings:

| Item | Setting |
| --- | --- |
| Backbone | PreAct ResNet-34 / current code `PreResNet34` |
| Epochs | 200 |
| Warmup | First 50 epochs use SGD |
| Batch size | 128 |
| Learning rate | 0.05 |
| Momentum | 0.9 |
| Weight decay | 0.001 |
| SAM rho | 0.05 |
| Noise type | Instance-dependent noise (`ins`) |

## Run Examples

CIFAR-100 with 20% instance-dependent noise:

```bash
python train_noise_CBN.py \
  --datasets cifar-100 \
  --noise_type ins \
  --noise_ratio 0.2 \
  --flip_ratio 0.2 \
  --num_clusters 5 \
  --noise_grad_k 5
```

CIFAR-100 with 40% instance-dependent noise:

```bash
python train_noise_CBN.py \
  --datasets cifar-100 \
  --noise_type ins \
  --noise_ratio 0.4 \
  --flip_ratio 0.4 \
  --num_clusters 5 \
  --noise_grad_k 15
```

CIFAR-100 with 60% instance-dependent noise:

```bash
python train_noise_CBN.py \
  --datasets cifar-100 \
  --noise_type ins \
  --noise_ratio 0.6 \
  --flip_ratio 0.4 \
  --num_clusters 5 \
  --noise_grad_k 30
```

If you run CIFAR-10, first make sure `class_num` in `train_noise_CBN.py` matches the dataset. The current code hard-codes:

```python
class_num = 100
```

For CIFAR-10 experiments, change it to:

```python
class_num = 10
```

## Outputs

Training writes to:

```text
other/whole_train_time.txt
other/accuracy-noise+sam+dw-cifar10-f0.4.txt
```

After training, `utility/save_file.py` copies the following files into the result directory:

```text
results/{dataset}/{model}/sam_noise_{noise_ratio}/{timestamp}/
├── whole_train_time.txt
├── accuracy.txt
├── sam.py
└── train_noise_CBN.py
```

Note: test accuracy is currently written to `other/accuracy-noise+sam+dw-cifar10-f0.4.txt`, while the archive list contains `other/accuracy.txt`. If you need complete accuracy archiving, make the write path in `train_noise_CBN.py` consistent with `other/accuracy.txt`.

## Notes

- The current code is organized mainly around the CIFAR-100 default configuration. Handle `class_num` before running CIFAR-10.
- `--num_clusters` defaults to 3 in code, but the PDF main experiments use K=5. Pass `--num_clusters 5` explicitly when reproducing the reported experiments.
- CBNI is designed for instance-dependent noise. The PDF also notes that the local-feature-structure assumption is weaker under symmetric noise, where CBNI may underperform the original NCSAM.
- The code selects the device through `torch.cuda.is_available()`. The detected PyTorch build includes CUDA 11.8, but no available GPU was detected during this check.
- KMeans runs inside every batch. Larger `K` increases overhead. Table V in the PDF shows that K=5 and K=10 are close, so K=5 is a stable default.
