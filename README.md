# BMPD

Code for **BMPD: Beamwise Motion Polar Decomposition for Cross-User Gesture Recognition Using Millimeter-Wave Sensing**.

The repository contains the BMPD and dimension-matched FrameDiff experiments with LiteCNN and ResNet-18. Both representations are evaluated with the same eight-fold leave-one-user-out (LOUO) splits and training seeds 42, 43, and 44.

## Requirements

The experiments were run with Python 3.10, PyTorch 2.0.1, torchvision 0.15.2, CUDA 11.7, and an NVIDIA GeForce RTX 2080 Ti.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Download 5GmmGesture from the mmHSense [IEEE DataPort page](https://ieee-dataport.org/documents/mmhsense-multi-modal-and-distributed-mmwave-isac-datasets-human-sensing). Further dataset information is available in the [authors' repository](https://github.com/nisarnabeel/Multi-Modal-and-Distributed-mmWave-ISAC-Datasets-for-Human-Sensing).

Set `data_dir` in the YAML configuration to a directory containing:

```text
PPBP_data_user1.pth
labels_user1.pth
...
PPBP_data_user8.pth
labels_user8.pth
```

The experimental set contains the first 539 stored windows for every participant-class combination, giving 34,496 windows. The code creates local float32 caches without modifying the original files.

## Representation

FrameDiff uses 19 adjacent-frame differences and one zero channel. BMPD uses 19 regularized beamwise temporal profiles and one log-relative RMS magnitude map. The reported BMPD configuration uses the lower median of the 2,800 beam-pair RMS values as the within-window reference, a mean fallback when the median does not exceed `1e-6`, and `tau = 0.10`.

Basic implementation checks can be run with:

```bash
python main.py --self-test
```

## Training and evaluation

Run the complete 96-run experiment:

```bash
python main.py --config config_train.yaml
```

LiteCNN is trained for 30 epochs with Adam, learning rate 0.001, and batch size 32. ResNet-18 is trained without pretrained weights for up to 150 epochs with Adam, learning rate 0.0003, batch size 32, and early-stopping patience 20. Checkpoints are selected by source-validation accuracy.

Completed runs are skipped and interrupted runs resume from `latest.pth`. A subset can be selected without changing the stored protocol:

```bash
python main.py --config config_train.yaml --backbones litecnn --methods bmpd --seeds 42 --users 1
```

Re-evaluate checkpoints produced under `outputs/train/` with:

```bash
python main.py --config config_test.yaml
```

Run artifacts, predictions, metrics, and configuration records are written under `outputs/`. Aggregate reference values are provided in `expected_results/`.

## Evaluation protocol

Each input has dimensions `20 x 50 x 56`. In every LOUO fold, one participant provides 4,312 outer-test windows. For each source participant and class, 462 windows are used for training and 77 for source checkpoint validation. This gives 25,872 training, 4,312 validation, and 4,312 test windows per fold. The held-out participant is not used for training or checkpoint selection.

The seven consecutive 77-window blocks reproduce the published split but cannot be independently identified as the original recording sessions from the released tensors. Metrics are averaged across the three seeds within each held-out participant and then equally across the eight participants.

## Results

The following results were obtained with the protocol above and are reported in the manuscript.

| Backbone | Method | Accuracy (%) | Macro-F1 (%) | Worst user (%) | Seed SD (pp) |
|---|---|---:|---:|---:|---:|
| LiteCNN | FrameDiff | 84.14 | 83.98 | 76.34 | 0.37 |
| LiteCNN | BMPD | 88.72 | 88.66 | 81.82 | 0.14 |
| ResNet-18 | FrameDiff | 79.70 | 79.30 | 70.11 | 1.10 |
| ResNet-18 | BMPD | 83.15 | 82.99 | 75.26 | 0.39 |

BMPD and FrameDiff are compared within each backbone. The absolute LiteCNN and ResNet-18 results are not used as an architecture ranking because their training schedules differ.

## References

- [mmHSense: Multi-Modal and Distributed mmWave ISAC Datasets for Human Sensing](https://doi.org/10.1109/ACCESS.2026.3691174)
- [Integrated Sensing and Communications for Real-time Avatar Control in XR over 5G](https://arxiv.org/abs/2606.23771)
