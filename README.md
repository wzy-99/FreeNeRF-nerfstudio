# (unofficial) FreeNeRF NeRFStudio implementation

## Installation

```bash
pip install -e .
```

## Usage

```bash
ns-train free-nerf --vis wandb --project-name free-nerf --experiment-name freenerf --max-num-iterations 30000 --steps-per-eval-all-images 30000 --pipeline.model.T 30000 blender-data --data your_path_to_data
```

You can compare it with the vanilla nerf method

```bash
ns-train vanilla-nerf --vis wandb --project-name free-nerf --experiment-name nerf --max-num-iterations 30000 --steps-per-eval-all-images 30000  blender-data --data your_path_to_data
```

## Notes

- The results of the original paper have not yet been reproduced

## TODO

- [ ] Implement the **Occlusion Regularization**