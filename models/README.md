# Models Directory

This directory is used to store trained model weights and related files.

## Files Created During Training

When you run `train.py`, the following files will be created here:

- `vehicle_classifier.pth` - Best model weights (based on validation accuracy)
- `class_names.json` - List of class names (folder names from dataset)
- `class_info.json` - Detailed class information (make, model, year parsed from folder names)
- `checkpoint_epoch_*.pth` - Training checkpoints saved at regular intervals

## Using Trained Models

To use a trained model for inference:

```bash
python example.py --image path/to/image.jpg --weights models/vehicle_classifier.pth
```

Or in Windows:
```cmd
python example.py --image path\to\image.jpg --weights models\vehicle_classifier.pth
```

## Pre-trained Models

If you have pre-trained models, place them in this directory with the expected filenames:

- Model weights: `vehicle_classifier.pth`
- Class names: `class_names.json`

## Notes

- Model files are not tracked by Git (see `.gitignore`)
- Training will overwrite existing `vehicle_classifier.pth` if validation accuracy improves
- Checkpoints allow you to resume training or use models from specific epochs
