# Quick Start Guide - E.D.I.T.H. on Windows

This is a condensed guide to get E.D.I.T.H. running on Windows with your dataset.

## 1. One-Time Setup (5-10 minutes)

```cmd
# Open Command Prompt and navigate to where you want to install
cd C:\

# Download E.D.I.T.H. (Option A: Git)
git clone https://github.com/boss0007000/E.D.I.T.H..git
cd E.D.I.T.H.

# Or Option B: Download ZIP and extract, then:
# cd C:\E.D.I.T.H.

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies (takes 5-10 minutes)
pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

## 2. Prepare Your Dataset

Organize your images like this:

```
E.D.I.T.H.\
└── dataset\
    ├── acura_cl_1997\
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── honda_civic_2005\
    │   ├── image1.jpg
    │   └── ...
    └── toyota_camry_2020\
        ├── image1.jpg
        └── ...
```

**Folder naming format**: `<make>_<model>_<year>`

Examples:
- `toyota_camry_2020`
- `honda_civic_2005`
- `bmw_3_series_2018`
- `ford_f_150_2022`

## 3. Check Your Dataset

```cmd
python check_dataset.py --dataset dataset
```

This will show you:
- Number of classes (vehicle types)
- Number of images per class
- Data quality warnings
- Suggestions for training

## 4. Train the Model (Optional)

If you have a dataset, train a custom model:

```cmd
# Basic training (recommended for first time)
python train.py --dataset dataset --epochs 50 --batch-size 32

# GPU training (if you have NVIDIA GPU)
python train.py --dataset dataset --epochs 100 --batch-size 64
```

Training will create:
- `models\vehicle_classifier.pth` (trained model)
- `models\class_names.json` (class information)

## 5. Run Vehicle Identification

### Option A: Test with generated image (no training needed)
```cmd
python test_pipeline.py
```

### Option B: Identify vehicles in your images

**Using batch file (easiest):**
```cmd
run.bat C:\path\to\your\car_image.jpg
```

**Using Python directly:**
```cmd
# With default mock model
python example.py --image C:\Users\YourName\Pictures\car.jpg

# With your trained model
python example.py --image car.jpg --weights models\vehicle_classifier.pth --verbose
```

## Common Commands Quick Reference

```cmd
# Activate environment (do this every time you open a new terminal)
venv\Scripts\activate

# Check dataset structure
python check_dataset.py --dataset dataset

# Train model
python train.py --dataset dataset --epochs 50

# Identify vehicle (with batch file)
run.bat path\to\image.jpg

# Identify vehicle (with Python)
python example.py --image path\to\image.jpg

# Identify with trained model
python example.py --image path\to\image.jpg --weights models\vehicle_classifier.pth

# Run test
python test_pipeline.py
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| "Python is not recognized" | Add Python to PATH in Environment Variables |
| "pip is not recognized" | Reinstall Python with "Add to PATH" checked |
| Dependencies fail to install | Update pip: `python -m pip install --upgrade pip` |
| Out of memory during training | Reduce batch size: `--batch-size 16` |
| Slow processing | Install CUDA PyTorch or reduce image size |
| Can't find image | Use full path: `C:\full\path\to\image.jpg` |

## Need More Help?

- **Complete Windows Guide**: See [WINDOWS_INSTALL.md](WINDOWS_INSTALL.md)
- **Training Details**: See [TRAINING.md](TRAINING.md)
- **API Documentation**: See [API.md](API.md)
- **Main Documentation**: See [README.md](README.md)
- **GitHub Issues**: https://github.com/boss0007000/E.D.I.T.H./issues

## Performance Tips

1. **Use GPU**: Install CUDA PyTorch for 5x faster processing
2. **Batch Training**: Use larger batch sizes if you have enough RAM/VRAM
3. **Data Augmentation**: Already included in train.py
4. **Regular Checkpoints**: Training saves checkpoints every 10 epochs

---

**Ready to identify vehicles? Start with Step 1!** 🚗🔍
