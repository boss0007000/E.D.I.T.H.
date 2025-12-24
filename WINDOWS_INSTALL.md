# E.D.I.T.H. - Windows Installation Guide

## Complete Setup Guide for Windows

This guide provides step-by-step instructions to install and run E.D.I.T.H. (Enhanced Deep Intelligence for Tactical Handling) on Windows.

---

## Prerequisites

### 1. System Requirements
- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: At least 5GB free space
- **GPU** (Optional): NVIDIA GPU with CUDA support for faster processing

### 2. Required Software

#### Python 3.8 or Higher
1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important**: During installation, check "Add Python to PATH"
3. Verify installation:
   ```cmd
   python --version
   ```
   Should show: `Python 3.8.x` or higher

#### Git (Optional, for cloning)
1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Use default installation settings
3. Verify:
   ```cmd
   git --version
   ```

---

## Installation Steps

### Step 1: Download E.D.I.T.H.

**Option A: Using Git**
```cmd
git clone https://github.com/boss0007000/E.D.I.T.H..git
cd E.D.I.T.H.
```

**Option B: Download ZIP**
1. Go to https://github.com/boss0007000/E.D.I.T.H.
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file to a folder (e.g., `C:\E.D.I.T.H.`)
4. Open Command Prompt and navigate to the folder:
   ```cmd
   cd C:\E.D.I.T.H.
   ```

### Step 2: Create a Virtual Environment (Recommended)

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command prompt.

### Step 3: Install Dependencies

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

This will install all required packages including PyTorch, OpenCV, and other dependencies.

**Note**: Installation may take 5-10 minutes depending on your internet speed.

### Step 4: Verify Installation

```cmd
python verify_installation.py
```

If successful, you should see:
```
✓ All imports successful
✓ Pipeline initialized successfully
✓ Installation verified!
```

---

## Running E.D.I.T.H.

### Quick Test Run

Test the pipeline with a generated image:

```cmd
python test_pipeline.py
```

### Run with Your Own Image

```cmd
python example.py --image path\to\your\image.jpg
```

Example:
```cmd
python example.py --image C:\Users\YourName\Pictures\car.jpg
```

### Using the Windows Batch File

For convenience, you can use the provided batch file:

```cmd
run.bat path\to\image.jpg
```

This will automatically:
1. Activate the virtual environment (if it exists)
2. Run the identification pipeline
3. Display results

---

## Training Models with Your Dataset

If you have a dataset organized as:
```
dataset/
├── acura_cl_1997/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── honda_civic_2005/
│   ├── image1.jpg
│   └── ...
└── toyota_camry_2020/
    ├── image1.jpg
    └── ...
```

### Step 1: Prepare Your Dataset

Place your dataset folder in the E.D.I.T.H. directory:
```
E.D.I.T.H.\
├── dataset\
│   ├── acura_cl_1997\
│   ├── honda_civic_2005\
│   └── ...
├── vehicle_identification\
├── example.py
└── ...
```

### Step 2: Train the Classification Model

```cmd
python train.py --dataset dataset --epochs 50 --batch-size 32
```

Options:
- `--dataset`: Path to dataset folder (default: `dataset`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--gpu`: Use GPU if available (automatically detected)

### Step 3: Use Trained Model

After training, the model will be saved to `models/vehicle_classifier.pth`

To use it:
```cmd
python example.py --image your_image.jpg --weights models\vehicle_classifier.pth
```

---

## GPU Acceleration (Optional)

For faster processing with NVIDIA GPU:

### Step 1: Check CUDA Compatibility

```cmd
nvidia-smi
```

If this command works, note your CUDA version.

### Step 2: Install CUDA-enabled PyTorch

For CUDA 11.8:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify GPU Support

```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Should show: `CUDA available: True`

---

## Troubleshooting

### Issue: "Python is not recognized"
**Solution**: Add Python to PATH
1. Search for "Environment Variables" in Windows
2. Edit "Path" variable
3. Add Python installation directory (e.g., `C:\Python310\`)
4. Restart Command Prompt

### Issue: "pip is not recognized"
**Solution**: Reinstall Python with "Add to PATH" option checked

### Issue: "ModuleNotFoundError"
**Solution**: Ensure virtual environment is activated and dependencies are installed
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU mode
```cmd
python train.py --dataset dataset --batch-size 16
```

### Issue: "Cannot find image file"
**Solution**: Use absolute paths or ensure you're in the correct directory
```cmd
cd C:\E.D.I.T.H.
python example.py --image C:\full\path\to\image.jpg
```

### Issue: Slow processing
**Solution**: 
1. Install CUDA-enabled PyTorch (see GPU Acceleration section)
2. Ensure antivirus isn't blocking Python
3. Close other heavy applications

---

## Directory Structure

After installation, your directory should look like:

```
E.D.I.T.H.\
├── venv\                          # Virtual environment
├── vehicle_identification\        # Main package
│   ├── modules\                  # Pipeline modules
│   ├── configs\                  # Configuration files
│   ├── utils\                    # Utility functions
│   └── data\                     # Database storage
├── dataset\                       # Your training dataset (optional)
├── models\                        # Trained model weights (created during training)
├── example.py                     # Example usage script
├── train.py                       # Training script
├── test_pipeline.py               # Test script
├── run.bat                        # Windows batch file
├── requirements.txt               # Dependencies
├── README.md                      # Main documentation
├── WINDOWS_INSTALL.md            # This file
└── TRAINING.md                    # Training guide
```

---

## Additional Configuration

### Custom Configuration File

Create a custom config (e.g., `my_config.yaml`):

```yaml
detection:
  model: "yolov8n"
  confidence_threshold: 0.3
  device: "cuda"  # or "cpu"

classification:
  model_name: "resnet50"
  num_classes: 196  # Adjust based on your dataset
  top_k: 5
  device: "cuda"  # or "cpu"

fusion:
  weights:
    classification: 0.4
    logo: 0.2
    ocr: 0.1
    embedding: 0.3
```

Use it:
```cmd
python example.py --image car.jpg --config my_config.yaml
```

---

## Next Steps

1. ✅ Verify installation with `python verify_installation.py`
2. ✅ Run test pipeline with `python test_pipeline.py`
3. ✅ Try with your own image: `python example.py --image your_image.jpg`
4. ✅ (Optional) Train on your dataset: `python train.py --dataset dataset`
5. ✅ Read [TRAINING.md](TRAINING.md) for advanced training options
6. ✅ Read [API.md](API.md) for programmatic usage

---

## Getting Help

- **GitHub Issues**: [https://github.com/boss0007000/E.D.I.T.H./issues](https://github.com/boss0007000/E.D.I.T.H./issues)
- **Documentation**: See README.md, API.md, and TRAINING.md
- **Examples**: Check example.py for usage patterns

---

## Performance Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for 5x speedup
2. **Batch Processing**: Process multiple images together
3. **Adjust Thresholds**: Lower confidence thresholds in config for faster processing
4. **Use SSD Storage**: Training and processing are I/O intensive

---

**E.D.I.T.H.** - Enhanced Deep Intelligence for Tactical Handling 🚗🔍

Happy Vehicle Identifying! 🎉
