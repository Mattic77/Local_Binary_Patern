# Local Binary Pattern (LBP) Implementation

A real-time computer vision application that implements three variants of Local Binary Pattern algorithms using OpenCV and NumPy.

## Features

This project implements three LBP algorithms:

1. **LBP (Local Binary Pattern)** - Classic LBP that compares 8 neighbors with the center pixel
2. **Mean LBP** - Compares neighbors with the mean of all neighbors instead of center pixel
3. **XCS-LBP (eXtended Center-Symmetric LBP)** - Compares opposite pairs of neighbors

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Mattic77/Local_Binary_Patern
cd "Local Binary Patern"
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install opencv-python numpy
```

## Usage

Run the application:

```bash
python app.py
```

### Controls

- **1** - Switch to LBP (Local Binary Pattern)
- **2** - Switch to Mean LBP
- **3** - Switch to XCS-LBP
- **s** - Save current frame
- **q** - Quit application

## How It Works

The application captures video from your default camera and applies the selected LBP algorithm in real-time. Three windows are displayed:

- Original frame
- Grayscale conversion
- LBP result (normalized for visualization)

## Applications

Local Binary Patterns are widely used in:

- Face recognition
- Texture classification
- Object detection
- Medical image analysis
- Feature extraction

## License

MIT License
