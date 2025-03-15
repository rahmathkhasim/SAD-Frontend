# Installation Guide

This guide will help you set up the required dependencies for your project.

## Prerequisites
- Ensure you have **Python 3.x** installed.
- It's recommended to use a **virtual environment** to avoid dependency conflicts.

## Installing Dependencies
Run the following command to install the necessary Python libraries:
```sh
pip install streamlit==1.39.0 \
            opencv-python==4.10.0.84 \
            numpy==1.26.2 \
            insightface==0.7.3 \
            requests==2.32.3 \
            pandas==2.2.3 \
            scikit-learn==1.6.1 \
            python-dotenv==1.0.1 \
            scipy==1.14.1 \
            pillow==10.3.0 \
            matplotlib==3.8.2 \
            seaborn==0.13.2 \
            mxnet-cu117==1.9.1
```

## Troubleshooting
If you encounter any issues, try the following:

### 1. OpenCV-related errors
If you face issues related to OpenCV, install the headless version:
```sh
pip install opencv-python-headless==4.11.0.86
```

### 2. MXNet-related errors
If MXNet installation fails, install it using:
```sh
pip install mxnet-cu117 --pre -f https://dist.mxnet.io/python
```

## Notes
- **MXNet with CUDA:** Ensure your system has a compatible **CUDA-enabled GPU** for `mxnet-cu117`.
- **Virtual Environment Usage:**
  ```sh
  python -m venv venv
  source venv/bin/activate  # On macOS/Linux
  venv\Scripts\activate  # On Windows
  ```
  Then install dependencies inside the virtual environment.

## Verification
To confirm successful installation, run:
```sh
python -c "import streamlit, cv2, numpy, insightface, requests, pandas, sklearn, dotenv, scipy, PIL, matplotlib, seaborn, mxnet; print('All dependencies installed successfully!')"
```

If you see the message **"All dependencies installed successfully!"**, the setup is complete.

---
Now you're ready to start using the project!

