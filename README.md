# 🎥 Real-Time Motion Detection & Analysis (OpenCV)

This project was developed as part of my **Digital Image Processing / Computer Vision** undergraduate course. It implements a robust system to detect, track, and log motion in video streams using classical computer vision techniques.

### 🧠 Core Functionality
- **Frame Differencing**: Efficiently detects motion by calculating the absolute difference between consecutive frames.
- **Contour Analysis**: Uses `findContours` and `boundingRect` to identify and isolate moving objects.
- **Dynamic Visualization**: Draws adaptive circles and rectangles around moving targets based on their average color and size.
- **Motion Logging**: Automatically records the timestamps of detected motions into a `motion_log.txt` file for further forensic analysis.

### 🛠 Tech Stack
- **Language**: Python
- **Library**: OpenCV (cv2), NumPy, Matplotlib.
- **Techniques**: Thresholding, Gaussian Blur, Dilation, and Contour Mapping.

### 📊 How it Works
1. Applies a **Gaussian Blur** to reduce high-frequency noise.
2. Calculates the **delta frame** to find changes.
3. Applies a **Threshold** to create a binary mask of the motion.
4. Filters out small noise by setting a minimum contour area (500 pixels).
