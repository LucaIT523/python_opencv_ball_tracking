# 



<div align="center">
   <h1>python_opencv_ball_tracking</h1>
</div>



This Python script implements a multi-object tracking system for colored balls in video footage, using OpenCV for computer vision operations. 



<div align="center">
   <img src=https://github.com/LucaIT523/python_opencv_ball_tracking/blob/main/images/1.png>
</div>



### 1. Core Architecture

**A. Tracking Pipeline**

```
Video Input → Frame Processing → Color Segmentation → Contour Detection → 
→ Object Tracking → Path Visualization → Output Rendering
```

**B. Key Components**

- **CBall Class**: Manages tracking state for individual balls (position, color, trajectory)
- **Color Thresholding**: HSV-based detection for red, yellow, white balls
- **Dynamic ROI**: Automatic play area detection (green surface)

### 2. Technical Implementation

**A. Object Detection**

```
# Color segmentation example
red1 = cv2.inRange(HSVImg, red1_light, red1_dark)
red2 = cv2.inRange(HSVImg, red2_light, red2_dark)
red = cv2.bitwise_or(red1, red2)
```

- Dual-range HSV thresholding for robust color detection
- Adaptive brightness compensation based on frame analysis

**B. Tracking Logic**

```
class CBall:
    def setPos(self, x, y, r, area):
        # Implements velocity-based position updating
        distance = sqrt(pow(x-self.x, 2) + pow(y-self.y, 2))
        if distance > self.distanceThreshold:
            self.update_position(x,y)
```

- Motion prediction with distance thresholds (5px default)
- Inertia modeling through `countStop` mechanism
- Trajectory smoothing with historical position averaging

**C. Adaptive Processing**

```
# Dynamic parameter adjustment
w_RateX = (g_EndX - g_StartX)/WIDTH
AREA = AREA * w_Rate
SIZEMIN = SIZEMIN * w_RateX
```

- Automatic scaling of detection parameters based on ROI size
- Brightness-adaptive color thresholds

### 3. Key Features

**A. Multi-Object Handling**

- Independent tracking for 3 colored balls
- Collision-resistant position verification

```
if len(w_Regions) > 1:
    # Select closest match to previous position
    temp_sum = abs(x_diff) + abs(y_diff) + abs(r_diff)
```

**B. Visualization System**

- Real-time trajectory drawing

- Diagnostic view composition:

  ```
  sqlCopy[Original Frame | Masked View]
  [Tracking Result | Processed View]
  ```

- Position markers with crosshairs and labels

**C. Performance Optimization**

- ROI-based processing (g_StartX/Y to g_EndX/Y)
- Adaptive thresholding based on scene brightness
- Frame-resized processing (640x360 default)

### 4. Operational Parameters

| **Detection Settings** | **Value Range**       |
| ---------------------- | --------------------- |
| Frame Size             | 640x360 px            |
| Processing FPS         | 30                    |
| Ball Diameter          | 10-40 px              |
| Minimum Area           | 65 px²                |
| Motion Threshold       | 5 px/frame            |
| Stop Detection         | 10 consecutive frames |

### 5. Usage Scenarios

1. **Sports Analytics**: Track ball movement in games
2. **Robotics Vision**: Object following for autonomous systems
3. **Industrial Monitoring**: Part tracking on conveyor systems
4. **Physics Education**: Motion trajectory visualization













### **Contact Us**

For any inquiries or questions, please contact us.

telegram : @topdev1012

email :  skymorning523@gmail.com

Teams :  https://teams.live.com/l/invite/FEA2FDDFSy11sfuegI