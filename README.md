# Pedestrian Counting App

![Demo](demo.gif)

This GitHub repository contains a Python project for counting people on escalators, both going up and down, using video input. The project also supports real-time tracking from a camera feed. One of its unique features is the ability to apply a mask to optimize the YOLO model and only detect objects within the specified region.

## Features

- Count people on escalators going up and down in video footage.
- Real-time object tracking from a camera feed.
- Apply a mask to limit object detection to a specific region of interest.
- User-friendly GUI for selecting video sources and drawing counting lines.

## Masking & Tracking

https://github.com/Kareem411/AI_People_counter/assets/65580300/d05d3a5d-7e12-4504-85bc-43a058cd50c3


https://github.com/Kareem411/AI_People_counter/assets/65580300/b47f5e29-80a3-456b-8b09-4ff592ecc00b



## Installation

To use this application, follow these steps:

1. Clone this GitHub repository:

   ```bash
   git clone https://github.com/yourusername/pedestrian-counting-app.git
   ```

2. Navigate to the project directory:

   ```bash
   cd pedestrian-counting-app
   ```

3. Install the required Python packages using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Make sure that the `Pedestrian-Counting-App.py` and `Video_Processing_Functions.py` files are in the same directory.

2. Run the application using the following command:

   ```bash
   python Pedestrian-Counting-App.py
   ```

3. The application will prompt you to enter the video URL or select a video file.

4. Draw counting lines on the video frame to define the region of interest (ROI).

5. Click the "Continue" button to proceed to the counting lines configuration window.

6. Configure counting lines for both up and down directions.

7. Click the "Start Processing" button to begin counting people.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project utilizes the [YOLOv8](https://github.com/ultralytics/ultralytics) model for object detection.
- The object tracking component is powered by [DeepSort](https://github.com/ultralytics/deepsort).
- Special thanks to the open-source community for their contributions to computer vision and object tracking.

## Contact

If you have any questions or suggestions, please feel free to [contact me](zadkareem@gmail.com).
