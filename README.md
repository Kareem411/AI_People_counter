# Pedestrian Counting App

![Demo](demo.gif)

This GitHub repository contains a Python project for counting people on escalators, both going up and down, using video input. The project also supports real-time tracking from a camera feed. One of its unique features is the ability to apply a mask to optimize the YOLO model and only detect objects within the specified region.

## Features

- Count people on escalators going up and down in video footage.
- Real-time object tracking from a camera feed.
- Apply a mask to limit object detection to a specific region of interest.
- User-friendly GUI for selecting video sources and drawing counting lines.

## Masking & Tracking



https://github.com/Kareem411/AI_People_counter/assets/65580300/156769c7-d883-442c-a28b-5b6e7485acdb



https://github.com/Kareem411/AI_People_counter/assets/65580300/78a70605-7269-4fe8-a853-40ff7134b1aa


## Installation

To use this application, follow these steps:

1. Clone this GitHub repository:

   ```bash
   git clone https://github.com/Kareem411/AI_People_counter.git
   ```

2. Navigate to the project directory:

   ```bash
   cd "Project - People Counter"
   ```

3. Install the required Python packages in the following order using `pip`:

   ```bash
   pip install ultralytics
   pip uninstall torchvision
   pip install --pre torchvision -f https://download.pytorch.org/whl/nightly/cu118/torch_nightly.html
   pip install deep_sort_realtime
   ```

   These commands will install the necessary packages and configure your environment for running the application.

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

If you have any questions or suggestions, please feel free to [contact me](mailto:zadkareem@gmail.com).
