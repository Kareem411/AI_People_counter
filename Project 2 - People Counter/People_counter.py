import sys
import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cvzone
from sort import *

app = tk.Tk()

# Load a frame from the video
video_url = "C:\\Users\\zadka\\Desktop\\Comp Vision Course\\REQ\\Project 2 - People Counter\\Videos\\people.mp4"
video_cap = cv2.VideoCapture(video_url)
ret, frame = video_cap.read()
video_cap = None
lines = []
color_map = {
    "blue": (255, 0, 0),  # BGR format
    "red": (0, 0, 255)}
class_colors = {}

model = YOLO("yolov8l.pt")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


# Convert the frame to a PhotoImage for the background image
if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = frame_rgb.shape
    pil_image = Image.fromarray(frame_rgb)
    photo_image = ImageTk.PhotoImage(pil_image)

    canvas = tk.Canvas(app, width=width, height=height)
    canvas.create_image(0, 0, image=photo_image, anchor=tk.NW)
    canvas.pack()

    # Create variables to track drawing state and points
    drawing = False
    start_point = None
    end_point = None
    current_line_color = "blue"  # Initialize color to blue
    lines = []  # Stores the drawn lines
    line_limit = 2  # Maximum number of lines


    def mouse_press(event):
        global drawing, start_point, end_point, current_line_color
        if len(lines) < line_limit:
            drawing = True
            start_point = (event.x, event.y)
            end_point = (event.x, event.y)

            # Set the line color based on line count
            if len(lines) == 0:
                current_line_color = "blue"
            else:
                current_line_color = "red"

            paint_canvas()


    def mouse_move(event):
        global drawing, end_point, current_line_color
        if drawing:
            end_point = (event.x, event.y)

            # Adjust the color while drawing
            if len(lines) == 0:
                current_line_color = "blue"
            else:
                current_line_color = "red"

            paint_canvas()


    def mouse_release(event):
        global drawing, start_point, end_point, current_line_color
        if drawing and len(lines) < line_limit:
            drawing = False
            end_point = (event.x, event.y)
            lines.append((start_point, end_point, current_line_color))
            paint_canvas()

            # Print the locations of the lines
            print("Line Locations:", lines)


    def play_video():
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            photo_image = ImageTk.PhotoImage(pil_image)
            canvas.create_image(0, 0, image=photo_image, anchor=tk.NW)
            canvas.update()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' key is pressed
                break

        video_cap.release()
        cv2.destroyAllWindows()


    def done_button_clicked():
        global video_cap, lines, class_colors  # Reference the global variables
        app.destroy()  # Close the Tkinter window
        if video_cap:
            video_cap.release()  # Release the video capture
        cv2.destroyAllWindows()  # Close any remaining cv2 windows

        # Load the video again for processing the mask
        video_cap = cv2.VideoCapture(video_url)

        # Define a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw the region between the blue and red lines on the mask
        for line_start, line_end, line_color in lines:
            if line_color == "blue":
                blue_line = line_start, line_end
            elif line_color == "red":
                red_line = line_start, line_end

        # Calculate the region for the mask
        mask_region = np.array([
            [blue_line[0][0], blue_line[0][1]],
            [blue_line[1][0], blue_line[1][1]],
            [red_line[1][0], red_line[1][1]],
            [red_line[0][0], red_line[0][1]]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [mask_region], 255)

        # Apply the mask to each frame
        while True:
            success, img = video_cap.read()
            if not success:
                break

            imgRegion = cv2.bitwise_and(img, img, mask=mask)

            results = model(imgRegion, stream=True)

            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    w, h = xmax - xmin, ymax - ymin

                    # Confidence
                    conf = round(float(box.conf[0]), 2)
                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = model.names[cls]

                    if currentClass == "person" and conf > 0.3:
                        # Generate and store color for each class
                        if cls not in class_colors:
                            class_colors[cls] = tuple(np.random.randint(0, 255, 3).tolist())
                        color = class_colors[cls]

                        currentArray = np.array([xmin, ymin, xmax, ymax, conf])
                        detections = np.vstack((detections, currentArray))

                        resultsTracker = tracker.update(detections)

                        for result in resultsTracker:
                            x1, x2, y1,y2, id = map(int, result)
                            cvzone.cornerRect(img, (xmin, ymin, w, h), l=9, rt=5,colorC=color)
                            cvzone.putTextRect(img, f'{id}', (max(0, xmin), max(35, ymin)),
                                               scale=0.6, thickness=1, offset=3)
            cv2.imshow("Output", img)
            cv2.waitKey(1)

        video_cap.release()
        cv2.destroyAllWindows()


    def paint_canvas():
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo_image, anchor=tk.NW)

        # Redraw all lines
        for line_start, line_end, line_color in lines:
            canvas.create_line(line_start, line_end, fill=line_color, width=2)

        # Draw the current line
        if drawing:
            canvas.create_line(start_point, end_point, fill=current_line_color, width=2)

        # Add the legend
        canvas.create_text(width - 10, 10, anchor=tk.NE, text="Start counting line", fill="blue", font=("Helvetica", 16, "bold"))
        canvas.create_text(width - 10, 30, anchor=tk.NE, text="Stop counting line", fill="red", font=("Helvetica", 16, "bold"))


    done_button = tk.Button(app, text="Done", command=done_button_clicked)
    done_button.pack()
    canvas.bind("<ButtonPress-1>", mouse_press)
    canvas.bind("<B1-Motion>", mouse_move)
    canvas.bind("<ButtonRelease-1>", mouse_release)

    app.mainloop()

sys.exit(0)
