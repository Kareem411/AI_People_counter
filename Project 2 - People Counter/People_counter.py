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
second_window = None
drawing_second = False
start_point_second = None
end_point_second = None
current_line_color_second = None
current_line_second = None  # Rename to avoid conflict
lines_second = []
lines = []
limitsUp = None
limitsDown = None
people_crossed_up = {}
people_crossed_down = {}
color_map = {
    "blue": (255, 0, 0),  # BGR format
    "red": (0, 0, 255)}
class_colors = {}

model = YOLO("yolov5-deepsort.pt")

# Tracking
tracker = Sort(max_age=25, min_hits=10, iou_threshold=0.5)


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
    canvas_2 = None
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


    def mouse_press_second(event):
        global drawing_second, start_point_second, end_point_second, current_line_color_second, current_line_second
        if len(lines_second) < line_limit:
            drawing_second = True
            start_point_second = (event.x, event.y)
            end_point_second = (event.x, event.y)

            # Use red for the first line and blue for subsequent lines in the second window
            if len(lines_second) == 0:
                current_line_color_second = 'red'
            else:
                current_line_color_second = 'blue'

            current_line_second = None  # Clear the current line
            paint_canvas_2()  # Update the canvas after starting a new line


    def mouse_move(event):
        global drawing, end_point, current_line_color
        if drawing:     # Leaving incase a new feature of adding lines is added
            end_point = (event.x, event.y)

            # Adjust the color while drawing
            if len(lines) == 0:
                current_line_color = "blue"
            else:
                current_line_color = "red"

            paint_canvas()


    def mouse_move_second(event):
        if drawing_second:
            global end_point_second, current_line_second
            end_point_second = (event.x, event.y)
            canvas_2.delete("temp_line")
            current_line_second = (start_point_second, end_point_second, current_line_color_second)
            canvas_2.create_line(start_point_second, end_point_second, fill=current_line_color_second, width=2,
                                 tags="temp_line")


    def mouse_release(event):
        global drawing, start_point, end_point, current_line_color
        if drawing and len(lines) < line_limit:
            drawing = False
            end_point = (event.x, event.y)
            lines.append((start_point, end_point, current_line_color))
            paint_canvas()

            # Print the locations of the lines
            print("Masking Line Locations:", lines)


    def mouse_release_second(event):
        global drawing_second, current_line_second, lines_second, end_point_second, limitsUp, limitsDown
        if drawing_second and current_line_second:
            drawing_second = False
            end_point_second = (end_point_second[0], end_point_second[1])
            lines_second.append(current_line_second)
            paint_canvas_2()  # Update the canvas after adding the line

            # Assign values of the first line in lines_second to limitsUp
            if len(lines_second) >= 1:
                limitsUp = [lines_second[0][0][0], lines_second[0][0][1], lines_second[0][1][0], lines_second[0][1][1]]

            # Assign values of the second line in lines_second to limitsDown
            if len(lines_second) >= 2:
                limitsDown = [lines_second[1][0][0], lines_second[1][0][1], lines_second[1][1][0],
                              lines_second[1][1][1]]


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


    def calculate_mask_and_process_video():
        global video_cap, lines, class_colors, limitsUp, limitsDown, people_crossed_up, people_crossed_down

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
                            x1, y1, x2, y2, id = map(int, result)
                            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=5, colorC=color)
                            cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)),
                                               scale=0.6, thickness=1, offset=5)

                            # Check if the person's center crosses the 'limitsUp' line
                            if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
                                if id not in people_crossed_up:
                                    people_crossed_up[id] = True
                                    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0),
                                             5)

                            # Check if the person's center crosses the 'limitsDown' line
                            if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
                                if id not in people_crossed_down:
                                    people_crossed_down[id] = True
                                    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]),
                                             (0, 255, 0), 5)

                        # Update the counts on the canvas
                        cv2.putText(img, str(len(people_crossed_up)), (limitsUp[0], limitsUp[1] - 50),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                        cv2.putText(img, str(len(people_crossed_down)), (limitsDown[0], limitsDown[1] - 50),
                                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.imshow("Output", img)
            cv2.waitKey(1)

        video_cap.release()
        cv2.destroyAllWindows()


    def press_me_button_clicked():
        global second_window
        if second_window:
            second_window.destroy()  # Close the second window if it exists
        calculate_mask_and_process_video()  # Continue with mask calculation and video processing


    def done_button_clicked():
        global second_window, canvas_2, lines_second
        app.withdraw()
        if video_cap:
            video_cap.release()
        cv2.destroyAllWindows()

        if not second_window:
            second_window = tk.Toplevel()
            second_window.title("Counting Lines Window")

            canvas_2 = tk.Canvas(second_window, width=width, height=height)
            canvas_2.create_image(0, 0, image=photo_image, anchor=tk.NW)
            canvas_2.pack()

            Counting_window_label = tk.Label(second_window, text="Mark a line indicating where to tally pedestrians.")
            Counting_window_label.pack()

            Counting_window_button = tk.Button(second_window, text="Continue", command=press_me_button_clicked)
            Counting_window_button.pack()

            canvas_2.bind("<ButtonPress-1>", mouse_press_second)
            canvas_2.bind("<B1-Motion>", mouse_move_second)
            canvas_2.bind("<ButtonRelease-1>", mouse_release_second)

            paint_canvas_2()  # Update the canvas initially


    def paint_canvas_2():
        canvas_2.delete("all")
        canvas_2.create_image(0, 0, image=photo_image, anchor=tk.NW)
        for line_start, line_end, line_color in lines_second:
            canvas_2.create_line(line_start, line_end, fill=line_color, width=2)

        if drawing_second:
            canvas_2.create_line(start_point_second[0], start_point_second[1], end_point_second[0], end_point_second[1],
                                 fill=current_line_color_second, width=2)


        canvas_2.create_text(width - 10, 10, anchor=tk.NE, text="Mask Start line", fill="blue",
                             font=("Helvetica", 16, "bold"))
        canvas_2.create_text(width - 10, 30, anchor=tk.NE, text="Mask End line", fill="red",
                             font=("Helvetica", 16, "bold"))


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
        canvas.create_text(width - 10, 10, anchor=tk.NE, text="Mask Start line", fill="blue", font=("Helvetica", 16, "bold"))
        canvas.create_text(width - 10, 30, anchor=tk.NE, text="Mask End line", fill="red", font=("Helvetica", 16, "bold"))


    done_button = tk.Button(app, text="Done", command=done_button_clicked)
    done_button.pack()
    canvas.bind("<ButtonPress-1>", mouse_press)
    canvas.bind("<B1-Motion>", mouse_move)
    canvas.bind("<ButtonRelease-1>", mouse_release)

    app.mainloop()

sys.exit(0)
