import sys
import tkinter as tk
from tkinter import simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Importing created functions
from Video_Processing_Functions import calculate_mask_and_process_video


while True:
    video_url = simpledialog.askstring("Video URL", "Enter the video URL:")
    if video_url is None:
        sys.exit(0)  # Exit the application if the user cancels
    video_cap = cv2.VideoCapture(video_url)
    if video_cap.isOpened():
        video_cap.release()  # Closing the video capture
        break  # Exit the loop if the provided URL is a valid video file
    else:
        simpledialog.messagebox.showerror(
            "Invalid URL",
            "The provided URL is not a valid video file path. Please try again.",
        )


app = tk.Tk()

video_cap = cv2.VideoCapture(video_url)
ret, frame = video_cap.read()
video_cap = None
second_window = None
drawing_second = False
start_point_second = None
end_point_second = None
current_line_color_second = None
current_line_second = None
lines_second = []
counting_line_limitsUp = None
counting_line_limitsDown = None
color_map = {"blue": (255, 0, 0), "red": (0, 0, 255)}


# Convert the frame to a PhotoImage for the background image
if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_height, video_width, channels = frame_rgb.shape
    pil_image = Image.fromarray(frame_rgb)
    photo_image = ImageTk.PhotoImage(pil_image)

    canvas = tk.Canvas(app, width=video_width, height=video_height)
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

            # Set the line color based on line_count
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
                current_line_color_second = "red"
            else:
                current_line_color_second = "blue"

            current_line_second = None  # Clear the current line
            paint_canvas_2()  # Update the canvas after starting a new line

    def mouse_move(event):
        global drawing, end_point, current_line_color
        if drawing:  # Leaving incase a new feature of adding lines is added
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
            current_line_second = (
                start_point_second,
                end_point_second,
                current_line_color_second,
            )
            canvas_2.create_line(
                start_point_second,
                end_point_second,
                fill=current_line_color_second,
                width=2,
                tags="temp_line",
            )

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
        global drawing_second, current_line_second, lines_second, end_point_second, counting_line_limitsUp, counting_line_limitsDown
        if drawing_second and current_line_second:
            drawing_second = False
            end_point_second = (end_point_second[0], end_point_second[1])
            lines_second.append(current_line_second)
            paint_canvas_2()  # Update the canvas after adding the line

            # Assign values of the first line in lines_second to counting_line_limitsUp
            if len(lines_second) >= 1:
                counting_line_limitsUp = [
                    lines_second[0][0][0],
                    lines_second[0][0][1],
                    lines_second[0][1][0],
                    lines_second[0][1][1],
                ]

            # Assign values of the second line in lines_second to counting_line_limitsDown
            if len(lines_second) >= 2:
                counting_line_limitsDown = [
                    lines_second[1][0][0],
                    lines_second[1][0][1],
                    lines_second[1][1][0],
                    lines_second[1][1][1],
                ]


    def process__video_button():
        global second_window, canvas_2, lines_second, counting_line_limitsUp, counting_line_limitsDown
        if second_window:
            second_window.destroy()  # Close the second window if it exists

        detection_model = YOLO("yolov8s.pt")
        # Calculate the mask and process the video
        calculate_mask_and_process_video(
            detection_model,
            lines,
            video_url,
            video_height,
            video_width,
            counting_line_limitsUp,
            counting_line_limitsDown,
        )  # Start processing the video


    def to_counting_window_btn_clicked():
        global second_window, canvas_2, lines_second
        app.withdraw()
        if video_cap:
            video_cap.release()
        cv2.destroyAllWindows()
        # Create a new window to display the mask
        mask_display_window = tk.Toplevel()
        mask_display_window.title("Mask Review")

        # Define a blank mask
        mask = np.zeros((video_height, video_width), dtype=np.uint8)

        # Draw the region between the blue and red lines on the mask
        for line_start, line_end, line_color in lines:
            if line_color == "blue":
                blue_line = line_start, line_end
            elif line_color == "red":
                red_line = line_start, line_end

        # Calculate the region for the mask
        mask_region = np.array(
            [
                [blue_line[0][0], blue_line[0][1]],
                [blue_line[1][0], blue_line[1][1]],
                [red_line[1][0], red_line[1][1]],
                [red_line[0][0], red_line[0][1]],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [mask_region], 255)

        # Apply the mask to the first frame to show the mask overlay
        mask_frame = cv2.bitwise_and(frame.copy(), frame.copy(), mask=mask)

        # Convert the frame to a PhotoImage for display in the tkinter window
        pil_mask_frame = Image.fromarray(mask_frame)
        mask_photo = ImageTk.PhotoImage(pil_mask_frame)

        mask_canvas = tk.Canvas(
            mask_display_window, width=video_width, height=video_height
        )
        mask_canvas.create_image(0, 0, image=mask_photo, anchor=tk.NW)
        mask_canvas.pack()

        # Ask the user whether to continue or redo
        user_response = tk.messagebox.askquestion("Mask Review", "Do you want to continue with this mask?", icon="question")
        if user_response and user_response.lower() == "yes":
            # Continue with processing or any other logic you want
            print("Continuing with the mask.")
            mask_display_window.destroy()  # Close the mask display window
            if not second_window:
                second_window = tk.Toplevel()
                second_window.title("Counting Lines Window")

                canvas_2 = tk.Canvas(second_window, width=video_width, height=video_height)
                canvas_2.create_image(0, 0, image=photo_image, anchor=tk.NW)
                canvas_2.pack()

                Counting_window_label = tk.Label(
                    second_window,
                    text="Mark a line indicating where to tally pedestrians.",
                    font=("Helvetica", 12),
                )
                Counting_window_label.pack()

                Counting_window_button = tk.Button(
                    second_window,
                    text="Start Processing",
                    command=process__video_button,
                    cursor="hand2",
                )
                Counting_window_button.pack()

                canvas_2.bind("<ButtonPress-1>", mouse_press_second)
                canvas_2.bind("<B1-Motion>", mouse_move_second)
                canvas_2.bind("<ButtonRelease-1>", mouse_release_second)

                paint_canvas_2()  # Update the canvas initially
        else:
            print("Exiting the script...")
            mask_display_window.destroy()  # Close the mask display window
            sys.exit()

        

    def paint_canvas_2():
        canvas_2.delete("all")
        canvas_2.create_image(0, 0, image=photo_image, anchor=tk.NW)
        for line_start, line_end, line_color in lines_second:
            canvas_2.create_line(line_start, line_end, fill=line_color, width=2)

        if drawing_second:
            canvas_2.create_line(
                start_point_second[0],
                start_point_second[1],
                end_point_second[0],
                end_point_second[1],
                fill=current_line_color_second,
                width=2,
            )

        canvas_2.create_text(
            video_width - 10,
            10,
            anchor=tk.NE,
            text="Counting line 1",
            fill="blue",
            font=("Helvetica", 16, "bold"),
        )
        canvas_2.create_text(
            video_width - 10,
            30,
            anchor=tk.NE,
            text="Counting Line 2",
            fill="red",
            font=("Helvetica", 16, "bold"),
        )

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
        canvas.create_text(
            video_width - 10,
            10,
            anchor=tk.NE,
            text="Mask Start line",
            fill="blue",
            font=("Helvetica", 16, "bold"),
        )
        canvas.create_text(
            video_width - 10,
            30,
            anchor=tk.NE,
            text="Mask End line",
            fill="red",
            font=("Helvetica", 16, "bold"),
        )

    to_counting_window_btn = tk.Button(
        app,
        text="Continue",
        command=to_counting_window_btn_clicked,
        relief=tk.RAISED,
        borderwidth=2,
        cursor="hand2",
    )
    to_counting_window_btn.pack()
    canvas.bind("<ButtonPress-1>", mouse_press)
    canvas.bind("<B1-Motion>", mouse_move)
    canvas.bind("<ButtonRelease-1>", mouse_release)

    app.mainloop()

sys.exit(0)