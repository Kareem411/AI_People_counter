import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

params = {
    "max_iou_distance": 0.7,
    "max_age": 20,
    "n_init": 1,
    "nms_max_overlap": 0.1,
    "max_cosine_distance": 0.2,
    "nn_budget": None,
    "gating_only_position": True,
    "embedder": "mobilenet",
    "half": False,
    "bgr": False,
    "embedder_gpu": True,
    "polygon": False,
}
tracker = DeepSort(**params)


def tracking(detections, frame):
    # detections expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class
    tracks = tracker.update_tracks(detections, frame=frame)
    det = {int(track.track_id): [int(x) for x in track.to_ltrb()] for track in tracks if track.is_confirmed()}
    return det


def process_frame(
    img,
    tracker_detections,
    limitsUp,
    limitsDown,
    people_crossed_up,
    people_crossed_down,
):
    offset = 15
    # Calculating the values outside the loop
    limitsUp_y_minus_15 = limitsUp[1] - offset
    limitsUp_y_plus_15 = limitsUp[1] + offset
    limitsDown_y_minus_15 = limitsDown[1] - offset
    limitsDown_y_plus_15 = limitsDown[1] + offset

    # Creating a list to store draw commands for improved performance
    draw_commands = []

    for tracker_id, tracker_bbox in tracker_detections.items():
        x, y, x_max, y_max = tracker_bbox
        cx, cy = x + (x_max - x) // 2, y + (y_max - y) // 2

        # Adding a rectangle to draw commands
        draw_commands.append((cv2.rectangle, (img, (x, y), (x_max, y_max), (0, 0, 200), 3)))

        # Adding text to draw commands
        text_x, text_y = max(0, x), max(35, y - 5)
        draw_commands.append((cv2.putText, (img, f"Person ID: {tracker_id}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2, cv2.LINE_AA)))

        # Adding circle to draw commands
        draw_commands.append((cv2.circle, (img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)))

        if limitsUp[0] < cx < limitsUp[2] and limitsUp_y_minus_15 < cy < limitsUp_y_plus_15:
            if tracker_id not in people_crossed_up:
                people_crossed_up.add(tracker_id)
                draw_commands.append((cv2.line, (img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)))

        if limitsDown[0] < cx < limitsDown[2] and limitsDown_y_minus_15 < cy < limitsDown_y_plus_15:
            if tracker_id not in people_crossed_down:
                people_crossed_down.add(tracker_id)
                draw_commands.append((cv2.line, (img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)))

    # Executing all draw commands
    for cmd, args in draw_commands:
        cmd(*args)

    # Adding the text for people crossed
    cv2.putText(
        img,
        str(len(people_crossed_up)),
        (limitsUp[0], limitsUp[1] - 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 255, 0),
        3,
    )
    cv2.putText(
        img,
        str(len(people_crossed_down)),
        (limitsDown[0], limitsDown[1] - 50),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 255, 0),
        3,
    )

    cv2.imshow("Output", img)
    cv2.waitKey(1)



def calculate_mask_and_process_video(
    model, lines, video_url, height, width, limitsUp, limitsDown
):
    # Load the video again for processing the mask
    video_cap = cv2.VideoCapture(video_url)
    # Initializing the variable with an empty dictionary
    tracker_detections = {}
    people_crossed_up = set()
    people_crossed_down = set()
    # Define a blank mask
    mask = np.zeros((height, width), dtype=np.uint8)

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

    # Apply the mask to each frame
    while True:
        success, img = video_cap.read()
        if not success:
            break

        imgRegion = cv2.bitwise_and(img, img, mask=mask)

        results = model.track(source=imgRegion, stream=False, show=False)
        del imgRegion
        for r in results:
            detections = []
            for box in r.boxes:
                conf = round(float(box.conf[0]), 2)
                if conf > 0.3 and (
                    int(box.cls[0]) == 0
                ):  # Class ID 0 represents "person"
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    detection = (
                        [xmin, ymin, xmax - xmin, ymax - ymin],
                        conf,
                        "Person",
                    )
                    detections.append(detection)
            # Using the tracker function
            tracker_detections = tracking(detections, img)
        # Using the frame processing function
        process_frame(
            img,
            tracker_detections,
            limitsUp,
            limitsDown,
            people_crossed_up,
            people_crossed_down,
        )
    video_cap.release()
    cv2.destroyAllWindows()