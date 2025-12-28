import cv2 as cv
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def draw_contour_circle(frame, contour):
    (x, y, w, h) = cv.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.fillPoly(mask, [contour], 255)
    mean_color = cv.mean(frame, mask=mask)[:3]
    color = tuple([int(c) for c in mean_color])
    cv.circle(frame, (center_x, center_y), int((w + h) / 3), color, 2)

def draw_contour_rectangle(frame, contour):
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def check_direction_change(contours, prev_center_x, prev_center_y):
    for contour in contours:
        if cv.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        if prev_center_x is None or prev_center_y is None:
            prev_center_x = center_x
            prev_center_y = center_y
        else:
            dx = center_x - prev_center_x
            dy = center_y - prev_center_y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if abs(angle) > 45:
                print("Direction of motion changed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        prev_center_x = center_x
        prev_center_y = center_y
    return prev_center_x, prev_center_y

def calculate_motion_timeline(motion_start_times):
    time_intervals = []
    current_time = motion_start_times[0]
    count = 1
    for time in motion_start_times[1:]:
        if (time - current_time).total_seconds() >= 1:
            time_intervals.append((current_time, count))
            current_time = time
            count = 1
        else:
            count += 1
    time_intervals.append((current_time, count))
    return time_intervals

def plot_motion_timeline(time_intervals):
    if time_intervals:
        motion_timeline_dates = [interval[0] for interval in time_intervals]
        motion_timeline_counts = [interval[1] for interval in time_intervals]
        plt.figure(figsize=(10, 6))
        plt.plot(motion_timeline_dates, motion_timeline_counts, marker='o', linestyle='-', color='b')
        plt.title('Motion Detection Timeline')
        plt.xlabel('Time')
        plt.ylabel('Motion Count')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.tight_layout()
        plt.show()

def main():
    cap = cv.VideoCapture('bb.mp4')
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    if frame1 is None or frame2 is None:
        print("Error: Could not read frames from the video file.")
        cap.release()
        return
    if frame1.shape != frame2.shape:
        print("Error: Frames dimensions do not match.")
        cap.release()
        return
    motion_count = 0
    motion_start_times = []
    motion_dates = []
    video_speed = 20.0
    prev_center_x = None
    prev_center_y = None
    while cap.isOpened():
        frame_start_time = datetime.now()
        diff = cv.absdiff(frame1, frame2)
        gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        blurred_diff = cv.GaussianBlur(gray_diff, (5, 5), 0)
        _, thresh = cv.threshold(blurred_diff, 20, 255, cv.THRESH_BINARY)
        morph = cv.morphologyEx(thresh, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv.findContours(morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        prev_center_x, prev_center_y = check_direction_change(contours, prev_center_x, prev_center_y)
        for contour in contours:
            if cv.contourArea(contour) < 500:
                continue
            draw_contour_rectangle(frame1, contour)
            draw_contour_circle(frame1, contour)
            motion_count += 1
            if motion_count == 1:
                motion_start_time = datetime.now()
                motion_start_times.append(motion_start_time)
                motion_dates.append(motion_start_time.strftime("%Y-%m-%d %H:%M:%S"))
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(frame1, f'Motion Count: {motion_count}', (10, 50), font, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow('Motion Detection', frame1)
        key = cv.waitKey(int(1000 / video_speed)) & 0xFF
        if key == ord('q'):
            break
        frame1 = frame2
        _, frame2 = cap.read()
        if frame2 is None:
            break
    if motion_dates:
        print("Start time of first motion:", motion_dates[0])
    else:
        print("No motions detected.")
    with open('motion_log.txt', 'w') as f:
        for date in motion_dates:
            f.write(date + '\n')
    print("Total motions:", motion_count)
    time_intervals = calculate_motion_timeline(motion_start_times)
    plot_motion_timeline(time_intervals)
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
