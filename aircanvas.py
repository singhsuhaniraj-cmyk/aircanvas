from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

app = Flask(__name__)

# Drawing points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

color_index = 0


paintWindow = np.ones((720, 1280, 3), dtype=np.uint8) * 255

mpHands = mp.solutions.hands
hands = mpHands.Hands( max_num_hands=1,  min_detection_confidence=0.7)

camera = cv2.VideoCapture(0)


def generate_frames():
    global blue_index
    global green_index
    global red_index
    global yellow_index
    global color_index

    global bpoints
    global gpoints
    global rpoints
    global ypoints
    global paintWindow

    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(framergb)

        paintWindow = np.ones((720, 1280, 3), dtype=np.uint8) * 255

        cv2.rectangle(paintWindow, (40, 1),(140, 65), (0, 0, 0), 2)
        cv2.rectangle(paintWindow, (160, 1),(255, 65), (255, 0, 0), 2)
        cv2.rectangle(paintWindow, (275, 1),(370, 65), (0, 255, 0), 2)
        cv2.rectangle(paintWindow, (390, 1),(485, 65), (0, 0, 255), 2)
        cv2.rectangle(paintWindow, (505, 1),(600, 65), (0, 255, 255), 2)

        cv2.putText(paintWindow, "CLEAR",(49, 33), cv2.FONT_HERSHEY_SIMPLEX,  0.5,(0, 0, 0),2)
        cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 0, 0),2)
        cv2.putText(paintWindow, "GREEN",(298, 33),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
        cv2.putText(paintWindow, "RED",(420, 33),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(paintWindow, "YELLOW",(520, 33),cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 2)

        if result.multi_hand_landmarks:

            for handslms in result.multi_hand_landmarks:

                landmarks = []

                for lm in handslms.landmark:

                    lmx = int(lm.x * 1280)
                    lmy = int(lm.y * 720)

                    landmarks.append([lmx, lmy])

                # Index finger tip
                fore_finger = landmarks[8]
                center = fore_finger

                # Thumb tip
                thumb = landmarks[4]

                # Stop drawing gesture
                if (thumb[1] - center[1] < 30):

                    bpoints.append(deque(maxlen=512))
                    blue_index += 1

                    gpoints.append(deque(maxlen=512))
                    green_index += 1

                    rpoints.append(deque(maxlen=512))
                    red_index += 1

                    ypoints.append(deque(maxlen=512))
                    yellow_index += 1

                elif center[1] <= 65:

                    # Clear
                    if 40 <= center[0] <= 140:

                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0

                    # Blue
                    elif 160 <= center[0] <= 255:
                        color_index = 0

                    # Green
                    elif 275 <= center[0] <= 370:
                        color_index = 1

                    # Red
                    elif 390 <= center[0] <= 485:
                        color_index = 2

                    # Yellow
                    elif 505 <= center[0] <= 600:
                        color_index = 3

                else:

                    if color_index == 0:
                        bpoints[blue_index].appendleft(center)

                    elif color_index == 1:
                        gpoints[green_index].appendleft(center)

                    elif color_index == 2:
                        rpoints[red_index].appendleft(center)

                    elif color_index == 3:
                        ypoints[yellow_index].appendleft(center)

        # Draw all points
        points = [bpoints, gpoints, rpoints, ypoints]

        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):

                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue

                    cv2.line(paintWindow,points[i][j][k - 1],points[i][j][k],  colors[i], 5 )

        ret, buffer = cv2.imencode('.jpg', paintWindow)

        paintWindow_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            paintWindow_bytes +
            b'\r\n'
        )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/canvas.html')
def canvas():
    return render_template('canvas.html')


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(debug=True)