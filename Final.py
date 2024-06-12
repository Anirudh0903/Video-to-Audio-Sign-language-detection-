import numpy as np
import pickle
import cv2
import PySimpleGUI as sg
import mediapipe as mp
import pyttsx3
import time



model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
               19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' '}

selected_character = ""
start_time = None
selected_word = ""

# Initialize the TTS engine
engine = pyttsx3.init()

# PySimpleGUI Window Layout
layout = [
    [sg.Image(filename="", key="-IMAGE-", size=(300, 300))],
    [sg.Text("Selected Word:", font=("Helvetica", 16)), sg.Text("", size=(120, 1), key="-WORD-", font=("Helvetica", 16), justification="left")],
    [sg.Button("Clear", key="-CLEAR-", size=(8, 2)), sg.Button("Space", key="-SPACE-", size=(8, 2)),
     sg.Button("Generate Audio", key="-AUDIO-", size=(14, 2))]
]

window = sg.Window("Sign Language Interpreter", layout, resizable=True, finalize=True)

cap = cv2.VideoCapture(0)

while True:
    event, values = window.read(timeout=20)

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture a frame. Exiting...")
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x:
                continue
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            data_aux = np.asarray(data_aux).reshape(1, -1)

            prediction = model.predict(data_aux)

            predicted_label = int(prediction[0])

            if predicted_label in labels_dict:
                predicted_character = labels_dict[predicted_label]

                if selected_character == predicted_character:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 2:
                        selected_word += selected_character
                        window["-WORD-"].update(selected_word)


                        # Reset variables
                        selected_character = ""
                        start_time = None
                else:
                    selected_character = predicted_character
                    start_time = None


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f"Selected Character: {selected_character}", (10, H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)



    if event == sg.WIN_CLOSED:  # If the user closes the window
        break
    elif event == "-CLEAR-":  # Press 'Clear' button to clear the word
        selected_word = ""
        window["-WORD-"].update(selected_word)
    elif event == "-SPACE-":  # Press 'Space' button to add space
        selected_word += " "
        window["-WORD-"].update(selected_word)
    elif event == "-AUDIO-":  # Press 'Generate Audio' button to convert to audio
        if selected_word:
            # Use TTS engine to speak the selected word
            engine.say(selected_word)
            engine.runAndWait()
            print("Audio Generated")

    # Update the image element with the current frame
    window["-IMAGE-"].update(data=cv2.imencode(".png", frame)[1].tobytes())

window.close()
cap.release()
cv2.destroyAllWindows()

