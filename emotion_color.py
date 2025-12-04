import cv2
from deepface import DeepFace
import numpy as np

# Boje za emocije (BGR format)
emotion_colors = {
    "happy": (0, 255, 255),      # žuta
    "sad": (255, 0, 0),          # plava
    "angry": (0, 0, 255),        # crvena
    "surprise": (0, 165, 255),   # narandžasta
    "neutral": (200, 200, 200),  # siva
    "fear": (128, 0, 128),       # ljubičasta
    "disgust": (0, 128, 0)       # tamno zelena
}

# Otvori kameru
cap = cv2.VideoCapture(0)

print("Pritisnite 'q' da zatvorite aplikaciju")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analiza emocije
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result['dominant_emotion']

        # Odredi boju za overlay
        color = emotion_colors.get(dominant_emotion, (255, 255, 255))

        # Kreiraj overlay iste veličine kao frame
        overlay = np.full(frame.shape, color, dtype=np.uint8)

        # Kombinuj overlay i originalni frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Prikazi dominantnu emociju
        cv2.putText(frame, f'Emotion: {dominant_emotion}', (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    except:
        # Ako nema lice ili greška, samo prikaži original
        pass

    cv2.imshow("Emotion Color Video", frame)

    # Kraj programa na 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
