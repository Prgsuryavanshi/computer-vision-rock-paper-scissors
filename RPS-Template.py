import time
import cv2
from keras.models import load_model
import numpy as np
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
i = 500
start = time.time()
while True: 
    ret, frame = cap.read()
    resized_frame2 = cv2.resize(frame, (960, 720), interpolation = cv2.INTER_AREA)
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image
    prediction = model.predict(data)
    font = cv2.FONT_HERSHEY_SIMPLEX
    resized_frame2 = cv2.putText(resized_frame2, f"Countdown is {10-int(time.time() - start)}", (200, 300), font, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
    # Press q to close the window
    cv2.imshow('frame', resized_frame2)
    
    print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(int(cap.get(3)))
print(int(cap.get(4)))         

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
