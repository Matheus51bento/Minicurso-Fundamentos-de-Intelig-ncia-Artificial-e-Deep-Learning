import cv2
import numpy as np
from PIL import Image
# from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array



#Load the saved model
model = load_model('model.h5')
video = cv2.VideoCapture(0)

NUMBERS = [str(i) for i in range(10)]
def get_number(idx):
    return NUMBERS[idx]


while True:
    _, frame = video.read()
    
    # cv2.imshow("Capturing", frame)
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((28,28))
    # img_array = np.array(im)

    # img_array = np.reshape(img_array, (3, 28, 28))

    img = img_to_array(im)
    img = img.reshape(3, 28, 28)

    img = img.astype('float32')
    img_array = img / 255.0

    prediction = int(model.predict(img_array)[0][0])

    # if prediction is 0, which means I am missing on the image, then show the frame in gray color.
    if prediction == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            recognized = model.predict(img_array)
            # recognized = recognized.argmax(axis=1)
            # recognized = list(map(get_number, recognized)) 
            
            print(np.argmax(recognized[0]))
            # print(recognized)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()