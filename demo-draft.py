import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def captureAndSaveImage():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Music Suggestion Based on Emotion Recognition")
    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Music Suggestion Based on Emotion Recognition", frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            # press ESC to exit camera.
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # press SPACE to capture and save image.
            img_name = "captured-image.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("written!".format(img_name))
            img_counter += 1
    cam.release()
    cv2.destroyAllWindows()
    

captureAndSaveImage()

input_img = cv2.imread('captured-image.png')
gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 4)

# loop over all detected faces
if len(faces) > 0:
   for i, (x, y, w, h) in enumerate(faces):
 
      # To draw a rectangle in a face
      cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
      face = input_img[y:y + h, x:x + w]
    #   cv2.imshow("Cropped Face", face)
      cv2.imwrite(f'face{i}.jpg', face)
      print(f"face{i}.jpg is saved")

classifications = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
def emotion_analysis(emotions):
    y_pos = np.arange(len(classifications))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, classifications)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()

model = load_model('model.h5')
img = tf.keras.utils.load_img("face0.jpg", grayscale=True, target_size=(48, 48))
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255
custom = model.predict(x)
print('Dominant emotion:', classifications[custom[0].argmax()], '-', (max(custom[0]))*100, '%')
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48])

plt.gray()
plt.title(classifications[custom[0].argmax()])
plt.imshow(x)
plt.show()
