from keras.models import load_model  
import cv2  
import numpy as np

np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
classNames = open("labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Print the class name, confidence score to screen output
    try:
        image = cv2.rectangle(image,(0,0),(90, 60),(0,0,0),-1)
        image = cv2.putText(image, className[2:], (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.4, (125, 246, 55), 1)
        image = cv2.putText(image, str(np.round(confidenceScore * 100))[:-2], (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.4, (125, 246, 55), 1)
    except:
        pass


    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict using  the model
    prediction = model.predict(image)
    index = np.argmax(prediction)

    # Get className
    className = classNames[index]

    # Get confidence score
    confidenceScore = prediction[0][index]
    
    keyboardInput = cv2.waitKey(1)

    if keyboardInput == 27:
        break

camera.release()
cv2.destroyAllWindows()