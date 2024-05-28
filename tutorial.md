Traffic lights
================

In this activity, you will learn to train the model for the traffic and add it in the code.

<img src= "https://s3.amazonaws.com/media-p.slid.es/uploads/1525749/images/10563548/PCP.gif" width = "280" height = "300">

Follow the given steps to complete this activity:

1. Create a model

* Open [teachable machine](chablemachine.withgoogle.com/train).

* Create the classes for different traffic signals and train the model.

* Download the model and add it in the code.

2. Load the model in the code

*  Load the model.

    `model = load_model("keras_Model.h5", compile=False)`

* Load the labels.

    `classNames = open("labels.txt", "r").readlines()`


* Create the rectangle using `cv2.rectangle` around the loaded image.

    `try:`

        `image = cv2.rectangle(image,(0,0),(90, 60),(0,0,0),-1)`

* Make the image a numpy array and reshape it to the models input shape.

    `image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)`

* Normalize the image array.

    `image = (image / 127.5) - 1`

* Predict the images in the model.

    `prediction = model.predict(image)`
    `index = np.argmax(prediction)`

* Get the class name.

    `className = classNames[index]`

* Get confidence score.

    `confidenceScore = prediction[0][index]`

* Save and run the code to check the output.
