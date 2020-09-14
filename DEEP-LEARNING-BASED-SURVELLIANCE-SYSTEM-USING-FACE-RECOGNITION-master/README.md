# DEEP-LEARNING-BASED-SURVELLIANCE-SYSTEM-USING-FACE-RECOGNITION
NOTE:- Maintain Proper Readme File for setting up environment and testing purpose.
Use python 3.5 or upper
Install following libraries tensorflow==1.12.0, scikit-learn==0.19.1, scipy==0.17.0, Pillow==7.0.0, opencv-python==4.0.0.21, numpy==1.16.1
create a folder named class and store the classifier.pkl in it, create a folder named npy and store det1.npy, det2.npy, det3.npy in it.
Run Register.py to click images to train the model, create folder named pre_img to store captured images.
Run train_main.py to train the model with captured images.
Now run identify_face_video.py to identify the person in the live video stream.
Model cannot be uploaded due to limited space, You can use google's facenet model, rename the model to facenet_model.pb and store it in folder named model. The following code will not sun without the model make sure you have the model.
