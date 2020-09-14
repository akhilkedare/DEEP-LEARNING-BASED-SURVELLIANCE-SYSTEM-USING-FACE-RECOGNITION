import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start = time.time()


model_path = "W:\#AAA exp 182pi\A fingerprint\models\model.h5"
model_weights_path = "W:\#AAA exp 182pi\A fingerprint\models\weights.h5"
test_path = "A fingerprint/testing/a.png"

nameee=''

def my_function12(fname):
  global nameee
  nameee += fname
  print('fingerprint recognised:'+nameee)

model = load_model(model_path)
model.load_weights(model_weights_path)

img_width, img_height = 150, 150

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  
  answer = np.argmax(result)
  if answer == 0:
    my_function12("karan")
    
  elif answer == 1:
    my_function12("ayusha")
    
  elif answer == 2:
    my_function12("shruti")
    
  elif answer == 3:
    my_function12("sujata")
    
  elif answer == 4:
   my_function12("dishant")
    
  elif answer == 5:
    my_function12("suraj")
    
  elif answer == 6:
    my_function12("nimish")
    
  elif answer == 7:
    my_function12("avinash")
    
  elif answer == 8:
    my_function12("sunita")
    
  elif answer == 9:
    my_function12("vaishnavi")
    
  elif answer == 10:
    print("Predicted: not recognise")
  return answer

predict(test_path)
os.remove("A fingerprint/testing/a.png")

