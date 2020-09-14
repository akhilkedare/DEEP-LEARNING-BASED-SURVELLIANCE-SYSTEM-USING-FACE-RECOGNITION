from keras.models import Sequential
from keras.models import load_model
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

model = load_model("model\20170511-185253.pb")

test_generator = ImageDataGenerator()
test_data_generator = test_generator.flow_from_directory(
    'pre_img', 
	target_size=(182, 182),
    batch_size=32,
    shuffle=False)
test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_data_generator.classes
class_labels = list(test_data_generator.class_indices.keys())   

report = confusion_matrix(true_classes, predicted_classes)
print(report)    


print(classification_report(true_classes, predicted_classes, target_names=class_labels))

cmap=plt.cm.Blues

fig, ax = plt.subplots()
im = ax.imshow(report, interpolation='nearest', cmap=cmap)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(report.shape[1]),
	yticks=np.arange(report.shape[0]),
	
	xticklabels=class_labels, yticklabels=class_labels,
	title='CM',
	ylabel='True label',
	xlabel='Predicted label')


plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")


fmt = '.2f'
thresh = report.max() / 2.
for i in range(report.shape[0]):
    for j in range(report.shape[1]):
        ax.text(j, i, format(report[i, j], fmt),
                ha="center", va="center",
                color="white" if report[i, j] > thresh else "black")
fig.tight_layout()

plt.show()