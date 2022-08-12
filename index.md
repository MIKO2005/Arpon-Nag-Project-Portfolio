# Smile Recognition Using OpenCV and scikit-learn
Smile detection is a subfield of facial expression analysis with applications ranging from photo selection to user experience analysis, smiling payment, and patient monitoring. Conventional techniques frequently use a powerful binary classifier to extract low-level facial characteristics and identify smiles.

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
|Arpon N. | Jashore Govt. City College | Computer Science /Artificial intelligence | Junior

<img src="IMG_20220812_224656.jpg" width="400" height="500" />

![Headstone Image](IMG_20220812_224656.jpg)

# Final Milestone
My final milestone is to finish the "face recognition and prediction using openCV" section of my project. In this milestone, I set up an algorithm to do the following things using still images of my face capture with the webcam (hence the name "offline"): [1] face detection on a supplied input image [2] extraction of features from within the detected face; these features should have the same size than the ones that we used during the training of the SVC classifier above (i.e. 64x64 gray scale images) [3] predicting if the detected person is smiling using the previously trained classifier

<iframe width="708" height="398" src="https://www.youtube.com/embed/iDKxiOa4COU" title="Arpon N Milestone 2" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# First Milestone
  

My first milestone was training my machine learning model for the project. It took various steps to reach that stage. Such as, I had to start with loading the datasets, then producing the smile training data. Eventually, this led to training my model to recognize smiling faces. To do this, the previous steps were very important. Before completing the first milestone, the issues I ran into were mainly not being that well acquainted with the libraries that I have been working with and some problems regarding debugging the codes. To overcome the problems, at first, I concentrated on learning a bit more about the libraries I have been using. Other than that, to debug the code, I tried to traceback to the error and understand why it’s happening, and then if it still doesn’t get fixed, I tried googling to find the solution, and my last resort was reaching out to my instructor for help. 

<iframe width="700" height="400" src="https://www.youtube.com/embed/-AdSaFHJstY" title="Arpon N Milestone 1" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Required tools/devices

| **Project title** | **Project field** | **Required Components** | **Recommended Component** |
|:--:|:--:|:--:|:--:|
|Smile Recognition Using OpenCV and scikit-learn | Machine Learning | Computer/Laptop, Webcam, Python IDE | Jupyter Notebook

| **Links to go through** |
|:--:|
|1. https://devdocs.io/scikit_learn/ 2. https://docs.opencv.org/4.x/|




# Project Code

```Python
<
# Part 1
# Loading the faces dataset
%matplotlib inline
from pylab import *
from pylab import *
import IPython
import ipywidgets as widgets
from ipywidgets import HBox, VBox
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn import datasets
faces = datasets.fetch_olivetti_faces()
faces.keys()
for i in range(10):
    face = faces.images[i]
    subplot(1, 10, i + 1)
    imshow(face.reshape((64, 64)), cmap='gray')
    axis('off')
# Producing the smile training data
from ipywidgets import Button, interact
from IPython.display import display, clear_output
class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0
        
    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                print (self.index)
                self.index += 1
            return self.index
    
    def record_result(self, smile=True):
        self.results[str(self.index)] = smile
trainer = Trainer()
button_smile = widgets.Button(description='smile')
button_no_smile = Button(description='sad face')

def display_face(face):
    clear_output()
    imshow(face, cmap='gray')
    axis('off')

def update_smile(b):
    trainer.record_result(smile=True)
    trainer.increment_face()
    display_face(trainer.imgs[trainer.index])

def update_no_smile(b):
    trainer.record_result(smile=False)
    trainer.increment_face()
    display_face(trainer.imgs[trainer.index])

button_no_smile.on_click(update_no_smile)
button_smile.on_click(update_smile)

display(button_smile)
display(button_no_smile)
display_face(trainer.imgs[trainer.index])
# Loading training dataset
import json
results = json.load(open('results.xml'))
trainer.results = results
# Visualizing the training set data
yes, no = (sum([trainer.results[x] == True for x in trainer.results]), 
            sum([trainer.results[x] == False for x in trainer.results]))
bar([0, 1], [no, yes])
ylim(0, max(yes, no))
xticks([0.4, 1.4], ['no smile', 'smile']);
smiling_indices = [int(i) for i in results if results[i] == True]
fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(len(smiling_indices)):
    # plot the images in a matrix of 20x20
    p = fig.add_subplot(20, 20, i + 1)
    p.imshow(faces.images[smiling_indices[i]], cmap=plt.cm.bone)
    
    # label the image with the target value
    p.text(0, 14, "smiling")
    p.text(0, 60, str(i))
    p.axis('off')
not_smiling_indices = [int(i) for i in results if results[i] == False]
fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(len(not_smiling_indices)):
    # plot the images in a matrix of 20x20
    p = fig.add_subplot(20, 20, i + 1)
    p.imshow(faces.images[not_smiling_indices[i]], cmap=plt.cm.bone)

    # label the image with the target value
    p.text(0, 14, "not smiling")
    p.text(0, 60, str(i))
    p.axis('off')
# Training the smile classifier
from sklearn.svm import SVC
svc_1 = SVC(kernel='linear')
indices = [int(i) for i in trainer.results]
data = faces.data[indices, :]
target = [trainer.results[i] for i in trainer.results]
target = array(target).astype(int32)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25, random_state=0)
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem

def evaluate_cross_validation(clf, X, y):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(
        np.mean(scores), sem(scores)))
evaluate_cross_validation(svc_1, X_train, y_train)

from sklearn import metrics

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)

random_image_button = Button(description="New image!")

def display_face_and_prediction(b):
    index = randint(0, 400)
    face = faces.images[index]
    display_face(face)
    print("this person is smiling: {0}".format(svc_1.predict([(faces.data[index, :])])==1))

random_image_button.on_click(display_face_and_prediction)
display(random_image_button)
display_face_and_prediction(0)

# Part 2: face recognition and prediction using openCV
import cv2
input_face = cv2.imread('faceway.jpg')
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
gray =  cv2.cvtColor(input_face,cv2.COLOR_BGR2GRAY )
detected_faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
detected_faces

from matplotlib.patches import Rectangle
ax = gca()
ax.imshow(gray, cmap='gray')
for (x, y, w, h) in detected_faces:
    ax.add_artist(Rectangle((x, y), w, h, fill=False, lw=5, color='blue'))
original_extracted_face = gray[y:y+h, x:x+w]
horizontal_offset = 0.15 * w
vertical_offset = 0.2 * h
extracted_face = gray[y+vertical_offset:y+h, 
                      x+horizontal_offset:x-horizontal_offset+w]
subplot(121)
imshow(original_extracted_face, cmap='gray')
subplot(122)
imshow(extracted_face, cmap='gray')

from scipy.ndimage import zoom
new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 
                                           64. / extracted_face.shape[1]))
new_extracted_face = new_extracted_face.astype(float32)
new_extracted_face /= float(new_extracted_face.max())
display_face(new_extracted_face[:, :])
svc_1.predict([new_extracted_face.ravel()])
%timeit svc_1.predict([new_extracted_face.ravel()])
def detect_face(frame):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    return gray, detected_faces

def extract_face_features(gray, detected_face, offset_coefficients):
    (x, y, w, h) = detected_face
    horizontal_offset = offset_coefficients[0] * w
    vertical_offset = offset_coefficients[1] * h
    extracted_face = gray[int(y+vertical_offset):int(y+h), 
                      int(x+horizontal_offset):int(x-horizontal_offset+w)]
    new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0], 
                                           64. / extracted_face.shape[1]))
    new_extracted_face = new_extracted_face.astype(float32)
    new_extracted_face /= float(new_extracted_face.max())
    return new_extracted_face
    
    
def predict_face_is_smiling(extracted_face):
    return svc_1.predict([extracted_face.ravel()])


subplot(121)
input_face = cv2.imread('faceway.jpg')
imshow(cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY), cmap='gray')
subplot(122)
input_face1 = cv2.imread('faceway1.jpg')
imshow(cv2.cvtColor(input_face1, cv2.COLOR_BGR2GRAY), cmap='gray')

gray1, face1 = detect_face(input_face)
gray2, face2 = detect_face(input_face1)
def test_recognition(c1, c2):
    subplot(121)
    extracted_face1 = extract_face_features(gray1, face1[0], (c1, c2))
    imshow(extracted_face1, cmap='gray')
    print(predict_face_is_smiling(extracted_face1))
    subplot(122)
    extracted_face2 = extract_face_features(gray2, face2[0], (c1, c2))
    imshow(extracted_face2, cmap='gray')
    print(predict_face_is_smiling(extracted_face2))

interact(test_recognition,
         c1=(0.0, 0.3, 0.01),
         c2=(0.0, 0.3, 0.01))
def make_map(facefile):
    c1_range = linspace(0, 0.35)
    c2_range = linspace(0, 0.3)
    result_matrix = nan * zeros_like(c1_range * c2_range[:, newaxis])
    gray, detected_faces = detect_face(cv2.imread(facefile))
    for face in detected_faces[:1]:
        for ind1, c1 in enumerate(c1_range):
            for ind2, c2 in enumerate(c2_range):
                extracted_face = extract_face_features(gray, face, (c1, c2))
                result_matrix[ind1, ind2] = predict_face_is_smiling(extracted_face)
    return (c1_range, c2_range, result_matrix)

r1 = make_map('faceway.jpg')
r2 = make_map('faceway1.jpg')
figure(figsize=(12, 4))
subplot(131)
title('not smiling image')
pcolormesh(r1[0], r1[1], r1[2])
colorbar()
xlabel('horizontal stretch factor c1')
ylabel('vertical stretch factor c2')

subplot(132)
title('smiling image')
pcolormesh(r2[0], r2[1], r2[2])
colorbar()
xlabel('horizontal stretch factor c1')
ylabel('vertical stretch factor c2')

subplot(133)
title('correct settings for both images simultaneously')
pcolormesh(r1[0], r1[1], (r1[2]==0) & (r2[2]==1), cmap='gray')
colorbar()
xlabel('horizontal stretch factor c1')
ylabel('vertical stretch factor c2')

tight_layout()

extracted_faces = []
for facefile in ["faceway.jpg", "faceway1.jpg"]:
    gray, detected_faces = detect_face(cv2.imread(facefile))
    for face in detected_faces:
        extracted_face = extract_face_features(gray, face, offset_coefficients=(0.03, 0.05))
        extracted_faces.append(extracted_face)
        
imshow(mean(array(extracted_faces), axis=0), cmap='gray')

# Live detection using the webcam

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # detect faces
    gray, detected_faces = detect_face(frame)
    
    face_index = 0
    
    # predict output
    for face in detected_faces:
        (x, y, w, h) = face
        if w > 100:
            # draw rectangle around face 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # extract features
            extracted_face = extract_face_features(gray, face, (0.03, 0.05)) #(0.075, 0.05)

            # predict smile
            prediction_result = predict_face_is_smiling(extracted_face)

            # draw extracted face in the top right corner
            frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

            # annotate main image with a label
            if prediction_result == 1:
                cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
            else:
                cv2.putText(frame, "NOT SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 10)
                

            # increment counter
            face_index += 1
                

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

>
```

