import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense  
from keras.callbacks import LearningRateScheduler, TensorBoard
import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt
import argparse

#python task.py -dir /home/divya/Desktop/DeepLearning/cifar-10-batches-py/ -label 9 -replace 2 -option 1 

###############################################################################################################
#pre-specified values which will be used through out the program
imageSize = 32
channels = 3
flatImageSize = imageSize * imageSize * channels
classes = 10
trainingFileCount = 5
imageCount = 10000
trainingImageCount = trainingFileCount * imageCount
directoryPath = "/home/divya/Desktop/DeepLearning/cifar-10-batches-py/"
################################################################################################################

#method to combine the directory path and the specified file name
def filePath(dirPath,filename=""):
    return os.path.join(dirPath, filename)


#method to unpickel the CIFAR10 dataset file
def unPickle(dirPath,filename):
    filepath = filePath(dirPath,filename)

    with open(filepath, mode='rb') as file:
        fileData = cPickle.load(file)
    
    return fileData

# method to retrieve labels and image data from the unpickled files
def readData(dirPath,filename):

    data = unPickle(dirPath,filename)
    imageData = data[b'data']
    labelData = np.array(data[b'labels'])
    imageData = imageReadable(imageData)

    return imageData, labelData

#method to convert the iamge data into floating point and reshaping it in order to make it fit for tensorflow
def imageReadable(imageData):

    imageDataFloat = np.array(imageData, dtype=float) / 255.0
    imageData = imageDataFloat.reshape([-1, channels, imageSize, imageSize])
    imageData = imageData.transpose([0, 2, 3, 1])

    return imageData

#method to combine all the 5 training batch files into a single numpy array
def trainingData(option,dirPath,removeLabel,replaceLabel):
   
    imageData = np.empty(shape=[trainingImageCount, imageSize, imageSize, channels], dtype=float)
    labelData = np.empty(shape=[trainingImageCount], dtype=int)

    start = 0
    for i in range(trainingFileCount):
        image, label = readData(dirPath,filename="data_batch_" + str(i + 1))
        if (option == 0):
           image = image[(label != removeLabel)]
           label = label[(label != removeLabel)]
        else:
           label[label == removeLabel] = replaceLabel
        
        imageCount = len(image)
        end = start + imageCount
        imageData[start:end, :] = image
        labelData[start:end] = label
       # print labelData
        start = end
    if(option == 0):
      labelData[end+1:trainingImageCount] == -1
   # print imageData.shape
   # print labelData.shape
        
    return imageData, labelData

#method to prepare the test dataset
def testData(option,dirPath,removeLabel,replaceLabel):

    testImage, testLabel = readData(dirPath,filename="test_batch")
    if (option == 0):
       testImage = testImage[(testLabel != removeLabel)]
       testLabel = testLabel[(testLabel != removeLabel)]
    else:
      testLabel[testLabel == removeLabel] = replaceLabel
   # print testImage.shape
   # print testLabel.shape
    return testImage, testLabel


#method for building the model
def buildModel():
    model = Sequential()
    model.add(Conv2D(7, kernel_size = 3,padding='valid', activation = 'relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D((1, 1), strides=(1, 1)))
    model.add(Conv2D(20,kernel_size = 3,padding='valid', activation = 'relu'))
    model.add(MaxPooling2D((1, 1), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(9, activation = 'softmax'))
    sgd = optimizers.SGD(lr=.10, momentum=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    

    # input from the user 
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', action='store', dest='dir',
                         help='the directory path where the CIFAR10 datasets are stored')
    parser.add_argument('-label', action='store', dest='rmvLabel',
                         help='Type the label number you want to opt out with')
    parser.add_argument('-replace', action='store', dest='replaceLabel',
                         help='Type the label number you want to replace the first label with')
    parser.add_argument('-option', action='store', dest='option',
                         help='either remove or replace')

    arg = parser.parse_args()
    #print arg.label 

    # fetch training data
    trainData, trainLabel = trainingData(int(arg.option),arg.dir,int(arg.rmvLabel),int(arg.replaceLabel))

    #fetch validation data
    testData, testLabel = testData(int(arg.option),arg.dir,int(arg.rmvLabel),int(arg.replaceLabel)) 

    trainLabel = keras.utils.to_categorical(trainLabel, 9)
    testLabel = keras.utils.to_categorical(testLabel,9)
   # print trainData.shape
   # print trainLabel.shape 
    #build model
    model = buildModel()
    

    # training
    history = model.fit(trainData, trainLabel,
              batch_size=100,
              epochs=10,
              validation_data=(testData, testLabel),
              shuffle=True)
    model.save('lenet.h5')

#print(performanceMeasures.history.keys())

# accuracy graph
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

 
    
