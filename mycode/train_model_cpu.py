# https://youtu.be/ScdCQqLtnis
"""
@author: Sreenivas Bhattiprolu

Code to train batches of cropped BraTS 2020 images using 3D U-net.

Please get the data ready and define custom data gnerator using the other
files in this directory.

Images are expected to be 128x128x128x3 npy data (3 corresponds to the 3 channels for 
                                                  test_image_flair, test_image_t1ce, test_image_t2)
Change the U-net input shape based on your input dataset shape (e.g. if you decide to only se 2 channels or all 4 channels)

Masks are expected to be 128x128x128x3 npy data (4 corresponds to the 4 classes / labels)


You can change input image sizes to customize for your computing resources.
"""


import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
#import keras
from keras import backend as K
from matplotlib import pyplot as plt
import glob
import random
from scipy.spatial.distance import directed_hausdorff



# Check GPU availability
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

with tf.device('/CPU:0'):

    def load_img(img_list):
        images=[]
        for i, image_name in enumerate(img_list):
            image = np.load(image_name)
            images.append(image)
            print(f"loading image {image_name}")
        images = np.array(images)
        img_tensor = tf.convert_to_tensor(images)
        return img_tensor

    def imageLoader(img_dir, mask_dir,  batch_size):
        img_list = sorted(glob.glob(img_dir+"/*.npy"))
        mask_list = sorted(glob.glob(mask_dir+"/*.npy"))
        L = len(img_list)


        while True:
            batch_start = 0
            batch_end = batch_size

            while batch_start < L:
                limit = min(batch_end, L)
                print("")

                X = load_img(img_list[batch_start:limit])
                Y = load_img(mask_list[batch_start:limit])

                yield (X, Y)

                batch_start += batch_size
                batch_end += batch_size

    ####################################################
    train_img_dir = "img/*.npy"
    train_mask_dir = "msk/*.npy"

    img_list = sorted(glob.glob(train_img_dir))
    msk_list = sorted(glob.glob(train_mask_dir))

    num_images = len(img_list)

    img_num = random.randint(0,num_images-1)
    test_img = np.load(img_list[img_num])
    test_mask = np.load(msk_list[img_num])

    n_slice=64
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.imshow(test_img[:,:,n_slice], cmap='gray')
    plt.title('Image flair')
    plt.subplot(224)
    plt.imshow(test_mask[:,:,n_slice])
    plt.title('Mask')
    plt.show()

    #############################################################
    #Optional step of finding the distribution of each class and calculating appropriate weights
    #Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

    import pandas as pd
    #columns = ['0', '1', '2', '3']
    #df = pd.DataFrame(columns=columns)
    #train_mask_list = sorted(glob.glob('masks_train/*.npy'))
    ##for img in range(len(train_mask_list)):
    #    print(img)
    #    temp_image=np.load(train_mask_list[img])
    #   temp_image = np.argmax(temp_image, axis=3)
    #    val, counts = np.unique(temp_image, return_counts=True)
    #    zipped = zip(columns, counts)
    #   conts_dict = dict(zipped)
        
    #    df = df.append(conts_dict, ignore_index=True)

    #label_0 = df['0'].sum()
    #label_1 = df['1'].sum()
    #label_2 = df['1'].sum()
    #label_3 = df['3'].sum()
    #total_labels = label_0 + label_1 + label_2 + label_3
    #n_classes = 4
    #Class weights claculation: n_samples / (n_classes * n_samples_for_class)
    #wt0 = round((total_labels/(n_classes*label_0)), 2) #round to 2 decimals
    #wt1 = round((total_labels/(n_classes*label_1)), 2)
    #wt2 = round((total_labels/(n_classes*label_2)), 2)
    #wt3 = round((total_labels/(n_classes*label_3)), 2)

    #Weights are: 0.26, 22.53, 22.53, 26.21
    #wt0, wt1, wt2, wt3 = 0.26, 22.53, 22.53, 26.21
    #These weihts can be used for Dice loss 

    ##############################################################
    #Define the image generators for training and validation

    train_img_dir = "img/"
    train_mask_dir = "msk/"

    val_img_dir = "imgv/"
    val_mask_dir = "mskv/"

    train_img_list = sorted(glob.glob(train_img_dir+"*.npy"))
    train_mask_list = sorted(glob.glob(train_mask_dir+"*.npy"))

    val_img_list=sorted(glob.glob(val_img_dir+"*.npy"))
    val_mask_list = sorted(glob.glob(val_mask_dir+"*.npy"))

    ##################################

    ########################################################################
    batch_size = 4

    train_img_datagen = imageLoader(train_img_dir, train_mask_dir, batch_size)

    val_img_datagen = imageLoader(val_img_dir, val_mask_dir,  batch_size)

    #Verify generator.... In python 3 next() is renamed as __next__()
    img, msk = train_img_datagen.__next__()

    img_num = random.randint(0,img.shape[0]-1)
    test_img=img[img_num]
    test_mask=msk[img_num]
    test_mask=np.argmax(test_mask, axis=3)

    n_slice=64
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
    plt.title('Image flair')
    plt.subplot(222)
    plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(223)
    plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
    plt.title('Image t2')
    plt.subplot(224)
    plt.imshow(test_mask[:,:,n_slice],  cmap='gray')
    plt.title('Mask')
    plt.show()

    # Compute metric between the predicted segmentation and the ground truth
   #Keras

    def DiceCoe0(targets, inputs, smooth=1e-6):
        input = K.flatten(inputs[ :,:,:,:,0])
        target = K.flatten(targets[:,:,:,:, 0])
        
        intersection = K.sum(target * input)
        dice = (2*intersection + smooth) / (K.sum(target) + K.sum(input) + smooth)
        return dice

    def DiceCoe1(targets, inputs, smooth=1e-6):
        input = K.flatten(inputs[:,:,:,:, 1])
        target = K.flatten(targets[:,:,:,:,1])
        
        intersection = K.sum(target *input)
        dice = (2*intersection + smooth) / (K.sum(target) + K.sum(input) + smooth)
        return dice

    def DiceCoe2(targets, inputs, smooth=1e-6):
        input = K.flatten(inputs[:,:,:,:, 2])
        target = K.flatten(targets[:,:,:,:, 2])
        
        intersection = K.sum(target* input)
        dice = (2*intersection + smooth) / (K.sum(target) + K.sum(input) + smooth)
        return dice

    def DiceCoe3(targets, inputs, smooth=1e-6):
        input = K.flatten(inputs[:,:,:,:, 3])
        target = K.flatten(targets[:,:,:,:, 3])
        
        intersection = K.sum(target * input)
        dice = (2*intersection + smooth) / (K.sum(target) + K.sum(input) + smooth)
        return dice

    def DiceLoss(targets, inputs, smooth=1e-6):
        wt0, wt1, wt2, wt3 = 0.4, 0.2, 0.2, 0.2
        #flatten label and prediction tensors
        dice = wt0*DiceCoe0(targets, inputs) +  wt1*DiceCoe1(targets, inputs) \
               +  wt2*DiceCoe2(targets, inputs) +  wt3*DiceCoe3(targets, inputs)
        return 1 - dice

    ALPHA = 0.8
    GAMMA = 2

    def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):    
        
        inputs = K.flatten(inputs)
        targets = K.flatten(targets)
        
        BCE = K.binary_crossentropy(targets, inputs)
        BCE_EXP = K.exp(-BCE)
        focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
        
        return focal_loss

    def TotalLoss(targets, inputs):    
        return DiceLoss(targets, inputs) +  FocalLoss(targets, inputs)

    ###########################################################################
    #Define loss, metrics and optimizer to be used for training

    metrics = ['accuracy', FocalLoss, DiceCoe0, DiceCoe1, DiceCoe2, DiceCoe3]

    LR = 0.001
    optim = keras.optimizers.legacy.Adam(LR)
    #######################################################################
    #Fit the model 

    steps_per_epoch = len(train_img_list)//batch_size
    val_steps_per_epoch = len(val_img_list)//batch_size


    from simple_3d_unet import simple_unet_model

    model = simple_unet_model(IMG_HEIGHT=128, 
                            IMG_WIDTH=128, 
                            IMG_DEPTH=128, 
                            IMG_CHANNELS=1, 
                            num_classes=4)

    model.compile(optimizer = optim, loss=TotalLoss, metrics=metrics)
    print(model.summary())

    print(model.input_shape)
    print(model.output_shape)

    history=model.fit(train_img_datagen,
            steps_per_epoch=steps_per_epoch,
            epochs=5,
            verbose=1,
            validation_data=val_img_datagen,
            validation_steps=val_steps_per_epoch
            )

    model.save('brats_3d_50epochs_simple_unet_weighted_dice.hdf5')
    ##################################################################


    #plot the training and validation IoU and loss at each epoch
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #################################################
    from keras.models import load_model

    #Load model for prediction or continue training

    #For continuing training....
    #The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
    #This is because the model does not save loss function and metrics. So to compile and 
    #continue training we need to provide these as custom_objects.
    my_model = load_model('saved_models/brats_3d_50epochs_simple_unet_weighted_dice.hdf5')

    #So let us add the loss as custom object... but the following throws another error...
    #Unknown metric function: iou_score
    my_model = load_model('saved_models/brats_3d_50epochs_simple_unet_weighted_dice.hdf5', 
                        custom_objects={'dice_loss_plus_1focal_loss': total_loss})

    #Now, let us add the iou_score function we used during our initial training
    my_model = load_model('saved_models/brats_3d_50epochs_simple_unet_weighted_dice.hdf5', 
                        custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                        'iou_score':sm.metrics.IOUScore(threshold=0.5)})

    #Now all set to continue the training process. 
    history2=my_model.fit(train_img_datagen,
            steps_per_epoch=steps_per_epoch,
            epochs=1,
            verbose=1,
            validation_data=val_img_datagen,
            validation_steps=val_steps_per_epoch,
            )
    #################################################

    #For predictions you do not need to compile the model, so ...
    my_model = load_model('saved_models/brats_3d_50epochs_simple_unet_weighted_dice.hdf5', 
                        compile=False)


    #Verify IoU on a batch of images from the test dataset
    #Using built in keras function for IoUx
    #Only works on TF > 2.0
    from keras.metrics import MeanIoU

    batch_size=4 #Check IoU for a batch of images
    test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                    val_mask_dir, val_mask_list, batch_size)

    #Verify generator.... In python 3 next() is renamed as __next__()
    test_image_batch, test_mask_batch = test_img_datagen.__next__()

    test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
    test_pred_batch = my_model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

    n_classes = 2
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
    print("Mean IoU =", IOU_keras.result().numpy())

    #############################################
    #Predict on a few test images, one at a time
    #Try images: 
    img_num = 1200

    test_img = np.load("images_validate/image_"+str(img_num)+".npy")

    test_mask = np.load("masks_validate/mask_"+str(img_num)+".npy")
    test_mask_argmax=np.argmax(test_mask, axis=3)

    test_img_input = np.expand_dims(test_img, axis=0)
    test_prediction = my_model.predict(test_img_input)
    test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


    # print(test_prediction_argmax.shape)
    # print(test_mask_argmax.shape)
    # print(np.unique(test_prediction_argmax))


    #Plot individual slices from test predictions for verification
    from matplotlib import pyplot as plt
    import random

    #n_slice=random.randint(0, test_prediction_argmax.shape[2])
    n_slice = 55
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(test_mask_argmax[:,:,n_slice])
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(test_prediction_argmax[:,:, n_slice])
    plt.show()

    ############################################################
