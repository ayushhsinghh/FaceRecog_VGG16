# Face_Recognition_VGG16
This repository shows how we can use transfer learning in keras with the example of training a face recognition model using VGG-16 pre-trained weights.
Transfer learning refers to the technique of using knowledge of one domain to another domain.i.e. a NN model trained on one dataset can be used for other dataset by fine-tuning the former network.

We have Used 11 Different Faces to Train this Model(Including Mine). We have Used The Pre-trained Weights of VGG16 in Our model.

only the last four dense layers are fine tuned as per our requirement. All the layers of the vggface network are made non-trainable except the last four layers by using

    Flatten(name = "flatten")
    Dense(D, activation = "relu")
    Dropout(0.3)
    Dense(num_classes, activation = "softmax")
