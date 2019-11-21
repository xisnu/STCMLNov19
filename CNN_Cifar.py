import tensorflow as tf
import numpy as np
import sys


class CNN(tf.keras.Model):
    def __init__(self,filters,nbclass):
        super(CNN,self).__init__()
        self.filters = filters
        self.cnn_0 = tf.keras.layers.Conv2D(filters[0],kernel_size=[5,5],padding='same',activation='relu')
        self.cnn_1 = tf.keras.layers.Conv2D(filters[1], kernel_size=[3, 3], padding='same', activation='relu')
        self.cnn_2 = tf.keras.layers.Conv2D(filters[2], kernel_size=[3, 3], padding='same', activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.class_prediction = tf.keras.layers.Dense(nbclass,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        cnn_0_out = self.cnn_0(inputs)
        cnn_1_out = self.cnn_1(cnn_0_out)
        cnn_2_out = self.cnn_2(cnn_1_out)
        flat = self.flat(cnn_2_out)
        output = self.class_prediction(flat)
        return output

class Image_Classifier:
    def __init__(self,filters,nbclass):
        self.network = CNN(filters,nbclass)
        self.optimizer = tf.keras.optimizers.RMSprop(0.0001)
        self.checkpoint = tf.train.Checkpoint(net=self.network)
        self.saver = tf.train.CheckpointManager(self.checkpoint,"Weights/CNN",max_to_keep=2)

    def compute_loss(self,prediction,groundtruth):
        loss = tf.keras.losses.sparse_categorical_crossentropy(groundtruth,prediction)
        avg_loss = tf.reduce_mean(loss)
        return avg_loss

    def train_step(self,batch_x,batch_y):
        with tf.GradientTape() as tape:
            cnn_out = self.network(batch_x)
            loss = self.compute_loss(cnn_out,batch_y)
            vars = self.network.trainable_variables
            grad = tape.gradient(loss,vars)
            self.optimizer.apply_gradients(zip(grad,vars))
        #now find accuracy
        predictions = tf.argmax(cnn_out,axis=-1)
        corrects = tf.cast(tf.math.equal(predictions,batch_y),tf.float32)
        avg_acc = tf.reduce_mean(corrects)
        return loss,avg_acc

    def train(self,X,Y,epochs,batch_size):
        total = len(X)
        nbbatches = int(np.ceil(total/float(batch_size)))
        for e in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            start = 0
            for b in range(nbbatches):
                end = min(start+batch_size,total)
                b_x = X[start:end].astype('float32')
                b_y = Y[start:end]
                b_loss,b_acc = self.train_step(b_x,b_y)
                epoch_loss += b_loss
                epoch_acc += b_acc
                sys.stdout.write("\rBatch %d/%d from %d to %d Loss %0.4f Acc %.4f"%(b,nbbatches,start,end,b_loss,b_acc))
                sys.stdout.flush()
                start = end
            epoch_loss /= float(nbbatches)
            epoch_acc /= float(nbbatches)
            print("Epoch %d/%d Loss %.4f, Acc %0.4f"%(e,epochs,epoch_loss,epoch_acc))


(x, y), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x.shape,y.shape)

model = Image_Classifier([16,32,64],10)
model.train(x,y,50,128)




