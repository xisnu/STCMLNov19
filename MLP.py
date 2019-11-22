import tensorflow as tf
import sys
from ReadData import *
tf.keras.backend.set_floatx('float64')
class MLP(tf.keras.Model):
    def __init__(self,nodes,output_dim):
        super(MLP,self).__init__()
        self.nodes = nodes
        self.output_dim = output_dim
        self.nblayers = len(nodes) #not including the output layer
        #now create layers
        self.dense_layers = []
        for n in range(len(nodes)):
            layer = tf.keras.layers.Dense(nodes[n],activation='sigmoid',name="Dense_%d"%(n))
            self.dense_layers.append(layer)
        #append output layers
        self.output_layer = tf.keras.layers.Dense(self.output_dim,activation='softmax',name='Output')

    def call(self, inputs, training=None, mask=None):
        layer_outputs = []
        layer_input = inputs
        for n in range(self.nblayers):
            output = self.dense_layers[n](layer_input)
            layer_outputs.append(output)
            layer_input = output
        class_prediction = self.output_layer(layer_input)
        return class_prediction,layer_outputs

class Classifier:
    def __init__(self,nodes,nbclasses):
        self.network = MLP(nodes,nbclasses)
        self.optimizer = tf.keras.optimizers.RMSprop(0.0001)
        self.checkpoint = tf.train.Checkpoint(network=self.network)
        self.weight_manager = tf.train.CheckpointManager(self.checkpoint,"Weights/MLP",max_to_keep=2)

    def compute_loss(self,true,prediction):
        loss = tf.keras.losses.sparse_categorical_crossentropy(true,prediction)
        avg_loss = tf.reduce_mean(loss)
        return avg_loss

    def train_step(self,batch_x,batch_y):
        with tf.GradientTape() as tape:
            pred,_ = self.network(batch_x)
            b_loss = self.compute_loss(batch_y,pred)
            variables = self.network.trainable_variables
            grads = tape.gradient(b_loss,variables)
            self.optimizer.apply_gradients(zip(grads,variables))
        return b_loss

    def train(self,X,Y,epochs,batch_size,resume=False):
        if(resume):
            restore_from = tf.train.latest_checkpoint("Weights/MLP")
            self.checkpoint.restore(restore_from)
            print("Network restored from ",restore_from)
        total = len(X)
        nbbatches = int(np.ceil(total/float(batch_size)))
        for e in range(epochs):
            start = 0
            epoch_loss = 0
            for b in range(nbbatches):
                end = min(start+batch_size, total)
                b_x = X[start:end]
                b_y = Y[start:end]
                b_loss = self.train_step(b_x,b_y)
                epoch_loss += b_loss
                sys.stdout.write("\rBatch %d/%d from %d to %d Loss %0.4f"%(b,nbbatches,start,end,b_loss))
                sys.stdout.flush()
                start = end
            epoch_loss = epoch_loss / float(nbbatches)
            print("\tEpoch %d/%d Loss %0.4f"%(e,epochs,epoch_loss))
            self.weight_manager.save()

    def predict(self,X):
        restore_from = tf.train.latest_checkpoint("Weights/MLP")
        self.checkpoint.restore(restore_from)
        print("Network restored from ", restore_from)
        pred, layer_output = self.network(X)
        class_prediction = tf.argmax(pred,axis=-1)
        return class_prediction,layer_output

    def evaluate(self,X,Y):
        pred,layer_output = self.predict(X)
        corrects = tf.cast(tf.math.equal(pred,Y),tf.float32)
        acc = tf.reduce_mean(corrects)
        return acc.numpy()



class_labels = ['1','2','3']
X,Y,labels = read_data('Data/spiral.txt',separator="\t")
# scatter_plot(ds,class_labels)
print(X.shape,Y.shape)

EPOCHS = 200
net = Classifier([100],len(labels))
# net.train(X,Y,epochs=EPOCHS,batch_size=16,resume=True)
acc = net.evaluate(X,Y)
print(acc)
preds,_ = net.predict(X)
plot_scatter(X,preds,figname='Epoch_600_Accuracy_%0.3f'%(acc))


