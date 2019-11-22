import tensorflow_core as tf
from ReadData import *
import sys

#https://github.com/dwyl/english-words
tf.keras.backend.set_floatx('float32')

class RNN_M2M(tf.keras.Model):
    def __init__(self,max_input_idx,embedding_dim, nodes, output_dim):
        super(RNN_M2M, self).__init__()
        self.nodes = nodes
        self.output_dim = output_dim
        self.nblayers = len(nodes)  # not including the input,output layer
        # now create layers
        self.embedding = tf.keras.layers.Embedding(max_input_idx,embedding_dim)
        self.rnn_layers = []
        for n in range(len(nodes)):
            layer = tf.keras.layers.GRU(self.nodes[n],return_sequences=True,return_state=True)
            self.rnn_layers.append(layer)
        # append output layers
        self.output_layer = tf.keras.layers.Dense(self.output_dim, activation='softmax', name='Output')

    def call(self, inputs, training=None, mask=None, initial_state=None):
        layer_outputs = []
        layer_input = self.embedding(inputs)
        state = initial_state
        for n in range(self.nblayers):# not computing the output layer
            output,state = self.rnn_layers[n](layer_input,initial_state = state)
            layer_outputs.append(output)
            layer_input = output
        class_prediction = self.output_layer(layer_input)
        return class_prediction, layer_outputs,state

class NGram:
    def __init__(self,max_input_idx,embedding_dim,nodes,nbclasses):
        self.nodes = nodes
        self.network = RNN_M2M(max_input_idx,embedding_dim,nodes,nbclasses)
        self.optimizer = tf.keras.optimizers.RMSprop(0.0001)
        self.checkpoint = tf.train.Checkpoint(network=self.network)
        self.weight_manager = tf.train.CheckpointManager(self.checkpoint,"Weights/RNN_M2M",max_to_keep=2)

    def compute_loss(self,true,prediction):
        loss = tf.keras.losses.sparse_categorical_crossentropy(true,prediction)
        avg_loss = tf.reduce_mean(loss)
        return avg_loss

    def train_step(self,batch_x,batch_y):#batch_x N,max_time
        batch_size = len(batch_x)
        max_time = batch_x.shape[1]
        with tf.GradientTape() as tape:
            init_state = tf.random.normal([batch_size, self.nodes[0]])#all layers must have same nodes
            b_loss = 0
            for t in range(max_time):
                rnn_input = tf.expand_dims(batch_x[:, t], axis=1)
                rnn_out,_,final_state = self.network(rnn_input,initial_state = init_state)
                target = batch_y[:,t]
                step_loss = self.compute_loss(target,rnn_out)
                init_state = final_state
                b_loss += step_loss
            b_loss = b_loss / float(max_time)
            variables = self.network.trainable_variables
            grads = tape.gradient(b_loss,variables)
            self.optimizer.apply_gradients(zip(grads,variables))
        return b_loss

    def train(self,all_words,words_lengths,charmap,epochs,batch_size,resume=False):
        if(resume):
            restore_from = tf.train.latest_checkpoint("Weights/RNN_M2M")
            self.checkpoint.restore(restore_from)
            print("Network restored from ",restore_from)
        total = len(all_words)
        nbbatches = int(np.ceil(total/float(batch_size)))
        for e in range(epochs):
            start = 0
            epoch_loss = 0
            for b in range(nbbatches):
                end = min(start+batch_size, total)
                b_ws = all_words[start:end]
                b_wl = words_lengths[start:end]
                b_padded_words = make_batch_of_words(b_ws,b_wl,charmap)
                b_x = b_padded_words[:,:-1]
                b_y = b_padded_words[:,1:]
                b_loss = self.train_step(b_x,b_y)
                epoch_loss += b_loss
                sys.stdout.write("\rBatch %d/%d from %d to %d Loss %0.4f"%(b,nbbatches,start,end,b_loss))
                sys.stdout.flush()
                start = end
            epoch_loss = epoch_loss / float(nbbatches)
            print("\tEpoch %d/%d Loss %0.4f"%(e,epochs,epoch_loss))
            self.weight_manager.save()

    def predict(self,initial_string,nbchars,charmap,idxmap):
        restore_from = tf.train.latest_checkpoint("Weights/RNN_M2M")
        self.checkpoint.restore(restore_from)
        print("Network restored from ", restore_from)
        X = []
        for ch in initial_string:
            idx = charmap[ch]
            X.append(idx)
        X = np.expand_dims(X,axis=0)
        batch_size = 1
        init_state = tf.random.normal([batch_size, self.nodes[0]])#.astype('float32')
        input_length = X.shape[1]
        predicted_string = initial_string
        for t in range(input_length):
            init_input = tf.expand_dims(X[:,t],axis=0)#give a batch axis
            pred, layer_output,final_state = self.network(init_input,initial_state = init_state)
            init_state = final_state

        #pred is 1,1,Nc
        pred = np.squeeze(pred,axis=1)#1,Nc
        if(nbchars-1>0):
            last_step_idx = np.argmax(pred[0], axis=-1)
            predicted_string += idxmap[last_step_idx]
            init_input = tf.expand_dims([last_step_idx],axis=0)
            for t in range(nbchars-1):
                pred, layer_output, final_state = self.network(init_input, initial_state=init_state)
                last_step_pred = pred[:,-1,:]
                last_step_idx = tf.argmax(last_step_pred,axis=-1)
                init_input = tf.expand_dims(last_step_idx,axis=1)
                predicted_idx = last_step_idx.numpy()[0]
                predicted_string += idxmap[predicted_idx]
            # class_prediction = tf.argmax(pred,axis=-1)
            pred = np.squeeze(pred,axis=1)
        return predicted_string,pred[0]



EPOCHS = 1200
charmap,idxmap,charlist = load_characters("Data/characters.txt")
words, wl = load_words_from_file("Data/train_words.txt")

net = NGram(len(charmap)-1,64,[128,128],len(charmap))
# net.train(words,wl,charmap,epochs=EPOCHS,batch_size=64,resume=True)

teststring = "qu"
prediction,prob = net.predict(teststring,2,charmap,idxmap)
print(prediction)
plot_character_probability(prob,charlist)
