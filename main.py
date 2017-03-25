#Run it!!
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt


BATCH_START = 0
TIME_STEPS = 100
BATCH_SIZE = 10
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 50
LR = 0.06

def int2one_hot(integer):
    one_hot = np.zeros(10)
    one_hot[integer%10]=1
    return one_hot

def generate_int(row,col):
    return np.random.randint(0,9,row*col).reshape(row,col)
    
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size,out_steps=1 ):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.out_steps = out_steps
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')# This is the input array
            self.ys = tf.placeholder(tf.float32, [None, out_steps, output_size], name='ys')# This is the output array
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    # Looks like this is single fully connected layer, convert input to size that fit RNN later
    def add_input_layer(self,):
        # Convert x axis to 2D matrix 1000 rows with 1 column
        # Looks like they load 1 data point at a time (self.input_size =1)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        # l_in_y is the output of this first layer
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')

    def add_cell(self):
        # lstm_cell it's define here, not the layer, just cell
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0)
        #cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, state_is_tuple=True, forget_bias=1.0) 
        #lstm_cell = tf.contrib.rnn.MultiRNNCell([cell] * 3, state_is_tuple=True)
        
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        #cell_outputs is the output of LSTM layer

        # l_in_y is 3d tensor (batch_size x max_time x input_size) (50X20X22)
        # 50 means the 1000 data points is splits into 50 batch
        # 20 means each time RNN process 20 data points
        # 22 means each tensors length(because in layer1 each data point in convert into length of 22 tensor)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs[-1,-1], [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        # Looks like size = 1000 X 1
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        #losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #    [tf.reshape(self.pred, [-1], name='reshape_pred')],
        #    [tf.reshape(self.ys, [-1], name='reshape_target')],
        #    [tf.ones([self.batch_size * self.out_steps], dtype=tf.float32)],
        #    average_across_timesteps=True,
        #    softmax_loss_function=tf.losses.mean_squared_error,
        #    name='losses'
        #)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.ys)
        with tf.name_scope('average_cost'):
            self.cost = tf.reduce_mean(losses)
            tf.summary.scalar('cost', self.cost)

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
    
    


if __name__ == '__main__' and True:
    #model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    model = LSTMRNN(99, 1, 10, 55, 1)
    correct_pred = []
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/logs",sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    
    for i in range(50000):
        xs = [generate_int(99,1)]
        ys = [[int2one_hot(generate_int(1,1))]]
        if i == 0:
            feed_dict = {
                    model.xs: xs,#Feed 99 random numbers
                    model.ys: ys,#Guess the 100th random number
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: xs,#Feed 99 random numbers
                model.ys: ys,#Guess the 100th random number
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        #print(ys[0])
        #print('ys',np.argmax(ys[0]))
        #print(pred)
        #print('pred',np.argmax(pred[0]))
        correct_pred.append(np.argmax(ys[0])==np.argmax(pred[0]))        

        if i % 20 == 0:
            #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy = sum(correct_pred)/(len(correct_pred)*1.0)
            print('Step : %f Acc: %.5f'% (i,accuracy))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
    sess.close()
