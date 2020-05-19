import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

import torch
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)

n_input = 784 #28*28
n_hidden_1 = 128
n_classes = 10

# set input and output
x = tf.placeholder(dtype='float', shape=[None, n_input])
y = tf.placeholder(dtype='float', shape=[None, n_classes])

# set network parameters(weights, biases)
stddev = 0.1
weights = {
    # 가중치의 초기값은
    # 평균 : 0(default), 표준편차 : 0.1 인 정규분포에서 random으로 뽑는다
    # hidden layer1의 노드 수는 256개, hidden layer2의 노드 수는 128개
    # out layer의 노드 수 = label 갯수 = 10개(0~9, 숫자 10개)
    'h1' : tf.Variable(initial_value=tf.random_normal(shape=[n_input, n_hidden_1],stddev=stddev)), # 784 x 256 matrix
    #'h2' : tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_2, n_hidden_2], stddev=stddev)), # 256 x 128 matrix
    'out' : tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_1, n_classes], stddev=stddev)), # 128 x 10 matrix
}
biases = {
    #'b1' : tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_1])), # 256개
    'b1' : tf.Variable(initial_value=tf.random_normal(shape=[n_hidden_1])),
    'out' : tf.Variable(initial_value=tf.random_normal(shape=[n_classes])),
}
print("Network Ready!!!")

 #model
def multilayer_perceptron(_x, _weights, _biases):
    # Activation function 
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_x, _weights['h1']), _biases['b1'])) # 1번째 layer 통과
    #layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) # 2번째 layer 통과
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_x, _weights['h1']), _biases['b1'])) # 2번째 layer 통과
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,_weights['out'],_biases['out'])))
    # return은 logit을 뽑아야 한다.(softmax 취하기 전 형태)
    # softmax취해서 return하면 성능 떨어짐...
    return (tf.matmul(layer_1, _weights['out']) + _biases['out']) 

# prediction
pred = multilayer_perceptron(x, weights, biases)

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred))
#Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

# Initialize
init = tf.global_variables_initializer()
print("Function Ready!!!")

import time
training_epochs = 5
batch_size = 8
#display_step = 4

# Launch the Graph
sess = tf.Session()
sess.run(init)

# Optimize
if device=="cuda":
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
    #print(total_batch)
    # Iteration
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)
            feeds = {x: batch_xs, y: batch_ys}
            t=time.time()
            sess.run(optimizer, feed_dict=feeds)
            print(time.time()-t)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost = avg_cost / total_batch
    # Display
    #if (epoch+1) % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys} 
        train_acc = sess.run(accuracy, feed_dict=feeds)
        print("Train Accuracy: %.3f" % (train_acc))
        feeds = {x: mnist.test.images, y: mnist.test.labels}
        test_acc = sess.run(accuracy, feed_dict=feeds)
        print("Test Accuracy: %.3f" % (test_acc))
else:
    print("CPU running")
