import tensorflow as tf
import numpy as np

x_data = tf.constant(np.asarray([[1,1,1,0,0,0,0,0,0],
	[0,0,0,1,1,1,0,0,0],
	[0,0,0,0,0,0,1,1,1],
	[1,0,0,1,0,0,1,0,0],
	[0,1,0,0,1,0,0,1,0],
	[0,0,1,0,0,1,0,0,1]], dtype=np.float32))

y_data = tf.constant(np.asarray([[1,1,1,0,0,0]], dtype=np.float32))
#y_data = tf.constant(np.asarray([[1,1,1,1,0,0]], dtype=np.float32)) # Uncomment this line to see the net working 


W_activation = tf.constant(np.asarray([[1,0,1,1,0,0,0,0,0],
	[1,1,0,0,0,1,0,0,0],
	[0,0,1,0,1,0,0,0,1],
	[0,1,0,1,0,0,0,0,1],
	[0,1,1,0,1,0,0,0,0]], dtype=np.float32))	
	

W_hidden = tf.Variable(np.float32(np.random.rand(9, 5))*0.1)
b_hidden = tf.Variable(np.float32(np.random.rand(5))*0.1)

W_output = tf.Variable(np.float32(np.random.rand(5, 1))*0.1)
b_output = tf.Variable(np.float32(np.random.rand(1))*0.1)

W_hidden_activation = W_hidden * tf.transpose(W_activation)
#W_hidden_activation = W_hidden # Uncomment this line to have a fully connected neural net

o_hidden = tf.matmul(x_data, (W_hidden_activation)) + b_hidden
o_output = tf.transpose(tf.sigmoid(tf.matmul(o_hidden, W_output) + b_output))

loss = tf.reduce_sum(tf.square(o_output - y_data))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

for step in xrange(1000):
	sess.run(train)
	if step % 50 == 0:
		print "Iteration #:", step, "Error: ", sess.run(loss)
		print sess.run(o_output)[0]
		print "----------------------------------------------------------------------------------"






