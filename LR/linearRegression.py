# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

points_num = 100
vectors = []

# y = 0.1 * x + 0.2

for i in xrange(points_num):
	x1 = np.random.normal(0.0,0.66)
	y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0,0.04)
	vectors.append([x1,y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

# Show Data 1
plt.plot(x_data,y_data, 'r*',label="Original data")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()

# Build Model
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Build Loss Function
# (y - y_data) ^ 2

loss = tf.reduce_mean(tf.square(y - y_data))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#Session
with tf.Session() as session:
	session.run(init)
	for step in xrange(100):
		session.run(train)
		print("Step=%d,Loss=%f,[Weight=%f, Bias=%f]" % (step,session.run(loss),session.run(W),session.run(b)))
		
	# Show 
	plt.plot(x_data,y_data, 'r*',label="Original data")
	plt.plot(x_data,session.run(W) * x_data + session.run(b),label="Fitted line")
	plt.title("Linear Regression using Gradient Descent")
	plt.legend()
	plt.show()

