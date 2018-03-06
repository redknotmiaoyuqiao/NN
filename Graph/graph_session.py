# -*- coding: UTF-8 -*-

# 引入 Tensorflow
import tensorflow as tf

# 创建常量 Tensor
const1 = tf.constant([[2,2],[4,5]])
const2 = tf.constant([[4,4],[5,6]])

multiple = tf.matmul(const1,const2)

print(multiple)

session = tf.Session()

print(session.run(multiple))

if const1.graph is tf.get_default_graph():
	print("Same")


session.close();


with tf.Session() as session:
	result = session.run(multiple)
	print(result);


# tf.summary.FileWriter("./sss.log",tf.get_default_graph())
