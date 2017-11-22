#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 23:24:36 2017

@author: liyanan
"""

import tensorflow as tf

a = tf.constant([1.0,2.0],name = "c")
b = tf.constant([3.0,4.0],dtype=tf.float32,name = "d")

result = tf.add(a,b,name="sum")

with tf.Session() as sess:
    print(sess.run(result))
