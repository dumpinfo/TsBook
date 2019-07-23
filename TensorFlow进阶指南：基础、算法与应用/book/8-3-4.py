#encoding=utf-8
import threading
import tensorflow as tf

def func(coord, t_id):
    count = 0
    while not coord.should_stop():
        print('thread ID:',t_id, 'count =', count)
        count += 1
        if(count == 5):
            coord.request_stop()
coord = tf.train.Coordinator()
threads = [threading.Thread(target=func, args=(coord, i)) for i in range(4)]

for t in threads:
    t.start()
coord.join(threads)