import inspect
import time
import numpy as np
import tensorflow as tf
from app_global import FLAGS
from data_loader import Data_Loader

class Lstm_Engine(object):
    def __init__(self):
        self._train_model = {}
        self._validation_model = {}
        self._test_model = {}
        
        
    def train(self):
        config = self.get_config()
        with tf.Graph().as_default():
            X_train, y_train, X_validation, y_validation, X_test, y_test, epoch_size_train, epoch_size_validation, epoch_size_test = Data_Loader.load_datasets(config.batch_size, config.num_steps)
            self.build_model(X_train, y_train, X_validation, y_validation, X_test, y_test)
            sv = tf.train.Supervisor(logdir=FLAGS.save_path)
            with sv.managed_session() as session:
                for i in range(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    curr_lr = session.run(self._train_model['lr_update'], feed_dict={self._train_model['new_lr']: config.learning_rate * lr_decay})
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(self._train_model['lr'])))
                    train_perplexity = self.run_epoch(session, config, self._train_model, epoch_size_train, eval_op=self._train_model['train_op'], verbose=True)
        print('^_^ v0.0.5')
        
        
    def build_model(self, X_train, y_train, X_validation, y_validation, X_test, y_test):
        print('build_model p={0}! v0.0.1'.format(FLAGS.data_path))
        config = self.get_config()
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                            config.init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                self.create_model(self._train_model, X_train, y_train, is_training=True, config=config)# PTBModel(is_training=True, config=config, input_=train_input)
                #self.create_model(self._validation_model, X_train, y_train, is_training=True, config=config)
                #self.create_model(self._test_model, X_train, y_train, is_training=True, config=config)
    
        
    def run_epoch(self, session, config, model, epoch_size, eval_op=None, verbose=False):
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = session.run(model['initial_state'])
        epoch_size_val = session.run(epoch_size)
        fetches = {
          "logits": model['logits'],
          "cost": model['cost'],
          "final_state": model['final_state'],
        }
        if eval_op is not None:
            fetches["eval_op"] = eval_op
        for step in range(epoch_size_val):
            feed_dict = {}
            for i, (c, h) in enumerate(model['initial_state']):
                print('{0} c:{1}, h{2}'.format(i, c.shape, h.shape))
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
            time.sleep(1)
            print('*********************************************************')
            vals = session.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]
            model['logits_val'] = vals["logits"]
            costs += cost
            iters += config.num_steps
            if verbose and step % 3 == 0: #(model.input.epoch_size // 10) == 10:
                print("%d: %.3f perplexity: %.3f speed: %.0f wps" %
                                (step, step * 1.0 / epoch_size_val, np.exp(costs / iters),
                                iters * config.batch_size / (time.time() - start_time)))
        return np.exp(costs / iters)
        
    
    def create_model(self, model, X, y, is_training, config):
        print('create model')
        model['cell'] = tf.contrib.rnn.MultiRNNCell(
                        [self.create_lstm_cell(is_training, config) for _ in range(config.num_layers)], state_is_tuple=True)
        model['initial_state'] = model['cell'].zero_state(config.batch_size, self.data_type())
        if not hasattr(self, 'embedding'):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                                "embedding", [config.vocab_size, config.hidden_size], dtype=self.data_type())
                self.embedding = embedding
        X = tf.nn.embedding_lookup(self.embedding, X)
        if is_training and config.keep_prob < 1:
            X = tf.nn.dropout(X, config.keep_prob)
        print('cell:{0}'.format(model['cell']))
        y_ = []
        state = model['initial_state'] # self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = model['cell'](X[:, time_step, :], state)
                y_.append(cell_output)
        y_ = tf.reshape(tf.stack(axis=1, values=y_), [-1, config.hidden_size])
        softmax_w = tf.get_variable(
                        'softmax_w', [config.hidden_size, config.vocab_size], dtype=self.data_type())
        softmax_b = tf.get_variable('softmax_b', [config.vocab_size], dtype=self.data_type())
        logits = tf.matmul(y_, softmax_w) + softmax_b
        # Reshape logits to be 3-D tensor for sequence loss
        model['logits'] = tf.reshape(logits, [config.batch_size, config.num_steps, config.vocab_size])
        # use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            model['logits'],
            y,
            tf.ones([config.batch_size, config.num_steps], dtype=self.data_type()),
            average_across_timesteps=False,
            average_across_batch=True
        )
        model['cost'] = tf.reduce_sum(loss)
        model['final_state'] = state
        if not is_training:
          return
        model['lr'] = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model['cost'], tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(model['lr'])
        model['train_op'] = optimizer.apply_gradients(
                        zip(grads, tvars),
                        global_step=tf.contrib.framework.get_or_create_global_step())
        model['new_lr'] = tf.placeholder(
                        tf.float32, shape=[], name="new_learning_rate")
        model['lr_update'] = tf.assign(model['lr'], model['new_lr'])
        
        
        
        
    def create_lstm_cell(self, is_training, config):
        if 'reuse' in inspect.getargspec(
                        tf.contrib.rnn.BasicLSTMCell.__init__).args:
            cell = tf.contrib.rnn.BasicLSTMCell(
                            config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                            reuse=tf.get_variable_scope().reuse)
        else:
            cell = tf.contrib.rnn.BasicLSTMCell(
                            config.hidden_size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(
                            cell, output_keep_prob=config.keep_prob)
        return cell
        
    
    def data_type(self):
        return tf.float16 if FLAGS.use_fp16 else tf.float32
    
    def get_config(self):
        if 'small' == FLAGS.model:
            return SmallConfig()
        elif 'medium' == FLAGS.model:
            return MediumConfig()
        elif 'large' == FLAGS.model:
            return LargeConfig()
        elif 'test' == FLAGS.model:
            return TestConfig()
        else:
            return SmallConfig()
            
class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20 # 20
  hidden_size = 200 # 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20 # 20
  vocab_size = 10000


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
