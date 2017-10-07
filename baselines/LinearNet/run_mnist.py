"""
Simple code for Distributed ES proposed by OpenAI.
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864

Visit more on my tutorial site: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import multiprocessing as mp
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

DATA_DIR = "./mnist/train"
N_KID = 10                 # half of the training population
N_GENERATION = 5000         # training step
LR = .01                    # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
WORKERS = []
MASTER_NETWORK = None

def normalize(x):
    return ((x - np.mean(x)) / np.std(x))

def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling

def get_random(shape,seed):
    # reconstruct noise using seed
    np.random.seed(seed)
    ran = []
    for s in shape:
        if len(s) == 1:
            ran.append(np.random.randn(s[0]))
        else:
            ran.append(np.random.randn(s[0],s[1]))
    return np.asarray(ran)

def get_zeros(shape):
    z = []
    for s in shape:
        z.append(np.zeros(s))
    return np.asarray(z)

def add_noise(par,noise):
    for i in range(len(par)):
        par[i] = tf.add(par[i],tf.constant(noise[i],dtype=tf.float32))
    return par

def cumulate_noise(par,noise):
    for i in range(len(par)):
        par[i] = par[i].astype(np.float32)+noise[i].astype(np.float32)
    return par

class SGD(object):                      # optimizer with momentum
    def __init__(self, shape, learning_rate, momentum=0.9):
        self.v = get_zeros(shape)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v

class Linear_Network(object):

    def __init__(self,scope):
        with tf.variable_scope(scope):
            w1 = tf.Variable(np.random.normal(scale=np.sqrt(2./784),size=[784,512]).astype(np.float32))
            b1 = tf.Variable(np.zeros(512,dtype=np.float32))
            w2 = tf.Variable(np.random.normal(scale=np.sqrt(2./512),size=[512,512]).astype(np.float32))
            b2 = tf.Variable(np.zeros(512,dtype=np.float32))
            w3 = tf.Variable(np.random.normal(scale=np.sqrt(2./512),size=[512,10]).astype(np.float32))
            b3 = tf.Variable(np.zeros(10,dtype=np.float32))
            self.par = [w1,b1,w2,b2,w3,b3]
            self.shape = [tuple(p.get_shape().as_list()) for p in self.par]

class Worker(object):

    def __init__(self,k_id):
        self.k_id = k_id
        self.shape = MASTER_NETWORK.shape
        self.par = MASTER_NETWORK.par

    def update(self):
        self.par = MASTER_NETWORK.par
        
def get_reward(k_id,env, seed, send_end):
    # perturb parameters using seed
    worker = WORKERS[k_id]

    if seed is not None:
        worker.par = add_noise(worker.par,sign(worker.k_id) * SIGMA * get_random(worker.shape,seed))
        # run episode
        
    sess = tf.Session()
    par = worker.par
    x = tf.placeholder(shape=[None,784],dtype=tf.float32)
    y = tf.placeholder(shape=[None,10],dtype=tf.float32)
    scaling = 2**125
    h1 = tf.nn.bias_add(tf.matmul(x, par[0]), par[1]) / scaling
    h2 = tf.nn.bias_add(tf.matmul(h1, par[2]) , par[3] / scaling)   
    o = tf.nn.bias_add(tf.matmul(h2, par[4]), par[5]/ scaling)*scaling
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=o))
    sess.run(tf.global_variables_initializer())
    loss = sess.run(loss,feed_dict={x:env['x'],y:env['y']})
    sess.close()
    ep_r = -loss
    send_end.send(ep_r)

def get_reward_no_noise(k_id,env):
    # perturb parameters using seed
    worker = WORKERS[k_id]
        
    sess = tf.Session()
    par = worker.par
    x = tf.placeholder(shape=[None,784],dtype=tf.float32)
    y = tf.placeholder(shape=[None,10],dtype=tf.float32)
    scaling = 2**125
    h1 = tf.nn.bias_add(tf.matmul(x, par[0]), par[1]) / scaling
    h2 = tf.nn.bias_add(tf.matmul(h1, par[2]) , par[3] / scaling)   
    o = tf.nn.bias_add(tf.matmul(h2, par[4]), par[5]/ scaling)*scaling
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=o))
    sess.run(tf.global_variables_initializer())
    loss = sess.run(loss,feed_dict={x:env['x'],y:env['y']})
    sess.close()
    ep_r = -loss
    return ep_r

def train(optimizer, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID).repeat(2)    # mirrored sampling
    # distribute training in parallel
    jobs = []
    pipe_list = []
    for worker in WORKERS:
        seed = noise_seed[worker.k_id]
        k_id = worker.k_id
        recv_end, send_end = mp.Pipe(False)
        #j = mp.Process(target = get_reward, args=(k_id,env,seed,send_end))
        j = pool.apply_async(get_reward,(k_id,env,seed,send_end))
        jobs.append(j)
        pipe_list.append(recv_end)
    for j in jobs:
        j.get()
    rewards = [x.recv() for x in pipe_list]
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward
    cumulative_update = get_zeros(MASTER_NETWORK.shape)      # initialize update values
    for ui, k_id in enumerate(kids_rank):        
        cumulative_update = cumulate_noise(cumulative_update, utility[ui] * sign(k_id) * get_random(MASTER_NETWORK.shape,noise_seed[k_id]))

    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
    MASTER_NETWORK.par  = add_noise(MASTER_NETWORK.par,gradients)
    # update woker parameters
    for worker in WORKERS:
        worker.update()
        
    return rewards


def get_feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(1000)
    else:
      xs, ys = mnist.test.images, mnist.test.labels
    return {'x': normalize(xs), 'y': ys}

if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    mnist = input_data.read_data_sets(DATA_DIR,
                                    one_hot=True)
    base = N_KID * 2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    MASTER_NETWORK = Linear_Network("global")
    for k_id in range(N_KID*2):
        WORKERS.append(Worker(k_id))
        WORKERS[k_id].update()
    # training
    env = get_feed_dict(True)
    optimizer = SGD(MASTER_NETWORK.shape, LR)
    pool = mp.Pool(processes=N_CORE)
    mar = None      # moving average reward
    for g in range(N_GENERATION):
        t0 = time.time()
        kid_rewards = train(optimizer, utility, pool)

        # test trained net without noise
        net_r = get_reward_no_noise(k_id=0,env=env)
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        env = get_feed_dict(True)
        print(
            'Gen: ', g,
            '| Net_R: %.3f' % mar,
            '| Kid_avg_R: %.3f' % np.mean(kid_rewards),
            '| Gen_T: %.2f' % (time.time() - t0))
