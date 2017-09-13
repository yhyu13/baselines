from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
from baselines.common.distributions import make_pdtype
import numpy as np
from helper import dlrelu

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var):
        #assert isinstance(ob_space, gym.spaces.Box)

        #self.pdtype = pdtype = tf.contrib.distributions.Normal()#make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + [ob_space])

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(0.01))[:,0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))   
        mean = dlrelu(U.dense(last_out, ac_space, "polfinal_mean", U.normc_initializer(0.01)))
        
        if gaussian_fixed_var: #and isinstance(ac_space, gym.spaces.Box):    
            logstd = tf.Variable(initial_value = np.ones((1,ac_space)).astype(np.float32)*-3.)
            self.logstd = tf.get_variable(name="logstd",initializer=logstd.initialized_value())
            self.pd = tf.contrib.distributions.Normal(mean,tf.exp(self.logstd))
        else:
            self.var = tf.nn.softplus(U.dense(last_out, ac_space, "polfinal_var", U.normc_initializer(0.01)))
            self.pd = tf.contrib.distributions.Normal(mean,tf.sqrt(self.var))
            
        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

