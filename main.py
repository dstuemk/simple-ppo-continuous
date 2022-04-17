import sys
import time
import pathlib
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from multiproc_env import *

ENVIRONMENT = 'Pendulum-v0'
STATE_DIMS = 3
ACTION_DIMS = 1

class AdvantageEstimator:
    def __init__(self):
        self.lmda = 0.95
        self.gamma = 0.99
        self.normalize = False

    def get_advantage(self, values, rewards):
        """
        Input:
            values - each row represents a state sequence of an episode
            rewards - each row represents a reward sequence of an episode
        Output:
            adv - each row represents advantage values for an episode
            delta - each row represents td errors for an episode
        """
        values = tf.cast(values, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        toeplitz_c = tf.concat([[1.0], tf.zeros(rewards.shape[1]-1)],axis=0)
        toeplitz_r = tf.pow(self.lmda*self.gamma, 
            tf.linspace(0.0, rewards.shape[1]-1.0, rewards.shape[1]))
        toeplitz_m = tf.linalg.LinearOperatorToeplitz(toeplitz_c, toeplitz_r)
        delta = rewards + self.gamma*values[:,1:] - values[:,0:-1]
        adv = toeplitz_m.matmul(delta)
        if self.normalize:
            adv = adv - tf.reduce_mean(adv)
            adv = adv / tf.math.reduce_std(adv)
        return adv, delta

class Critic:
    def __init__(self):
        self.learning_rate = 1e-3
        inp = tf.keras.Input(shape=(None,STATE_DIMS))
        x = tf.keras.layers.Dense(32)(inp)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Activation('relu')(x)
        outp = tf.keras.layers.Dense(1)(x)
        self.model = tf.keras.Model(inp, outp, name="critic")

    def get_value(self, states):
        return self.model(states, training=True)

class Actor:
    def __init__(self):
        self.sigma = 2e-1
        self.learning_rate = 3e-4
        inp = tf.keras.Input(shape=(None,STATE_DIMS))
        x = tf.keras.layers.Dense(32)(inp)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(32)(x)
        x = tf.keras.layers.Activation('relu')(x)
        outp_mean = tf.keras.layers.Dense(ACTION_DIMS)(x)
        self.model = tf.keras.Model(inp, outp_mean, name="actor")

    def act(self, states):
        """
        Input:
            states - each row in this matrix represents a state
        Output:
            distribution - distribution of actions
        """
        mean = self.model(states)
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.ones_like(mean)*self.sigma)


class Learner:
    def __init__(self, actor, critic, advantage_estimator):
        self.weights_folder = "./weights"
        self.c_critic = 0.5
        self.c_entropy = 0.0
        self.epsilon = 0.2
        self.actor = actor
        self.critic = critic
        self.advantage_estimator = advantage_estimator

    def save(self):
        pathlib.Path(self.weights_folder).mkdir(parents=True, exist_ok=True)
        self.actor.model.save_weights(self.weights_folder + "/actor.ckpt")
        self.critic.model.save_weights(self.weights_folder + "/critic.ckpt")

    def load(self):
        self.actor.model.load_weights(self.weights_folder + "/actor.ckpt")
        self.critic.model.load_weights(self.weights_folder + "/critic.ckpt")
    
    def enjoy(self, T_timesteps = 200):
        env = gym.make(ENVIRONMENT).env
        state = np.reshape(env.reset(), (1,1,-1))
        for step_idx in range(T_timesteps):
            action_distr = self.actor.act(state)
            action = action_distr.sample()
            state_next,reward,done,_ = env.step(action)
            state = np.reshape(state_next, (1,1,-1))
            env.render()
        env.close()

    def train(self, I_iterations, N_trajectories, T_timesteps, K_epochs):
        multiproc_env = MultiprocEnv(ENVIRONMENT, N_trajectories)
        for iter_idx in range(I_iterations):
            # Sample trajectories with current policy
            probs_all = np.zeros([N_trajectories, T_timesteps, 1])
            rewards_all = np.zeros([N_trajectories, T_timesteps, 1])
            actions_all = np.zeros([N_trajectories, T_timesteps, ACTION_DIMS])
            states_all = np.zeros([N_trajectories, T_timesteps + 1, STATE_DIMS])
            
            t_start  = time.process_time()
            states = multiproc_env.reset()
            for step_idx in range(T_timesteps):
                action_distr = self.actor.act(np.reshape(states, (-1,1,STATE_DIMS)))
                action = action_distr.sample()
                states_next,rewards,dones = multiproc_env.step(action)
                probs_all[:, step_idx, :] = action_distr.prob(action)
                rewards_all[:, step_idx, :] = np.reshape(rewards, rewards_all[:, step_idx, :].shape)
                actions_all[:, step_idx, :] = np.reshape(action, actions_all[:, step_idx, :].shape)
                states_all[:, step_idx, :] = np.reshape(states, states_all[:, step_idx, :].shape)
                states = states_next
            states_all[:, step_idx+1, :] = np.reshape(states, states_all[:, step_idx+1, :].shape)
            t_elapsed = time.process_time() - t_start

            print( "\n")
            print( ".-----------------------------------------")
            print(f"| Rollout: {iter_idx}                 ")
            print( "|-----------------------------------------")
            print(f"| Elapsed:     {t_elapsed} sec.           ")
            print(f"| Avg. Reward: {np.mean(rewards_all):.3f} ")
            print(f"| Std. Reward: {np.std(rewards_all):.3f}  ")
            print( "'-----------------------------------------")
            
            print( "\n")
            print( ".-----------------------------------------")
            print(f"| Improve: {iter_idx}                     ")
            print( "|-----------------------------------------")
            optimizer_critic = tf.keras.optimizers.Adam(self.critic.learning_rate)
            optimizer_actor = tf.keras.optimizers.Adam(self.actor.learning_rate)
            for epoch_idx in range(K_epochs):
                # Improve critic
                with tf.GradientTape() as tape:
                    values_all = self.critic.get_value(states_all)
                    advantages,td_errors = self.advantage_estimator.get_advantage(values_all, rewards_all)
                    critic_loss = self.get_critic_loss(td_errors)
                critic_gradients = tape.gradient(critic_loss, self.critic.model.trainable_variables)
                optimizer_critic.apply_gradients(zip(critic_gradients, self.critic.model.trainable_variables))

                # Improve actor
                with tf.GradientTape() as tape:
                    action_distr = self.actor.act(states_all[:,:-1,:])
                    probs_new = action_distr.prob(actions_all)
                    probs_new = tf.reshape(probs_new, probs_all.shape)
                    actor_loss = self.get_actor_loss(
                        advantages, probs_new, probs_all, action_distr.entropy())
                actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)
                optimizer_actor.apply_gradients(zip(actor_gradients, self.actor.model.trainable_variables))

                if epoch_idx % int(K_epochs / 5) == 0:
                    print(f"| Epoch: {epoch_idx}, Actor-Loss: {actor_loss:.3f}, Critic-Loss: {critic_loss:.3f}")            
            print( "'-----------------------------------------")

            # Show what's learned so far every 10th iteration
            if iter_idx % 10 == 0:
                self.enjoy()

    def get_critic_loss(self, td_errors):
        # Calculate value function loss
        loss = tf.reduce_mean(tf.pow(td_errors, 2.0))
        return self.c_critic*loss

    def get_actor_loss(self, advantages, prob, prob_old, entropy):
        # Calculate ratio between new and old action probabilities
        r = prob / prob_old
        # Calculate clipped loss
        clip = tf.clip_by_value(r, 1.0 - self.epsilon, 1.0 + self.epsilon)
        loss_clip = -tf.reduce_mean(tf.minimum(r*advantages, clip*advantages))
        # Calculate entropy loss
        loss_entropy = -tf.reduce_mean(entropy)
        # Combine loss
        actor_loss = self.c_entropy*loss_entropy + loss_clip
        return actor_loss

if __name__ == '__main__':
    gae = AdvantageEstimator()
    actor = Actor()
    critic = Critic()
    learner = Learner(actor, critic, gae)

    if sys.argv[1] == "train":
        learner.train(350, 10, 200, 100)
        learner.save()
    elif sys.argv[1] == "enjoy":
        learner.load()
        learner.enjoy()
    else:
        print("Invalid command line arguments")