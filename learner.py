import time
import math
import gym
import pathlib
import numpy as np
import tensorflow as tf

# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')

import tensorflow_probability as tfp

tfd = tfp.distributions

from datetime import datetime
from progress_writer import *

class AdvantageEstimator:
    def __init__(self):
        self.lmda = 0.95
        self.gamma = 0.99

    def get_advantage(self, values, rewards):
        values = tf.cast(values, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        toeplitz_c = tf.concat([[1.0], tf.zeros(rewards.shape[1]-1)],axis=0)
        toeplitz_r = tf.pow(self.lmda*self.gamma, 
            tf.linspace(0.0, rewards.shape[1]-1.0, rewards.shape[1]))
        toeplitz_m = tf.linalg.LinearOperatorToeplitz(toeplitz_c, toeplitz_r)
        delta = rewards + self.gamma*values[:,1:] - values[:,0:-1]
        adv = toeplitz_m.matmul(delta)
        return adv

class Critic:
    def __init__(self, observation_dims, learning_rate=1e-3):
        self.learning_rate = learning_rate
        initializer = tf.keras.initializers.glorot_uniform()
        inp = tf.keras.Input(shape=(None,observation_dims))
        x = tf.keras.layers.Dense(50, kernel_initializer=initializer)(inp)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(50, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Activation('relu')(x)
        outp = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)
        self.model = tf.keras.Model(inp, outp, name="critic")
    
    def get_value(self, obs):
        return self.model(obs, training=True)

class Actor:
    def __init__(self, observation_dims, action_dims, sigma=2e-1, learning_rate=3e-4):
        self.sigma = sigma
        self.learning_rate = learning_rate
        initializer = tf.keras.initializers.glorot_uniform()
        inp = tf.keras.Input(shape=(None,observation_dims))
        x = tf.keras.layers.Dense(50, kernel_initializer=initializer)(inp)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(50, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Activation('relu')(x)
        outp_mean = tf.keras.layers.Dense(action_dims, kernel_initializer=initializer)(x)
        self.model = tf.keras.Model(inp, outp_mean, name="actor")

    def act(self, obs):
        """
        Input:
            obs - each row in this matrix represents an observation
        Output:
            distribution - distribution of actions
        """
        mean = self.model(obs)
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.ones_like(mean)*self.sigma)

class Learner:
    def __init__(
        self, 
        env_id,
        env_kwargs,
        make_actor, 
        make_critic, 
        make_advantage_estimator,
        actor_kwargs={},
        critic_kwargs={},
        advantage_estimator_kwargs={},
        c_critic=0.5,
        c_entropy=0.0,
        epsilon=0.2,
        actor_sigma_start=0.5,
        actor_sigma_end=0.01,
        E_episodes=1000,
        B_batch_size=5000,
        N_trajectories_per_episode=10,
        T_trajectory_len=1000,
        L_sequence_len=32,
        U_updates=80,
        progress_folder='./progress'):
        # General info
        self.timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
        self.progress_folder = progress_folder
        self.progress_writer = ProgressWriter(folder=self.progress_folder)
        self.highest_avg_reward = -float("inf")
        # Learning parameters
        self.E_episodes = E_episodes
        self.N_trajectories_per_episode = N_trajectories_per_episode
        self.T_trajectory_len = T_trajectory_len
        self.B_batch_size = B_batch_size
        self.U_updates = U_updates
        self.L_sequence_len = L_sequence_len
        # Hyper parameters
        self.actor_sigma_start = actor_sigma_start
        self.actor_sigma_end = actor_sigma_end
        self.epsilon = epsilon
        self.c_entropy = c_entropy
        self.c_critic = c_critic
        # Environment
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.env = gym.make(env_id, **env_kwargs)
        self._obs_dims = self.env.observation_space.shape[0]
        self._action_dims = self.env.action_space.shape[0]
        # Components
        self.advantage_estimator = make_advantage_estimator(**advantage_estimator_kwargs)
        self.critic = make_critic(self._obs_dims, **critic_kwargs)
        self.actor = make_actor(self._obs_dims, self._action_dims, **actor_kwargs)
    
    def _make_rollouts(self, vector_env):
        # Rollout memory
        dones_all = []
        probs_all = []
        rewards_all = []
        actions_all = []
        obs_all = []
        values_all = []
        # Run rollouts
        reshape_3d = lambda x: np.reshape(x,(x.shape[0], 1, -1))
        obs = vector_env.reset()
        for step_idx in range(self.T_trajectory_len):
            value = self.critic.model(reshape_3d(obs))
            action_distr = self.actor.act(reshape_3d(obs))
            action = action_distr.sample()
            obs_next,rewards,dones,infos = vector_env.step(action.numpy()[:,0,:])
            prob = action_distr.prob(action)
            probs_all.append(reshape_3d(prob))
            actions_all.append(reshape_3d(action))
            values_all.append(reshape_3d(value))
            obs_all.append(reshape_3d(obs))
            rewards_all.append(reshape_3d(rewards))
            dones_all.append(reshape_3d(dones))
            obs = obs_next
        # Return rollout object
        return { 
            "done":   np.concatenate(dones_all,   axis=1), 
            "prob":   np.concatenate(probs_all,   axis=1), 
            "action": np.concatenate(actions_all, axis=1), 
            "obs":    np.concatenate(obs_all,     axis=1), 
            "reward": np.concatenate(rewards_all, axis=1),
            "value":  np.concatenate(values_all,  axis=1)
        }

    def _rollouts2trajectories(self, rollouts:dict):
        info = {}
        dones = rollouts["done"]
        dones[:,-1,0] = True    # Mark end of rollout, too
        traj_start_indices = np.flatnonzero([1, *dones.flatten()])
        traj_lens = traj_start_indices[1:] - traj_start_indices[:-1]
        info["trajlen/max"] = np.max(traj_lens)
        info["trajlen/min"] = np.min(traj_lens)
        info["trajlen/avg"] = np.mean(traj_lens)
        info["trajlen/std"] = np.std(traj_lens)
        # Concatenate rollouts along 2nd axis into one big rollout
        big_rollout = dict((k, np.concatenate(v,axis=0)) for k,v in rollouts.items())
        info["reward/avg"] = np.mean(big_rollout["reward"])
        info["reward/std"] = np.std(big_rollout["reward"])
        # Split up rollout into consecutive trajectories
        trajs = []
        for traj_start_idx,traj_len in zip(traj_start_indices[:-1], traj_lens):
            traj_range = range(traj_start_idx, traj_start_idx+traj_len)
            traj_dict = dict((k, v[traj_range]) for k,v in big_rollout.items())
            traj_dict["traj_len"] = traj_len
            trajs.append(traj_dict)
        return trajs,info
    
    def _trajectories2sequences(self, trajectories:list, seq_len=32):
        seqs = []
        for traj in trajectories:
            traj_len = traj.pop("traj_len", 0)
            for seq_idx in range(math.ceil(traj_len / seq_len)):
                seq_start = seq_idx*seq_len
                seq_end = min(traj_len, seq_start + seq_len)
                seq_pad = seq_len - seq_end + seq_start
                seq_range = range(seq_start, seq_end)
                seq = dict((k, np.pad(v[seq_range], [[0,seq_pad],[0,0]])) for k,v in traj.items())
                seq["mask"] = np.array([
                    *[True]*(seq_len-seq_pad), 
                    *[False]*seq_pad])[:,np.newaxis]
                seqs.append(seq)
        return seqs

    def load(self):
        ckpt_path = pathlib.Path(self.progress_folder) / "checkpoints"
        self.actor.model.load_weights(str(ckpt_path / "actor.ckpt"))
        self.critic.model.load_weights(str(ckpt_path / "critic.ckpt"))

    def train(self):
        vector_env = gym.vector.make(
            self.env_id,self.N_trajectories_per_episode,**self.env_kwargs)
        t_train_start = time.process_time()
        for ep_idx in range(self.E_episodes):

            # Rollouts
            t_rollout_start = time.process_time()
            rollouts = self._make_rollouts(vector_env)
            trajs,trajs_info = self._rollouts2trajectories(rollouts)
            seqs = self._trajectories2sequences(trajs, self.L_sequence_len)
            t_rollouts_elapsed = time.process_time() - t_rollout_start
            print( "\n")
            print( ".-----------------------------------------")
            print(f"| Rollout: {ep_idx}                       ")
            print( "|-----------------------------------------")
            print(f"| Elapsed:     {t_rollouts_elapsed} sec.  ")
            print(f"| Tot. Trajs:  {len(trajs)}               ")
            print( "| " + "\n| ".join(
                [str(k) + ": " + f"{v:.3g}" for k,v in trajs_info.items()]))
            print( "'-----------------------------------------")
            print( "\n")

            # Save model if reward has improved
            if trajs_info["reward/avg"] > self.highest_avg_reward:
                self.highest_avg_reward = trajs_info["reward/avg"]
                print("Save model ...\n")
                ckpt_path = pathlib.Path(self.progress_folder) / "checkpoints"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                self.actor.model.save_weights(str(ckpt_path / "actor.ckpt"))
                self.critic.model.save_weights(str(ckpt_path / "critic.ckpt"))
            
            # Experimental
            print( "\n")
            print( ".-----------------------------------------")
            print(f"| Improve: {ep_idx}                       ")
            print( "|-----------------------------------------")
            t_improve_start = time.process_time()
            optimizer_critic = tf.keras.optimizers.Adam(self.critic.learning_rate)
            optimizer_actor = tf.keras.optimizers.Adam(self.actor.learning_rate)
            critic_losses = []
            actor_losses = []
            total_batches = math.ceil(
                self.T_trajectory_len*self.N_trajectories_per_episode / self.B_batch_size)
            seqs_per_batch = math.ceil(len(seqs) / total_batches)
            for upd_idx in range(self.U_updates):
                for batch_idx in range(total_batches):
                    seqs_start_idx = batch_idx*seqs_per_batch
                    seqs_end_idx = min(len(seqs), seqs_start_idx+seqs_per_batch)
                    subseqs = seqs[seqs_start_idx:seqs_end_idx]
                    subseqs_dict = {}
                    for k in subseqs[0].keys():
                        subseqs_dict[k] = np.array([v[k] for v in subseqs])
                    
                    # Compute losses and gradients
                    forward_info = self._compute_forward(subseqs_dict)
                    
                    optimizer_critic.apply_gradients(zip(
                        forward_info["criticgrads"], 
                        self.critic.model.trainable_variables))
                    critic_losses.append(forward_info["criticloss"].numpy())
                    
                    optimizer_actor.apply_gradients(zip(
                        forward_info["actorgrads"], 
                        self.actor.model.trainable_variables))
                    actor_losses.append(forward_info["actorloss"].numpy())
            
            actor_loss_avg = np.mean(actor_losses)
            actor_loss_std = np.std(actor_losses)
            critic_loss_avg = np.mean(critic_losses)
            critic_loss_std = np.std(critic_losses)
            t_improve_elapsed = time.process_time() - t_improve_start            
            print(f"| Elapsed:     {t_improve_elapsed} sec.   ")
            print(f"| Avg. Loss (Act / Crit): {actor_loss_avg:.3f} / {critic_loss_avg:.3f} ")
            print(f"| Std. Loss (Act / Crit): {actor_loss_std:.3f} / {critic_loss_std:.3f} ")
            print( "'-----------------------------------------")

             # Write progress
            progress_entry = {
                "ep": ep_idx,
                "t": f"{(time.process_time() - t_train_start):.6f}",
                **trajs_info,
                "actloss/avg": f"{np.mean(actor_losses):.6f}",
                "actloss/std": f"{np.std(actor_losses):.6f}",
                "critloss/avg": f"{np.mean(critic_losses):.6f}",
                "critloss/std": f"{np.std(critic_losses):.6f}"
            }
            self.progress_writer.add_entry(progress_entry)
    
    def _compute_forward(self, subseqs_dict):
        obs_seq = subseqs_dict["obs"]
        values_seq = subseqs_dict["value"]
        masks_seq = subseqs_dict["mask"][:,:-1,:]
        rewards_seq = subseqs_dict["reward"][:,:-1,:]
        advantages = self.advantage_estimator.get_advantage(
            values_seq, rewards_seq)
        sampled_return = values_seq[:,:-1,:] + advantages

        # Improve critic
        with tf.GradientTape() as tape:
            values = self.critic.get_value(obs_seq)
            critic_loss = self.get_critic_loss(
                values[:,:-1,:] - sampled_return, masks_seq)
        critic_gradients = tape.gradient(critic_loss, self.critic.model.trainable_variables)

        # Improve actor
        actions_seq = subseqs_dict["action"][:,:-1,:]
        probs_seq = subseqs_dict["prob"][:,:-1,:]
        with tf.GradientTape() as tape:
            action_distr = self.actor.act(obs_seq[:,:-1,:])
            probs_new = action_distr.prob(actions_seq)
            probs_new = tf.reshape(probs_new, probs_seq.shape)
            actor_loss = self.get_actor_loss(
                advantages, probs_new, probs_seq, action_distr.entropy(),
                masks_seq)
        actor_gradients = tape.gradient(actor_loss, self.actor.model.trainable_variables)

        return {
            "actorgrads": actor_gradients, 
            "actorloss": actor_loss,
            "criticgrads": critic_gradients,
            "criticloss": critic_loss }

    def get_critic_loss(self, errs, mask):
        errs = tf.ragged.boolean_mask(errs, mask)
        # Calculate value function loss
        loss = tf.reduce_mean(tf.pow(errs, 2.0))
        return self.c_critic*loss

    def get_actor_loss(self, advantages, prob, prob_old, entropy, mask):
        advantages = tf.ragged.boolean_mask(advantages, mask)
        prob = tf.ragged.boolean_mask(prob, mask)
        prob_old = tf.ragged.boolean_mask(prob_old, mask)
        entropy = tf.ragged.boolean_mask(entropy, mask[:,:,0])
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
    
    def enjoy(self, T_timesteps = 500):
        env = gym.make(self.env_id, **self.env_kwargs)
        obs = env.reset()[np.newaxis,:]
        running = True
        step_ctr = 1
        while running:
            action_distr = self.actor.act(obs)
            action = action_distr.sample()
            obs_next,reward,done,_ = env.step(action[0])
            if done:
                obs_next = env.reset()
            obs = obs_next[np.newaxis,:]
            env.render()
            if step_ctr == T_timesteps:
                running = False
            if T_timesteps > 0:
                step_ctr = step_ctr + 1
        env.close()


