import sys

ENV_ID = 'Pendulum-v0'
ENV_KWARGS = {}

if __name__ == '__main__':

    from learner import *

    make_gae = lambda *args, **kw: AdvantageEstimator(*args, **kw)
    make_act = lambda *args, **kw: Actor(*args, **kw)
    make_crit = lambda *args, **kw: Critic(*args, **kw)
    learner = Learner(
        ENV_ID, 
        ENV_KWARGS,
        make_act, 
        make_crit, 
        make_gae,
        E_episodes=350,
        T_trajectory_len=200,
        N_trajectories_per_episode=10,
        B_batch_size=2000,
        L_sequence_len=100,
        U_updates=80,
        actor_kwargs={"learning_rate": 1e-3},
        critic_kwargs={"learning_rate": 3e-4}
        )

    if sys.argv[1] == "train":
        learner.train()
    elif sys.argv[1] == "enjoy":
        learner.load()
        learner.enjoy(500)
    else:
        print("Invalid command line arguments")