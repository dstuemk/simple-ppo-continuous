from asyncio import queues
import gym
import multiprocessing

class MultiprocEnv:
    def __init__(self, env_id, N_workers):
        self.N_workers = N_workers
        self.queues_in = []
        self.queues_out = []
        self.procs = []
        for _ in range(N_workers):
            q_in = multiprocessing.Queue()
            q_out = multiprocessing.Queue()
            self.queues_in.append(q_in)
            self.queues_out.append(q_out)
            p = multiprocessing.Process(target=MultiprocEnv._proc_worker, args=(q_out,q_in,env_id,))
            self.procs.append(p)
            p.start()
    
    def __del__(self):
        for q in self.queues_out:
            q.put({'cmd': 'exit'})
        for p in self.procs:
            p.join()

    def _proc_worker(queue_in, queue_out, env_id):
        env = gym.make(env_id).env
        term = False
        while not term:
            cmd_obj = queue_in.get()
            if cmd_obj['cmd'] == 'reset':
                state  = env.reset()
                queue_out.put(state)
            elif cmd_obj['cmd'] == 'step':
                action = cmd_obj['action']
                state_next,reward,done,_ = env.step(action)
                queue_out.put((state_next,reward,done))
            elif cmd_obj['cmd'] == 'exit':
                term = True
            else:
                print("Invalid command")


    def reset(self):
        states = []
        for q_idx in range(self.N_workers):
            self.queues_out[q_idx].put({'cmd': 'reset'})
            states.append(self.queues_in[q_idx].get())
        return states

    def step(self, actions):
        states_next = [] 
        rewards = [] 
        dones = []
        for q_idx in range(self.N_workers):
            q = self.queues_out[q_idx]
            q.put({'cmd': 'step', 'action': actions[q_idx].numpy()})
        for q in self.queues_in:
            state_next,reward,done = q.get()
            states_next.append(state_next)
            rewards.append(reward)
            dones.append(done)
        return states_next,rewards,dones

