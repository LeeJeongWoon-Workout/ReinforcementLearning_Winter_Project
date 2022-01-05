#Replay Buffer PPO
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import tqdm
import time

#Hyperparameters
learning_rate  = 0.0003
gamma           = 0.9
lmbda           = 0.9
eps_clip        = 0.2
K_epoch         = 10
rollout_len    = 3
buffer_size    = 30
minibatch_size = 64

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, prob_lst,done_mask_lst = [], [], [], [], [],[]

        for transition in mini_batch:
            s, a, r, s_prime, prob,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
            prob_lst.append(prob)
        
        s_batch=torch.tensor(s_lst,dtype=torch.float)
        a_batch=torch.tensor(a_lst,dtype=torch.float)
        r_batch=torch.tensor(r_lst,dtype=torch.float)
        s_prime_batch=torch.tensor(s_prime_lst,dtype=torch.float)
        prob_batch=torch.tensor(prob_lst,dtype=torch.float)
        done_batch=torch.tensor(done_mask_lst,dtype=torch.float)
        
        return s_batch,a_batch,r_batch,s_prime_batch,prob_batch,done_batch
    
    def size(self):
        return len(self.buffer)
class PPO(nn.Module):
    
    def __init__(self):
        super(PPO,self).__init__()
        self.fc1=nn.Linear(3,128)
        self.fc_mu=nn.Linear(128,1)
        self.fc_std=nn.Linear(128,1)
        self.fc_v=nn.Linear(128,1)
    
    def pi(self,x,softmax_dim=0):
        x=F.relu(self.fc1(x))
        mu=2*torch.tanh(self.fc_mu(x))
        
        std=self.fc_std(x)
        std=F.softplus(std)
        
        return mu,std
    
    def v(self,x):
        
        x=F.relu(self.fc1(x))
        v=self.fc_v(x)
        return v
    

def train(ppo,ppo_target,ppo_optimizer,memory):
    
    s,a,r,s_prime,prob_log,done=memory.sample(64)
    
    td_target=r+gamma*ppo_target.v(s_prime)*done
    
    delta=td_target-ppo.v(s)
    delta=delta.detach().numpy()
    
    #advantage
    advantage=0
    advantage_lst=list()
    
    for delta_t in delta[::-1]:
        advantage=gamma*lmbda*advantage+delta_t[0]
        advantage_lst.append(advantage)
        
    advantage_lst.reverse()
    advantage=torch.tensor(advantage_lst,dtype=torch.float)
    
    mu,std=ppo.pi(s,softmax_dim=1)
    dist=Normal(mu,std)
    log_prob=dist.log_prob(a)
    
    ratio=torch.exp(log_prob-prob_log)
    
    surr1=ratio*advantage
    surr2=torch.clamp(ratio,1-eps_clip,1+eps_clip)*advantage
    
    loss=-torch.min(surr1,surr2)+F.smooth_l1_loss(ppo.v(s),td_target.detach())
    
    ppo_optimizer.zero_grad()
    loss.mean().backward()
    ppo_optimizer.step()
    
    
def soft_update(net,net_target):
    
    for param,param_target in zip(net.parameters(),net_target.parameters()):
        param_target.data.copy_(tau*param.data+(1.-tau)*param_target)
        
        
env=gym.make('Pendulum-v1')
model,target=PPO(),PPO()

target.load_state_dict(model.state_dict())
reward_list4=list()
memory=ReplayBuffer()
score=0.0
print_interval=10
optimizer=optim.Adam(model.parameters(),lr=learning_rate)


for n_epi in tqdm.trange(400):
    s=env.reset()
    done=False
    
    while True:
        mu,std=model.pi(torch.from_numpy(s).float())
        dist=Normal(mu,std)
        a=dist.sample()
        log_prob=dist.log_prob(a)
        s_prime,r,done,_=env.step([a.item()])
        
        memory.put((s,a,r/100.,s_prime,log_prob,done))
        
        s=s_prime
        score+=r
        if done:
            break
            
    if memory.size()>1000:
        for i in range(20):
            train1(model,target,optimizer,memory)
            soft_update1(model,target)
            
        
    if n_epi%10==0 and n_epi!=0:
        reward_list4.append(score/10)
        score=0
env.close()
