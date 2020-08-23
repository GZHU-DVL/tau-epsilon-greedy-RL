'''

This file contains only the code for the DDQN (τ,ε)-greedy algorithm, and the complete code will be published in the future.

'''

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))
Transition_chain = namedtuple('Transition_chain', ('net_input', 'next_net_input', 'action', 'reward'))

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)

class ReplayMemory(object):
    def __init__(self, capacity, window, input_length): 
        self.__window = window
        self.__input_length = input_length
        self.__capacity = capacity     
        self.__memory = []      
        self.__memory_chain = []       

    def __len__(self):
        return len(self.__memory_chain)

    def reset(self):
        self.__memory = []
        self.__memory_chain = []

    def get_net_input(self, state):
        memory_length = len(self.__memory)
        if (memory_length <= self.__window):
            return None
        else:
            net_input = []
            for i in range(memory_length - self.__window, memory_length):
                net_input = net_input + self.__memory[i].state.tolist()  
                net_input.append(self.__memory[i].action)  
            net_input = net_input + state.tolist() 
            net_input = np.array(net_input).reshape(-1) 
            return net_input

    def push(self, state, next_state, action, R):
        net_input = self.get_net_input(state)
        self.__memory.append(Transition(state, next_state, action, R))
        if (len(self.__memory) > self.__capacity):
            self.__memory.pop(0)
        next_net_input = self.get_net_input(next_state)
        if ((None is not net_input) and (None is not next_net_input)):
            self.__memory_chain.append(Transition_chain(net_input, next_net_input, action, R))
            if (len(self.__memory_chain) > self.__capacity):
                self.__memory_chain.pop(0)
        return net_input, next_net_input

    def sample(self, batch_size):
        return random.sample(self.__memory_chain, batch_size)


class DQN_model(nn.Module):
    def __init__(self, input_length, num_action):
        super(DQN_model, self).__init__()
        self.num_action = num_action 
        self.cov1 = nn.Conv1d(1, 20, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.cov2 = nn.Conv1d(20, 40, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(40)
        self.fc1 = nn.Linear(40 * (input_length + 1 - 3 + 1 - 2), 180)
        self.fc2 = nn.Linear(180, self.num_action)

    def forward(self, x):
        x = x.view(x.size(0), -1, x.size(-1))  
        x = F.leaky_relu(self.bn1(self.cov1(x)))
        x = F.leaky_relu(self.bn2(self.cov2(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

    def reset(self):
        self.cov1.reset_parameters()
        self.bn1.reset_parameters()
        self.cov2.reset_parameters()
        self.bn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2 = nn.Linear(180, self.num_action)  
        self.fc2.reset_parameters()

class DQN():
    def __init__(self, input_length, Num_action, memory_capacity, window, GAMMA=0.6, EPS_START=0.9, EPS_END=0.1, Anneal_step=100, learning_begin=50):

        self.Num_action = Num_action
        self.memory = ReplayMemory(memory_capacity, window, input_length)
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.Anneal_step = Anneal_step
        self.learning_begin = learning_begin
        self.POSAI = 0.002
        if use_cuda:
            self.model_0 = DQN_model((input_length + 1) * window + input_length, Num_action)
            self.model = torch.nn.DataParallel(self.model_0.cuda())
        else:
            self.model = DQN_model((input_length + 1) * window + input_length, Num_action)
            self.model2 = DQN_model((input_length + 1) * window + input_length, Num_action)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-7)
        self.steps_done = 0 

    def reset(self):
        self.memory.reset()
        self.steps_done = 0
        if use_cuda:
            self.model_0.reset()
            self.model = torch.nn.DataParallel(self.model_0.cuda())
        else:
            self.model.reset()
            self.model2.reset()
            
    def select_action_L(self, state,SINR_A,SINR_C):

        if(SINR_C - SINR_A>0) :
            self.POSAI = 1-(stats.norm.pdf(SINR_C - SINR_A, loc=0, scale=0.8))
        else:
            self.POSAI = 1-(stats.norm.pdf(SINR_C - SINR_A, loc=0, scale=0.85))
        net_input = self.memory.get_net_input(state)
        if (None is not net_input):  
            sample = random.random()
            eps_temp = self.EPS_START - (self.steps_done - self.learning_begin) * (self.EPS_START - self.EPS_END) / self.Anneal_step
            eps_temp = min(self.EPS_START, eps_temp)
            eps_threshold = max(self.EPS_END, eps_temp)
            eps_threshold = 0.01
            self.steps_done += 1
            if sample > eps_threshold + self.POSAI:
                xx=Variable(torch.from_numpy(net_input.reshape(1, -1)).float())
                _, action_ind = self.model(Variable(torch.from_numpy(net_input.reshape(1, -1)).float())).data.max(dim=1)
                return int(action_ind.item())
            elif   sample <= eps_threshold :
                self.last_action_ind = int(random.randrange(self.Num_action))
                return self.last_action_ind
            else:
                return self.last_action_ind
        else:
            self.last_action_ind = int(random.randrange(self.Num_action))
            return self.last_action_ind

    def select_action(self, state):
        net_input = self.memory.get_net_input(state)
        if (None is not net_input): 
            sample = random.random()
            eps_temp = self.EPS_START - (self.steps_done - self.learning_begin) * (self.EPS_START - self.EPS_END) / self.Anneal_step
            eps_temp = min(self.EPS_START, eps_temp)
            eps_threshold = max(self.EPS_END, eps_temp)
            self.steps_done += 1
            if sample > eps_threshold:
                xx=Variable(torch.from_numpy(net_input.reshape(1, -1)).float())
                _, action_ind = self.model(Variable(torch.from_numpy(net_input.reshape(1, -1)).float())).data.max(dim=1)
                return int(action_ind.item())
            else:
                return int(random.randrange(self.Num_action))
        else:
            return int(random.randrange(self.Num_action))
        
        
    def optimize_model(self, state, next_state, action, R, BATCH_SIZE=10):
        net_input, next_net_input = self.memory.push(state, next_state, action,R) 
        if len(self.memory) < BATCH_SIZE:  
            return

        experience = self.memory.sample(BATCH_SIZE)
        batch = Transition_chain(*zip(*experience))
        next_states = Variable(torch.cat([FloatTensor(batch.next_net_input)]))
        state_batch = Variable(torch.cat([FloatTensor(batch.net_input)]))
        action_batch = Variable(torch.cat([LongTensor(batch.action)]).view(-1, 1))
        reward_batch = Variable(torch.cat([FloatTensor(batch.reward)]))
        state_action_values = self.model(state_batch).gather(1, action_batch)
        max_action_indexes = self.model(state_batch).detach().argmax(1)
        next_state_action_values1 = self.model2(next_states).gather(1, max_action_indexes.unsqueeze(1)).squeeze(1)
        expected_state_action_values =  reward_batch + (next_state_action_values1 * self.GAMMA) 
        expected_state_action_values = expected_state_action_values.unsqueeze(1) 
        loss2 = F.smooth_l1_loss(state_action_values, expected_state_action_values) 
        self.optimizer.zero_grad()
        loss2.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
    def update_target_q_network(self, step,tau):
        for target_param, local_param in zip(self.model2.parameters(), self.model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        print('step '+str(step) +', target Q network params replaced!')
    