'''

This file contains only the code for the PDDQN (τ,ε)-greedy algorithm, and the complete code will be published in the future.

'''

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))
Transition_chain = namedtuple('Transition_chain', ('net_input', 'next_net_input', 'action', 'reward'))

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)

class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity  
        self.tree = np.zeros(2 * capacity - 1)   
        self.data = np.zeros(capacity, dtype=object)  

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p) 
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:  
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     
            cl_idx = 2 * parent_idx + 1        
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):       
                leaf_idx = parent_idx
                break
            else:       
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  


class ReplayMemory(object):
    def __init__(self, capacity, window, input_length):
        self.__window = window
        self.__input_length = input_length
        self.__capacity = capacity     
        self.__memory = []      
        self.__memory_chain = [] 
        self.tree = SumTree(capacity)
        self.epsilon = 0.01  
        self.alpha = 0.6  
        self.beta = 0.4  
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  
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
            transition = Transition_chain(net_input, next_net_input, action, R)
            self.store(transition)
            if (len(self.__memory_chain) > self.__capacity):              
                self.__memory_chain.pop(0)
        return net_input, next_net_input

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        memory_chain = []
        b_idx  = np.empty((n,), dtype=np.int32)
        ISWeights = np.empty((n, 1))
        pri_seg = self.tree.total_p / n       
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p    
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i] = idx
            memory_chain.append(data)
        return b_idx, memory_chain, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors, 1)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

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
        self.replay_total = 0
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = input_length
        self.action_dim = Num_action
        self.POSAI = 0.002

        if use_cuda:
            self.model_0 = DQN_model((input_length + 1) * window + input_length, Num_action)
            self.model = torch.nn.DataParallel(self.model_0.cuda())
        else:
            self.model = DQN_model((input_length + 1) * window + input_length, Num_action)
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
            self.POSAI= stats.norm.pdf(SINR_C - SINR_A, loc=0, scale=85)
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
                self.last_action_ind = int(action_ind.item())
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
        tree_idx, minibatch, ISWeights = self.memory.sample(BATCH_SIZE)     
        batch = Transition_chain(*zip(*minibatch))     
        next_states = Variable(torch.cat([FloatTensor(batch.next_net_input)]))
        state_batch = Variable(torch.cat([FloatTensor(batch.net_input)]))
        action_batch = Variable(torch.cat([LongTensor(batch.action)]).view(-1, 1))
        reward_batch = Variable(torch.cat([FloatTensor(batch.reward)]))
        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_action_values = self.model2(next_states).max(1)[0]
        expected_state_action_values =  reward_batch + (next_state_action_values * self.GAMMA) 
        expected_state_action_values = expected_state_action_values.unsqueeze(1)
        abs_errors =torch.abs(state_action_values - expected_state_action_values)
        abs_errors = abs_errors.detach().numpy()
        ISWeights = torch.from_numpy(ISWeights)      
        loss = self.my_mse_loss(state_action_values, expected_state_action_values,ISWeights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.batch_update(tree_idx, abs_errors) 

    def update_target_q_network(self, step,tau):
        for target_param, local_param in zip(self.model2.parameters(), self.model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def my_mse_loss(self,outputs, targets,ISWeights):
        return torch.mean(ISWeights *(torch.pow(outputs - targets,2)))
