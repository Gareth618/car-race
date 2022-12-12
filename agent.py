import math
import random
import numpy as np
from queue import Queue
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

class Agent:
    def __init__(self, action_space, memory_size, **hyper):
        self.alpha = hyper['alpha']
        self.gamma = hyper['gamma']
        self.epsilon = hyper['epsilon']
        self.epsilon_lower = hyper['epsilon_lower']
        self.epsilon_decay = hyper['epsilon_decay']
        self.action_space = action_space
        self.model = self.create_model()
        self.memory = Queue(maxsize=memory_size)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(6, (7, 7), 3, input_shape=(100, 100, 1), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(12, (4, 4), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space)))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.alpha))
        return model

    def model_predict(self, state):
        return self.model.predict(np.expand_dims(state, axis=0), verbose=False)[0]

    def step(self, state, take_action):
        if random.random() < self.epsilon:
            action_index = random.randint(0, len(self.action_space) - 1)
        else:
            q_values = self.model_predict(state)
            action_index = np.argmax(q_values)
        next_state, reward, game_over = take_action(self.action_space[action_index])
        if len(self.memory.queue) == self.memory.maxsize:
            self.memory.get()
        self.memory.put((state, action_index, next_state, reward, game_over))

    def replay(self):
        sample_space = list(self.memory.queue)
        batch = random.sample(sample_space, int(math.sqrt(len(sample_space))))
        training_inputs = []
        training_outputs = []
        for state, action_index, next_state, reward, game_over in batch:
            target = self.model_predict(state)
            if game_over:
                target[action_index] = reward
            else:
                q_values = self.model_predict(next_state)
                target[action_index] = reward + self.gamma * np.amax(q_values)
            training_inputs += [state]
            training_outputs += [target]
        self.model.fit(np.array(training_inputs), np.array(training_outputs), verbose=False)
        if self.epsilon > self.epsilon_lower:
            self.epsilon -= self.epsilon_decay
