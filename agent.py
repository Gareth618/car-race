import random
import numpy as np
from PIL import Image
from queue import Queue
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

class Agent:
    def __init__(self, action_space, memory_size, batch_size, **hyper):
        self.alpha = hyper['alpha']
        self.gamma = hyper['gamma']
        self.epsilon = hyper['epsilon']
        self.epsilon_lower = hyper['epsilon_lower']
        self.epsilon_decay = hyper['epsilon_decay']
        self.action_space = action_space
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96, 96, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space)))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.alpha))
        return model

    def process_state(self, state):
        real_state = []
        for frame in state:
            image = Image.fromarray(frame).convert('L')
            real_state += [np.array(image)]
        return np.expand_dims(np.array(real_state).transpose(1, 2, 0), 0)

    def model_predict(self, state):
        return self.model.predict(self.process_state(state), verbose=0)[0]

    def reset(self):
        self.memory = Queue(self.memory_size)

    def step(self, state, take_action):
        if random.random() < self.epsilon:
            action_index = random.randrange(len(self.action_space))
        else:
            q_values = self.model_predict(state)
            action_index = np.argmax(q_values)
        next_state, reward, game_over = take_action(self.action_space[action_index])
        if len(self.memory.queue) == self.memory.maxsize:
            self.memory.get()
        self.memory.put((state, action_index, next_state, reward, game_over))

    def replay(self):
        batch = random.sample(list(self.memory.queue), self.batch_size)
        training_inputs = []
        training_outputs = []
        for state, action_index, next_state, reward, game_over in batch:
            target = self.model_predict(state)
            if game_over:
                target[action_index] = reward
            else:
                q_values = self.model_predict(next_state)
                target[action_index] = reward + self.gamma * np.max(q_values)
            training_inputs += [self.process_state(state)[0]]
            training_outputs += [target]
        self.model.fit(np.array(training_inputs), np.array(training_outputs), verbose=0)
        if self.epsilon > self.epsilon_lower:
            self.epsilon *= self.epsilon_decay
