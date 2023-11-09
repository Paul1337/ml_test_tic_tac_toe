import tf from '@tensorflow/tfjs-node';
import { printBoard, winingLines } from './lib.js';
import readline from 'readline';
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

// const boardSize = 3; // Размер игрового поля
const numStates = 9; // Количество состояний в игровом поле (3x3)
const numActions = 9; // Количество возможных ходов
const initialState = Array(numStates).fill(0); // Начальное состояние среды

const model = tf.sequential();
model.add(tf.layers.dense({ units: 24, activation: 'relu', inputShape: [numStates] }));
model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
model.add(tf.layers.dense({ units: numActions, activation: 'softmax' }));

const learningRate = 0.001;
const discountFactor = 0.99;
const epsilonInitial = 1.0;
const epsilonFinal = 0.01;
const epsilonDecay = 0.995;
const replayBufferCapacity = 10000;
const batchSize = 32;

model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
});

class ReplayBuffer {
    constructor(capacity) {
        this.capacity = capacity;
        this.buffer = [];
        this.position = 0;
    }

    add(sample) {
        if (this.buffer.length < this.capacity) {
            this.buffer.push(sample);
        } else {
            this.buffer[this.position] = sample;
        }
        this.position = (this.position + 1) % this.capacity;
    }

    sample(batchSize) {
        return tf.tidy(() => {
            const batch = [];
            for (let i = 0; i < batchSize; i++) {
                const index = Math.floor(Math.random() * this.buffer.length);
                batch.push(this.buffer[index]);
            }
            return batch;
        });
    }

    size() {
        return this.buffer.length;
    }
}

const replayBuffer = new ReplayBuffer(replayBufferCapacity);

// Определите функцию выбора действия
function selectAction(state, epsilon) {
    if (Math.random() < epsilon) {
        const availableActions = getAvailableActions(state);
        return availableActions[Math.floor(Math.random() * availableActions.length)];
    } else {
        const qValues = model.predict(tf.tensor([state]));
        const probabilities = qValues.dataSync();
        const availableActions = getAvailableActions(state);
        const filteredProbabilities = availableActions.map(action => probabilities[action]);
        const actionIndex = filteredProbabilities.indexOf(Math.max(...filteredProbabilities));
        return availableActions[actionIndex];
    }
}

function updateReplayBufferAndTrain() {
    let state = [...initialState];
    // const availableActions = getAvailableActions(state);
    let totalReward = 0;
    let done = false;

    while (!done) {
        const action = selectAction(state, getEpsilon());
        const nextState = [...state];
        nextState[action] = 1;
        const reward = getReward(nextState);

        // Проверьте, является ли следующее состояние конечным состоянием
        done = doesWin(nextState) || isBoardFull(nextState);

        replayBuffer.add({
            state: state,
            action: action,
            reward: reward,
            nextState: nextState,
            done: done,
        });

        state = nextState;
        totalReward += reward;

        // Обучите модель с использованием экспериенс реплей буфера
        if (replayBuffer.size() >= batchSize) {
            trainModel();
        }
    }

    return totalReward;
}

// Функция обучения модели
function trainModel() {
    const batch = replayBuffer.sample(batchSize);
    const states = [];
    const targets = [];

    batch.forEach(sample => {
        const { state, action, reward, nextState, done } = sample;
        const target = [...state];
        target[action] = reward + (done ? 0 : discountFactor * getMaxQValue(nextState));
        states.push(state);
        targets.push(target);
    });

    model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
}

// Функция получения наилучшего Q-значения для следующего состояния
function getMaxQValue(state) {
    const qValues = model.predict(tf.tensor([state])).dataSync();
    const availableActions = getAvailableActions(state);
    const filteredQValues = availableActions.map(action => qValues[action]);
    return Math.max(...filteredQValues);
}

// Функция получения доступных действий для данного состояния
function getAvailableActions(state) {
    const actions = [];

    for (let i = 0; i < numActions; i++) {
        if (state[i] === 0) {
            actions.push(i);
        }
    }

    return actions;
}

// Функция получения эпсилон для эпсилон-жадной стратегии
function getEpsilon() {
    const epsilonDecayFactor = Math.exp(-epsilonDecay * totalSteps);
    return Math.max(epsilonFinal, epsilonInitial * epsilonDecayFactor);
}

function getReward(state) {
    if (doesWin(state)) {
        return 1;
    } else if (isBoardFull(state)) {
        return 0;
    } else {
        return -0.1;
    }
}

function isBoardFull(state) {
    for (let i = 0; i < numStates; i++) {
        if (state[i] === 0) {
            return false;
        }
    }
    return true;
}

function doesWin(state) {
    for (const [a, b, c] of winingLines) {
        if (state[a] !== 0 && state[a] === state[b] && state[a] === state[c]) {
            return true;
        }
    }

    return false;
}

async function promptUser() {
    return new Promise((resolve, reject) => {
        rl.question('Введите номер ячейки (от 0 до 8): ', answer => {
            resolve(Number(answer));
        });
    });
}

async function playWithHuman() {
    let state = initialState;
    let player = 1; // Игрок 1 начинает игру
    let done = false;

    while (!done) {
        if (player === 1) {
            const action = selectAction(state, getEpsilon());
            state[action] = 1; // Заполните ячейку крестиком
            console.log(`Player 1 move: ${action}`);
        } else {
            console.log('Enter the position (0-8):');
            printBoard(state);
            userInput = await promptUser();
            state[userInput] = -1; // Заполните ячейку ноликом
        }

        // Проверьте, есть ли выигрышная комбинация или заполнена ли доска
        done = doesWin(state) || isBoardFull(state);

        // Поменяйте очередь игрока
        player = -player;
    }

    // Выведите результат игры
    if (doesWin(state)) {
        console.log('Player', player, 'wins!');
    } else {
        console.log('Draw!');
    }
}

const numEpisodes = 100;
let totalSteps = 0;
for (let episode = 0; episode < numEpisodes; episode++) {
    const totalReward = updateReplayBufferAndTrain();
    totalSteps += replayBuffer.size();
    console.log(`Episode ${episode + 1}: Total Reward = ${totalReward}`);
}

await playWithHuman();

rl.close();
