import tf from '@tensorflow/tfjs-node';
import { getWinner, getAvailableActions, isBoardFull, printBoard } from './lib.js';
import { promptUser } from './rl.js';
import { cwd } from 'process';

// const boardSize = 3; // Размер игрового поля
const numStates = 9; // Количество состояний в игровом поле (3x3)
const numActions = 9; // Количество возможных ходов
const initialState = Array(numStates).fill(0); // Начальное состояние среды

let model = tf.sequential();
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

function selectRandomAction(state) {
    const availableActions = getAvailableActions(state);
    return availableActions[Math.floor(Math.random() * availableActions.length)];
}

// Определите функцию выбора действия
function selectAction(state, epsilon) {
    if (Math.random() < epsilon) {
        return selectRandomAction(state);
    } else {
        const qValues = model.predict(tf.tensor([state]));
        const probabilities = qValues.dataSync();
        const availableActions = getAvailableActions(state);
        const filteredProbabilities = availableActions.map(action => probabilities[action]);
        const actionIndex = filteredProbabilities.indexOf(Math.max(...filteredProbabilities));
        return availableActions[actionIndex];
    }
}

async function updateReplayBufferAndTrain() {
    // console.log('training episode start');

    let state = [...initialState];
    // const availableActions = getAvailableActions(state);
    let totalReward = 0;
    let done = false;

    while (!done) {
        // console.log(state);
        const action = selectAction(state, getEpsilon());
        const nextState = [...state];
        nextState[action] = 1;
        const reward = getReward(nextState);
        // console.log('reward', reward);

        done = getWinner(nextState) !== 0 || isBoardFull(nextState);

        replayBuffer.add({
            state: state,
            action: action,
            reward: reward,
            nextState: nextState,
            done: done,
        });

        const randomAction = selectRandomAction(nextState);
        nextState[randomAction] = -1;

        state = nextState;
        totalReward += reward;
        // console.log('total', totalReward);

        // Обучите модель с использованием экспериенс реплей буфера
        if (replayBuffer.size() >= batchSize) {
            await trainModel();
        }

        // console.log('board state:');
        // printBoard(state);
    }

    return totalReward;
}

// Функция обучения модели
async function trainModel() {
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

    await model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
    // console.log('retrained');
}

// Функция получения наилучшего Q-значения для следующего состояния
function getMaxQValue(state) {
    const qValues = model.predict(tf.tensor([state])).dataSync();
    const availableActions = getAvailableActions(state);
    const filteredQValues = availableActions.map(action => qValues[action]);
    return Math.max(...filteredQValues);
}

// Функция получения эпсилон для эпсилон-жадной стратегии
function getEpsilon() {
    const epsilonDecayFactor = Math.exp(-epsilonDecay * totalSteps);
    return Math.max(epsilonFinal, epsilonInitial * epsilonDecayFactor);
}

function getReward(state) {
    const winner = getWinner(state);
    if (winner === 1) {
        return 1;
    } else if (winner === -1) {
        return -1;
    } else if (isBoardFull(state)) {
        return -0.2;
    } else {
        return 0;
    }
}

async function playWithHuman() {
    console.log('Game starts!');

    let state = [...initialState];
    let player = 1; // Игрок 1 начинает игру
    let done = false;

    while (!done) {
        if (player === 1) {
            const action = selectAction(state, 0);
            state[action] = 1; // Заполните ячейку крестиком
            console.log(`Player 1 move: ${action}`);
        } else {
            console.log('Enter the position (0-8):');
            const userInput = await promptUser();
            state[userInput] = -1; // Заполните ячейку ноликом
        }

        printBoard(state);
        done = getWinner(state) !== 0 || isBoardFull(state);
        player = -player;
    }

    // Выведите результат игры
    const winner = getWinner(state);
    if (winner !== 0) {
        console.log('Player', winner, 'wins!');
    } else {
        console.log('Draw!');
    }
}

let totalSteps = 0;
const MODE = {
    Save: 0,
    LoadFromFile: 1,
};
const CurrentMode = MODE.LoadFromFile;
const SAVE_PATH = `file://${cwd()}/model`;
const LOAD_PATH = SAVE_PATH + '/model.json';

async function trainAll() {
    const numEpisodes = 2000;
    for (let episode = 0; episode < numEpisodes; episode++) {
        const totalReward = await updateReplayBufferAndTrain();
        totalSteps += replayBuffer.size();
        if ((episode + 1) % 10 === 0) {
            console.log(`Episode ${episode + 1}: Total Reward = ${totalReward}`);
        }
    }
}

if (CurrentMode === MODE.Save) {
    model = await tf.loadLayersModel(LOAD_PATH);
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'categoricalCrossentropy',
    });

    await trainAll();
    await model.save(SAVE_PATH);
} else {
    model = await tf.loadLayersModel(LOAD_PATH);
}

while (true) {
    await playWithHuman();
}
