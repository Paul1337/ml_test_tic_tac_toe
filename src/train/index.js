import tf from '@tensorflow/tfjs-node';
import { getWinner, isBoardFull, printBoard } from '../lib.js';
import { ReplayBuffer } from './replayBuffer.js';
import fs from 'fs';
import { config } from '../config.js';
import { Bot } from '../bot/bot.js';
import { getReward } from './reward.js';

const updateReplayBufferAndTrain = async () => {
    let state = [...config.initialState];
    let totalReward = 0;
    let done = false;

    while (!done) {
        const action = bot.selectMove(state, getEpsilon());
        if (action !== 0 && !action) {
            console.log('Action', action);
            throw new Error('Action is not valid!');
        }
        const nextState = [...state];
        nextState[action] = 1;
        const reward = getReward(nextState);

        done = getWinner(nextState) !== 0 || isBoardFull(nextState);

        replayBuffer.add({
            state: state,
            action: action,
            reward: reward,
            nextState: nextState,
            done: done,
        });

        const randomAction = bot.getRandomMove(nextState);
        nextState[randomAction] = -1;

        state = nextState;
        totalReward += reward;

        if (replayBuffer.size() >= config.batchSize) {
            await trainModel();
        }
    }

    return totalReward;
};

const trainModel = async () => {
    const batch = replayBuffer.sample(config.batchSize);
    const states = [];
    const targets = [];

    batch.forEach(sample => {
        const { state, action, reward, nextState, done } = sample;
        const target = [...state];
        target[action] = reward + (done ? 0 : config.discountFactor * getMaxQValue(nextState));
        states.push(state);
        targets.push(target);
    });

    await model.fit(tf.tensor2d(states), tf.tensor2d(targets), { epochs: 1, verbose: 0 });
};

const getMaxQValue = state => {
    const qValues = model.predict(tf.tensor([state])).dataSync();
    const availableMoves = bot.getAvailableMoves(state);
    const filteredQValues = availableMoves.map(action => qValues[action]);
    return Math.max(...filteredQValues);
};

const getEpsilon = () => {
    const epsilonDecayFactor = Math.exp(-config.epsilonDecay * totalSteps);
    return Math.max(config.epsilonFinal, config.epsilonInitial * epsilonDecayFactor);
};

let totalSteps = 0;

const trainAll = async () => {
    const numEpisodes = process.env.episodes ?? 1000;
    console.log(`Training start.. ${numEpisodes} episodes`);
    for (let episode = 0; episode < numEpisodes; episode++) {
        const totalReward = await updateReplayBufferAndTrain();
        totalSteps += replayBuffer.size();
        if ((episode + 1) % 10 === 0) {
            console.log(`Episode ${episode + 1}: Total Reward = ${totalReward}`);
        }
    }
    console.log('Training complete');
};

let model, replayBuffer, bot;

const main = async () => {
    replayBuffer = new ReplayBuffer(config.replayBufferCapacity);

    console.log(config.modelLocation.loadPath);
    if (fs.existsSync(config.modelLocation.loadPath.substring(7))) {
        console.log('Found model, additional training..');
        model = await tf.loadLayersModel(config.modelLocation.loadPath);
    } else {
        console.log('Model does not exists, creating from scratch..');
        model = tf.sequential();
        model.add(tf.layers.dense({ units: 24, activation: 'relu', inputShape: [config.numStates] }));
        model.add(tf.layers.dense({ units: 24, activation: 'relu' }));
        model.add(tf.layers.dense({ units: config.numActions, activation: 'softmax' }));
    }

    console.log('Compiling model...');

    model.compile({
        optimizer: tf.train.adam(config.learningRate),
        loss: 'categoricalCrossentropy',
    });

    bot = new Bot(model);

    await trainAll();
    // await model.save(config.modelLocation.savePath);
};

main();
