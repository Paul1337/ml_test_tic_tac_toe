import { cwd } from 'process';

const SAVE_PATH = `file://${cwd()}/model`;
const LOAD_PATH = SAVE_PATH + '/model.json';

export const config = {
    modelLocation: {
        savePath: SAVE_PATH,
        loadPath: LOAD_PATH,
    },
    numStates: 9,
    numActions: 9,
    learningRate: 0.001,
    discountFactor: 0.99,
    epsilonInitial: 1.0,
    epsilonFinal: 0.01,
    epsilonDecay: 0.995,
    replayBufferCapacity: 10000,
    batchSize: 32,
};

config.initialState = new Array(config.numStates).fill(0);
