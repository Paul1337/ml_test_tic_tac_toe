import tf from '@tensorflow/tfjs-node';
import { Bot } from '../bot/bot.js';
import { config } from '../config.js';
import { getWinner, isBoardFull, printBoard } from '../lib.js';
import { promptUser } from './rl.js';

async function playWithHuman() {
    console.log('Game starts!');

    let state = [...config.initialState];
    let player = 1;
    if (Math.random() < 0.5) {
        player = -1;
        printBoard(state);
    }

    let done = false;

    while (!done) {
        if (player === 1) {
            const action = bot.getPredictedMove(state);
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

const model = await tf.loadLayersModel(config.modelLocation.loadPath);
model.compile({
    optimizer: tf.train.adam(config.learningRate),
    loss: 'categoricalCrossentropy',
});
const bot = new Bot(model);

while (true) {
    await playWithHuman();
}
