import { getWinner, isBoardFull } from '../lib.js';

export const getReward = state => {
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
};
