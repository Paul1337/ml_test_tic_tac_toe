import tf from '@tensorflow/tfjs-node';

export class Bot {
    constructor(model) {
        this.model = model;
    }

    selectMove(state, epsilon) {
        // console.log(`Selecting move, epsilon: ${epsilon}`);
        if (Math.random() < epsilon) {
            return this.getRandomMove(state);
        } else {
            return this.getPredictedMove(state);
        }
    }

    getPredictedMove(state) {
        const qValues = this.model.predict(tf.tensor([state]));
        const probabilities = qValues.dataSync();
        const availableMoves = this.getAvailableMoves(state);
        const filteredProbabilities = availableMoves.map(action => probabilities[action]);
        const actionIndex = filteredProbabilities.indexOf(Math.max(...filteredProbabilities));
        return availableMoves[actionIndex];
    }

    getRandomMove(state) {
        const availableMoves = this.getAvailableMoves(state);
        return availableMoves[Math.floor(Math.random() * availableMoves.length)];
    }

    getAvailableMoves(state) {
        const moves = [];
        for (let i = 0; i < state.length; i++) {
            if (state[i] === 0) moves.push(i);
        }
        return moves;
    }
}
