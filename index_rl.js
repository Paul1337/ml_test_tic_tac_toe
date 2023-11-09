const tf = require('@tensorflow/tfjs-node');

// Определение среды
class Environment {
    constructor() {
        this.state = 0; // Исходное состояние
        this.target = -3;
    }

    // Метод для выполнения действия и получения награды
    step(action) {
        if (action === 0) {
            this.state -= 1; // Двигаться влево
        } else {
            this.state += 1; // Двигаться вправо
        }

        let reward = 0;
        if (this.state === -3) {
            reward = 1; // Агент достиг цели
        }

        return {
            state: this.state,
            reward,
            done: this.state === this.target, // Эпизод завершается при достижении цели
        };
    }
}

// Определение простого Q-обучения
class QLearning {
    constructor() {
        this.env = new Environment();
        this.qTable = tf.tensor([[0, 0]]); // Инициализация Q-таблицы
        this.learningRate = 0.1;
        this.discountFactor = 0.9;
    }

    // Метод для обучения
    train(episodesCount) {
        for (let i = 0; i < episodesCount; i++) {
            console.log('Episode', i);
            let state = this.env.state;
            let done = false;

            let loopInd = 0;
            while (!done) {
                console.log('Loop', loopInd++);
                const action = this.selectAction(state);
                const step = this.env.step(action);
                const { state: nextState, reward } = step;
                done = step.done;
                this.updateQTable(state, action, reward, nextState);
                state = nextState;
            }
            this.env = new Environment(); // Сброс среды
        }
    }

    // Метод для выбора действия на основе Q-таблицы
    selectAction(state) {
        const qValues = this.qTable.arraySync();
        return qValues[state][0] > qValues[state][1] ? 0 : 1;
    }

    // Метод для обновления Q-таблицы
    updateQTable(state, action, reward, nextState) {
        console.log('Update q-table', state, action, reward, nextState);

        const qValues = this.qTable.arraySync();
        console.log(qValues, qValues[nextState]);
        qValues[state][action] +=
            this.learningRate *
            (reward + this.discountFactor * Math.max(...qValues[nextState]) - qValues[state][action]);
        this.qTable = tf.tensor(qValues);
    }
}

// Обучение агента
const qAgent = new QLearning();
qAgent.train(1000);

// Протестируем обученного агента
let state = 0;
while (state !== -3) {
    const action = qAgent.selectAction(state);
    console.log(`State: ${state}, Action: ${action}`);
    const { nextState } = qAgent.env.step(action);
    state = nextState;
}
