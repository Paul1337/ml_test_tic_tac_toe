const tf = require('@tensorflow/tfjs-node');

// Создание модели
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [2, 2] }));
// model.add(tf.layers.dense({ units: 1, outputShape: [2] }));
model.add(tf.layers.reshape({ targetShape: [2] }));

// Компиляция модели
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

// Подготовка обучающих данных

const xs = tf.tensor([
    [
        [1, 3],
        [1, 2],
    ],
]);
const ys = tf.tensor([[1, 0]]);

// const xs = tf.tensor2d([1], [1, 1]);
// const ys = tf.tensor2d([2], [1, 1]);

xs.print();
ys.print();
// Обучение модели
model.fit(xs, ys, { epochs: 100 }).then(() => {
    // Предсказание значения
    const input = tf.tensor([
        [
            [3, 3],
            [1, 4],
        ],
    ]);
    const output = model.predict(input);

    console.log(output);
    output.print();
});
