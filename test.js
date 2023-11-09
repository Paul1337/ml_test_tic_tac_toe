const tf = require('@tensorflow/tfjs');

// Создание модели
const model = tf.sequential();
model.add(tf.layers.flatten({ inputShape: [2, 2] }));
model.add(tf.layers.dense({ units: 1 }));

// Компиляция модели
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

// Подготовка обучающих данных
const xs = tf.tensor3d([
    [
        [1, 2],
        [3, 4],
    ],
    [
        [5, 6],
        [7, 8],
    ],
    [
        [9, 10],
        [11, 12],
    ],
]);
const ys = tf.tensor2d([[5], [11], [17]]);

// Обучение модели
model.fit(xs, ys, { epochs: 100 }).then(() => {
    // Предсказание значения
    const input = tf.tensor3d([
        [
            [13, 14],
            [15, 16],
        ],
    ]);
    const output = model.predict(input);
    output.print();
});
