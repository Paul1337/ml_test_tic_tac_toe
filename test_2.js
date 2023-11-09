const tf = require('@tensorflow/tfjs-node');

const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [4] }));
model.compile({ loss: 'binaryCrossentropy', optimizer: 'sgd' });

// Input data
// Array of days, and their capacity used out of
// 100% for 5 hour period
const xs = tf.tensor([
    [11, 23, 34, 45],
    [12, 23, 43, 56],
    [12, 23, 56, 67],
    [13, 34, 56, 45],
    [12, 23, 54, 56],
]);

// Labels
const ys = tf.tensor([[1], [2], [3], [4], [5]]);

// Train the model using the data.
model
    .fit(xs, ys)
    .then(() => {
        model.predict(tf.tensor([[11, 23, 34, 45]])).print();
    })
    .catch((e) => {
        console.log(e.message);
    });
