import tf from '@tensorflow/tfjs-node';

export class ReplayBuffer {
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
