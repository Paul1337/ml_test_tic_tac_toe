import readline from 'readline';

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

export const promptUser = () => {
    return new Promise((resolve, reject) => {
        rl.question('Введите номер ячейки (от 0 до 8): ', (answer) => {
            resolve(Number(answer));
        });
    });
};
