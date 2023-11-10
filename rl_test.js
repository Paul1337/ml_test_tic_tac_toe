import readline from 'readline';
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

rl.question('Введите номер ячейки (от 0 до 8): ', (answer) => {
    console.log('Answer:', answer);
});
