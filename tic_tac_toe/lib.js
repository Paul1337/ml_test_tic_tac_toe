export const winingLines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8], // Горизонтальные линии
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8], // Вертикальные линии
    [0, 4, 8],
    [2, 4, 6], // Диагональные линии
];

export const isBoardFull = state => state.every(item => item === 0);

export const printBoard = state => {};
