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

const normalizeStateItem = (item) => (item === -1 ? 2 : item);

export const printBoard = (state) => {
    for (let i = 0; i <= state.length - 3; i += 3) {
        const a = normalizeStateItem(state[i]);
        const b = normalizeStateItem(state[i + 1]);
        const c = normalizeStateItem(state[i + 2]);
        console.log(a, b, c);
    }
};

export const isBoardFull = (state) => state.every((item) => item !== 0);
export const getWinner = (state) => {
    for (const [a, b, c] of winingLines) {
        if (state[a] !== 0 && state[a] === state[b] && state[a] === state[c]) {
            return state[a];
        }
    }

    return 0;
};

// Функция получения доступных действий для данного состояния
export const getAvailableActions = (state) => {
    const actions = [];

    for (let i = 0; i < state.length; i++) {
        if (state[i] === 0) {
            actions.push(i);
        }
    }

    return actions;
};
