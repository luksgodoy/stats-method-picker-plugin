"""Neural network implemented from scratch (manual forward/backprop, no DL framework)."""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ann_from_scratch(x, y, n_hidden=6, learning_rate=0.5, epochs=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n_inputs = x.shape[1]

    w1 = rng.normal(0, 1, (n_inputs, n_hidden)) * np.sqrt(1 / n_inputs)
    b1 = np.zeros((1, n_hidden))
    w2 = rng.normal(0, 1, (n_hidden, 1)) * np.sqrt(1 / n_hidden)
    b2 = np.zeros((1, 1))

    loss_history = []
    for _ in range(epochs):
        hidden = sigmoid(x @ w1 + b1)
        output = sigmoid(hidden @ w2 + b2)
        loss = float(np.mean((y - output) ** 2))
        loss_history.append(loss)

        output_error = (output - y) * output * (1 - output)
        hidden_error = (output_error @ w2.T) * hidden * (1 - hidden)

        w2 -= learning_rate * hidden.T @ output_error / len(y)
        b2 -= learning_rate * output_error.mean(axis=0, keepdims=True)
        w1 -= learning_rate * x.T @ hidden_error / len(y)
        b1 -= learning_rate * hidden_error.mean(axis=0, keepdims=True)

    return {"initial_loss": loss_history[0], "final_loss": loss_history[-1],
            "train_accuracy": float(((sigmoid(sigmoid(x @ w1 + b1) @ w2 + b2) > 0.5) == y).mean())}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (300, 2))
    y = ((x[:, 0] * x[:, 1]) > 0).astype(float)  # XOR-like, non-linear - needs a hidden layer
    r = ann_from_scratch(x, y)
    print("loss:", r["initial_loss"], "->", r["final_loss"], "accuracy:", r["train_accuracy"])
