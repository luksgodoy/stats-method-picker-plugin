"""Recurrent neural network (vanilla RNN, forward + backprop through time from scratch)."""
import numpy as np


def _tanh_deriv(a):
    return 1 - a ** 2


def train_rnn(sequences, targets, n_hidden=10, learning_rate=0.05, epochs=300, seed=0):
    """sequences: array (n_samples, n_steps, n_inputs); targets: (n_samples,) - prediction
    of the next value of the series from the window."""
    rng = np.random.default_rng(seed)
    n_samples, n_steps, n_inputs = sequences.shape
    wxh = rng.normal(0, 0.3, (n_inputs, n_hidden))
    whh = rng.normal(0, 0.3, (n_hidden, n_hidden))
    why = rng.normal(0, 0.3, (n_hidden, 1))
    bh = np.zeros(n_hidden)
    by = np.zeros(1)

    loss_history = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for sample_i in range(n_samples):
            states = [np.zeros(n_hidden)]
            for t in range(n_steps):
                h = np.tanh(sequences[sample_i, t] @ wxh + states[-1] @ whh + bh)
                states.append(h)
            y_pred = states[-1] @ why + by
            error = y_pred - targets[sample_i]
            epoch_loss += float(error ** 2)

            d_why = np.outer(states[-1], error)
            d_by = error
            d_h = error @ why.T
            d_wxh = np.zeros_like(wxh)
            d_whh = np.zeros_like(whh)
            d_bh = np.zeros_like(bh)
            for t in reversed(range(n_steps)):
                d_h_raw = d_h * _tanh_deriv(states[t + 1])
                d_wxh += np.outer(sequences[sample_i, t], d_h_raw)
                d_whh += np.outer(states[t], d_h_raw)
                d_bh += d_h_raw
                d_h = d_h_raw @ whh.T

            for param, grad in ((wxh, d_wxh), (whh, d_whh), (why, d_why),
                                (bh, d_bh), (by, d_by)):
                param -= learning_rate * grad
        loss_history.append(epoch_loss / n_samples)

    return {"initial_loss": loss_history[0], "final_loss": loss_history[-1]}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40, 400)
    series = np.sin(t)
    window = 8
    x = np.array([series[i:i + window] for i in range(len(series) - window)])[..., None]
    y = series[window:]
    r = train_rnn(x[:150], y[:150], epochs=60)
    print("loss:", r["initial_loss"], "->", r["final_loss"])
