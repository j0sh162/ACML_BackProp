import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(a):
    # a = sigmoid(x)
    return a * (1.0 - a)

def feedforward(x, weights):
    """Return (output, list_of_activations)."""
    activations = []
    for W in weights:
        # add bias term to the current input
        x = np.concatenate([x, np.array([[1.0]])], axis=0)
        z = W @ x                # linear part
        a = sigmoid(z)           # activation
        activations.append(a)
        x = a                     # next layer input
    return x, activations          # x is final output

def backpropagation(x, y, weights, lr, l2):
    """Return gradients w.r.t. each weight matrix."""
    # ---- forward pass ----
    out, acts = feedforward(x, weights)

    # ---- backward pass ----
    grads = [None] * len(weights)

    # delta for output layer
    delta = (out - y) * sigmoid_derivative(out)          # (4,1)

    # gradient for last weight matrix
    a_prev = np.concatenate([acts[-2], np.array([[1.0]])], axis=0)
    grads[-1] = delta @ a_prev.T                         # (4,4) if hidden has 3+1 bias

    # propagate through hidden layers (only one hidden layer here)
    for i in range(len(weights) - 2, -1, -1):
        W_next = weights[i + 1]
        # remove bias row from W_next when back‑propagating
        W_next_no_bias = W_next[:, :-1]                  # (4,3)
        delta = (W_next_no_bias.T @ delta) * sigmoid_derivative(acts[i])

        a_prev = np.concatenate([x if i == 0 else acts[i-1],
                                 np.array([[1.0]])], axis=0)
        grads[i] = delta @ a_prev.T

    # ---- weight update (with L2 regularisation) ----
    for i, g in enumerate(grads):
        weights[i] -= lr * (g + l2 * weights[i])

    return out

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# ------------------- data -------------------
# 8‑bit one‑hot vectors (auto‑encoder task)
inputs = np.array([
    [0,0,0,0,0,0,0,1],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0]
], dtype=float).T                      # shape (8, 8)

targets = inputs.copy()                # auto‑encoder → same as input

# ------------------- network -------------------
np.random.seed(0)
hidden_neurons = 3

# weight matrices include bias column as the last column
W1 = np.random.randn(hidden_neurons, inputs.shape[0] + 1)   # (3, 9)
W2 = np.random.randn(targets.shape[0], hidden_neurons + 1) # (8, 4)

weights = [W1, W2]

# ------------------- training -------------------
lr     = 0.01
epochs = 20000
l2_reg = 0.2      # set >0 if you want L2 regularisation

for epoch in range(1, epochs + 1):
    # train on the whole batch (auto‑encoder → one epoch = one batch)
    for i in range(inputs.shape[1]):
        x = inputs[:, i:i+1]      # (8,1)
        y = targets[:, i:i+1]     # (8,1)
        out = backpropagation(x, y, weights, lr, l2_reg)

    if epoch % 1000 == 0 or epoch == 1:
        # compute RMSE on the whole set
        preds = []
        for i in range(inputs.shape[1]):
            p, _ = feedforward(inputs[:, i:i+1], weights)
            preds.append(p.T)      # (1,8)
        preds = np.vstack(preds).T   # (8,8)
        print(f"Epoch {epoch:5d}  RMSE = {rmse(targets, preds):.4f}")