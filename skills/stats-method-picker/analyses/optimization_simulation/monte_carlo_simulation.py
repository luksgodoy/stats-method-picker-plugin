"""Monte Carlo simulation for propagating uncertainty in models with random inputs."""
import numpy as np


def simulate_monte_carlo(model_function, input_generators, n_simulations=10000, seed=0):
    """model_function(inputs_dict) -> float; input_generators: dict name -> callable(rng, n) -> array."""
    rng = np.random.default_rng(seed)
    input_samples = {name: generator(rng, n_simulations) for name, generator in input_generators.items()}
    results = np.array([
        model_function({name: input_samples[name][i] for name in input_samples})
        for i in range(n_simulations)
    ])
    return {
        "mean": float(results.mean()),
        "std": float(results.std()),
        "percentile_5": float(np.percentile(results, 5)),
        "percentile_95": float(np.percentile(results, 95)),
        "loss_probability": float((results < 0).mean()),
    }


if __name__ == "__main__":
    def project_profit(inputs):
        return inputs["revenue"] - inputs["cost"] - inputs["fixed_cost"]

    generators = {
        "revenue": lambda rng, n: rng.normal(100000, 15000, n),
        "cost": lambda rng, n: rng.normal(60000, 10000, n),
        "fixed_cost": lambda rng, n: np.full(n, 20000.0),
    }
    r = simulate_monte_carlo(project_profit, generators, n_simulations=20000)
    print("mean profit:", round(r["mean"], 2), "std:", round(r["std"], 2))
    print("P(loss):", round(r["loss_probability"] * 100, 2), "%")
