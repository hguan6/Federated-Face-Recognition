import flwr as fl
from flwr.common import Metrics

# Define metric aggregation function
def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Federated facial recognition")
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        required=False,
        help="Specifies the number of rounds of aggregation.")
    args = parser.parse_args()

    # Define strategy
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
