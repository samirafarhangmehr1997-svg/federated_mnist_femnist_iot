import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import csv
import shutil
import numpy as np

# Keep TensorFlow logs quieter.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import tensorflow_datasets as tfds
except ImportError:
    print("\nERROR: tensorflow_datasets is not installed.")
    print("Install it inside your conda environment with:")
    print("python -m pip install tensorflow-datasets")
    print("or, if you need mirror:")
    print("python -m pip install -i https://mirror-pypi.runflare.com/simple tensorflow-datasets")
    sys.exit(1)


# ==========================================================
# START
# ==========================================================
print("Federated MNIST + FEMNIST-compatible EMNIST IoT Classification Started")
print("Python path:", sys.executable)
print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)


# ==========================================================
# GENERAL SETTINGS
# ==========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

NUM_CLIENTS = 20
NUM_EDGE_SERVERS = 4
HIDDEN_UNITS = 128

ROUNDS = 100
CLIENTS_PER_ROUND = 18

# For computational feasibility, the experiment uses controlled subsets.
TRAIN_SUBSET = 30000
TEST_SUBSET = 6000

LOCAL_BATCH_SIZE = 64
BASE_LEARNING_RATE = 0.035
DIRICHLET_ALPHA = 0.35

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "figures_mnist_femnist_results")
BASE_DATA_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_mnist_femnist_tables")
BASE_ABLATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ablation_figures_mnist_femnist")

if os.path.exists(BASE_OUTPUT_DIR):
    shutil.rmtree(BASE_OUTPUT_DIR)

if os.path.exists(BASE_DATA_OUTPUT_DIR):
    shutil.rmtree(BASE_DATA_OUTPUT_DIR)

if os.path.exists(BASE_ABLATION_OUTPUT_DIR):
    shutil.rmtree(BASE_ABLATION_OUTPUT_DIR)

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(BASE_ABLATION_OUTPUT_DIR, exist_ok=True)


DATASET_CONFIGS = [
    {
        "tag": "mnist",
        "display_name": "MNIST",
        "tfds_name": "mnist",
        "num_classes": 10,
        "train_split": "train",
        "test_split": "test",
        "train_subset": TRAIN_SUBSET,
        "test_subset": TEST_SUBSET
    },
    {
        "tag": "femnist_emnist_byclass",
        "display_name": "FEMNIST-compatible EMNIST ByClass",
        "tfds_name": "emnist/byclass",
        "num_classes": 62,
        "train_split": "train",
        "test_split": "test",
        "train_subset": TRAIN_SUBSET,
        "test_subset": TEST_SUBSET
    }
]


# ==========================================================
# DATASET LOADING
# ==========================================================
def load_image_classification_subset(dataset_name, split_name, max_samples, seed):
    dataset = tfds.load(
        dataset_name,
        split=split_name,
        as_supervised=True,
        shuffle_files=True
    )

    dataset = dataset.shuffle(
        buffer_size=100000,
        seed=seed,
        reshuffle_each_iteration=False
    )

    dataset = dataset.take(max_samples)

    images = []
    labels = []

    count = 0

    for image, label in dataset:
        image_np = image.numpy().astype(np.float32)

        # Convert 28 x 28 grayscale image to a 784-dimensional feature vector.
        image_np = np.squeeze(image_np) / 255.0
        image_np = image_np.reshape(-1)

        label_np = int(label.numpy())

        images.append(image_np)
        labels.append(label_np)

        count += 1

        if count % 5000 == 0:
            print(f"Loaded {count} samples from {dataset_name}/{split_name}...")

    x_array = np.stack(images).astype(np.float32)
    y_array = np.array(labels, dtype=np.int64)

    return x_array, y_array


# ==========================================================
# GLOBAL HELPER FUNCTIONS
# ==========================================================
def stable_softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def relu(x):
    return np.maximum(x, 0.0)


def macro_precision_recall_f1(y_true, y_pred, num_classes):
    """
    Computes macro-averaged Precision, Recall, and F1-score for multi-class
    classification without requiring scikit-learn.

    Values are returned as percentages to match the accuracy scale.
    """
    precisions = []
    recalls = []
    f1_scores = []

    for class_id in range(num_classes):
        true_positive = np.sum((y_pred == class_id) & (y_true == class_id))
        false_positive = np.sum((y_pred == class_id) & (y_true != class_id))
        false_negative = np.sum((y_pred != class_id) & (y_true == class_id))

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = float(np.mean(precisions) * 100.0)
    macro_recall = float(np.mean(recalls) * 100.0)
    macro_f1 = float(np.mean(f1_scores) * 100.0)

    return macro_precision, macro_recall, macro_f1


# ==========================================================
# MAIN EXPERIMENT FUNCTION
# ==========================================================
def run_experiment(dataset_config):
    dataset_tag = dataset_config["tag"]
    dataset_display_name = dataset_config["display_name"]
    dataset_name = dataset_config["tfds_name"]
    num_classes = dataset_config["num_classes"]

    print("\n" + "=" * 70)
    print(f"Running dataset: {dataset_display_name}")
    print("=" * 70)

    output_dir = os.path.join(BASE_OUTPUT_DIR, dataset_tag)
    data_output_dir = os.path.join(BASE_DATA_OUTPUT_DIR, dataset_tag)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_output_dir, exist_ok=True)

    x_train, y_train = load_image_classification_subset(
        dataset_name,
        dataset_config["train_split"],
        dataset_config["train_subset"],
        SEED
    )

    x_test, y_test = load_image_classification_subset(
        dataset_name,
        dataset_config["test_split"],
        dataset_config["test_subset"],
        SEED + 1
    )

    num_features = x_train.shape[1]

    model_size = (
        (num_features * HIDDEN_UNITS) +
        HIDDEN_UNITS +
        (HIDDEN_UNITS * num_classes) +
        num_classes
    )

    print("Dataset:", dataset_name)
    print("Train set:", x_train.shape, y_train.shape)
    print("Test set:", x_test.shape, y_test.shape)
    print("Number of classes:", num_classes)
    print("Model parameters:", model_size)

    # ======================================================
    # NON-IID CLIENT SPLIT
    # ======================================================
    def create_non_iid_clients(x_data, y_data, num_clients, alpha):
        client_indices = [[] for _ in range(num_clients)]

        for class_id in range(num_classes):
            class_indices = np.where(y_data == class_id)[0]
            np.random.shuffle(class_indices)

            proportions = np.random.dirichlet(np.ones(num_clients) * alpha)
            proportions = proportions / proportions.sum()

            split_points = (np.cumsum(proportions) *
                            len(class_indices)).astype(int)[:-1]
            class_splits = np.split(class_indices, split_points)

            for client_id, split in enumerate(class_splits):
                client_indices[client_id].extend(split.tolist())

        # Make sure no client is empty.
        for client_id in range(num_clients):
            if len(client_indices[client_id]) == 0:
                largest_client = np.argmax([len(idx) for idx in client_indices])
                moved = client_indices[largest_client][:10]
                client_indices[client_id].extend(moved)
                client_indices[largest_client] = client_indices[largest_client][10:]

        clients = []

        for client_id in range(num_clients):
            idx = np.array(client_indices[client_id])
            np.random.shuffle(idx)

            clients.append({
                "x": x_data[idx],
                "y": y_data[idx],
                "size": len(idx)
            })

        return clients

    clients = create_non_iid_clients(
        x_train,
        y_train,
        NUM_CLIENTS,
        DIRICHLET_ALPHA
    )

    print("\nClient data distribution:")
    for i, client in enumerate(clients):
        unique, counts = np.unique(client["y"], return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        print(
            f"Client {i + 1:02d}: {client['size']} samples | labels: {distribution}"
        )

    # Client quality indicators for data-aware adaptive scheduling.
    client_label_entropy = []
    client_size_score = []

    max_client_size = max(client["size"] for client in clients)

    for client in clients:
        label_counts = np.bincount(
            client["y"], minlength=num_classes).astype(np.float32)
        label_probs = label_counts / np.sum(label_counts)

        label_probs_nonzero = label_probs[label_probs > 0]
        entropy = -np.sum(label_probs_nonzero * np.log(label_probs_nonzero))
        normalized_entropy = entropy / np.log(num_classes)

        client_label_entropy.append(float(normalized_entropy))
        client_size_score.append(float(client["size"] / max_client_size))

    # ======================================================
    # MODEL FUNCTIONS: LIGHTWEIGHT MLP CLASSIFIER
    # ======================================================
    def initialize_model_vector():
        w1 = np.random.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / num_features),
            size=(num_features, HIDDEN_UNITS)
        ).astype(np.float32)

        b1 = np.zeros(HIDDEN_UNITS, dtype=np.float32)

        w2 = np.random.normal(
            loc=0.0,
            scale=np.sqrt(2.0 / HIDDEN_UNITS),
            size=(HIDDEN_UNITS, num_classes)
        ).astype(np.float32)

        b2 = np.zeros(num_classes, dtype=np.float32)

        return np.concatenate([
            w1.reshape(-1),
            b1,
            w2.reshape(-1),
            b2
        ]).astype(np.float32)

    def vector_to_params(model_vector):
        p1 = num_features * HIDDEN_UNITS
        p2 = p1 + HIDDEN_UNITS
        p3 = p2 + (HIDDEN_UNITS * num_classes)

        w1 = model_vector[:p1].reshape(num_features, HIDDEN_UNITS)
        b1 = model_vector[p1:p2]
        w2 = model_vector[p2:p3].reshape(HIDDEN_UNITS, num_classes)
        b2 = model_vector[p3:]

        return w1, b1, w2, b2

    def params_to_vector(w1, b1, w2, b2):
        return np.concatenate([
            w1.reshape(-1),
            b1,
            w2.reshape(-1),
            b2
        ]).astype(np.float32)

    def forward_pass(model_vector, x_data):
        w1, b1, w2, b2 = vector_to_params(model_vector)

        z1 = x_data @ w1 + b1
        a1 = relu(z1)
        logits = a1 @ w2 + b2

        return z1, a1, logits

    def cross_entropy_loss(model_vector, x_data, y_data):
        _, _, logits = forward_pass(model_vector, x_data)
        probabilities = stable_softmax(logits)

        eps = 1e-12
        correct_probs = probabilities[np.arange(len(y_data)), y_data]
        loss = -np.mean(np.log(correct_probs + eps))

        return float(loss)

    def evaluate_model(model_vector, x_data, y_data):
        loss = cross_entropy_loss(model_vector, x_data, y_data)
        _, _, logits = forward_pass(model_vector, x_data)
        predictions = np.argmax(logits, axis=1)

        accuracy = float(np.mean(predictions == y_data) * 100.0)
        precision, recall, f1_score = macro_precision_recall_f1(
            y_true=y_data,
            y_pred=predictions,
            num_classes=num_classes
        )

        return {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def local_train(model_vector, x_data, y_data, local_epochs, learning_rate):
        local_vector = model_vector.copy()

        for _ in range(local_epochs):
            indices = np.random.permutation(len(x_data))

            for start in range(0, len(indices), LOCAL_BATCH_SIZE):
                batch_indices = indices[start:start + LOCAL_BATCH_SIZE]

                xb = x_data[batch_indices]
                yb = y_data[batch_indices]

                w1, b1, w2, b2 = vector_to_params(local_vector)

                z1 = xb @ w1 + b1
                a1 = relu(z1)
                logits = a1 @ w2 + b2
                probabilities = stable_softmax(logits)

                one_hot = np.zeros_like(probabilities)
                one_hot[np.arange(len(yb)), yb] = 1.0

                dz2 = (probabilities - one_hot) / len(yb)

                grad_w2 = a1.T @ dz2
                grad_b2 = np.sum(dz2, axis=0)

                da1 = dz2 @ w2.T
                dz1 = da1 * (z1 > 0)

                grad_w1 = xb.T @ dz1
                grad_b1 = np.sum(dz1, axis=0)

                # Gradient clipping improves numerical stability in non-IID FL.
                clip_value = 2.0
                grad_w1 = np.clip(grad_w1, -clip_value, clip_value)
                grad_b1 = np.clip(grad_b1, -clip_value, clip_value)
                grad_w2 = np.clip(grad_w2, -clip_value, clip_value)
                grad_b2 = np.clip(grad_b2, -clip_value, clip_value)

                w1 = w1 - learning_rate * grad_w1
                b1 = b1 - learning_rate * grad_b1
                w2 = w2 - learning_rate * grad_w2
                b2 = b2 - learning_rate * grad_b2

                local_vector = params_to_vector(w1, b1, w2, b2)

        update = local_vector - model_vector

        return update

    # ======================================================
    # COMPRESSION AND AGGREGATION
    # ======================================================
    def top_k_compress(update, compression_ratio):
        if compression_ratio >= 1.0:
            return update.copy(), len(update)

        k = max(1, int(len(update) * compression_ratio))
        indices = np.argpartition(np.abs(update), -k)[-k:]

        compressed = np.zeros_like(update)
        compressed[indices] = update[indices]

        return compressed, k

    def weighted_average_updates(updates, weights):
        if len(updates) == 0:
            return np.zeros(model_size, dtype=np.float32)

        weights = np.array(weights, dtype=np.float32)
        weights = weights / np.sum(weights)

        aggregated = np.zeros_like(updates[0], dtype=np.float32)

        for update, weight in zip(updates, weights):
            aggregated += update * weight

        return aggregated

    def edge_aggregate(received_items):
        if len(received_items) == 0:
            return np.zeros(model_size, dtype=np.float32)

        edge_updates = []
        edge_weights = []

        for edge_id in range(NUM_EDGE_SERVERS):
            local_updates = []
            local_weights = []

            for item in received_items:
                if item["edge_id"] == edge_id:
                    local_updates.append(item["update"])
                    local_weights.append(item["weight"])

            if len(local_updates) > 0:
                edge_update = weighted_average_updates(local_updates, local_weights)
                edge_updates.append(edge_update)
                edge_weights.append(np.sum(local_weights))

        if len(edge_updates) == 0:
            return np.zeros(model_size, dtype=np.float32)

        return weighted_average_updates(edge_updates, edge_weights)

    # ======================================================
    # NETWORK PROFILES AND ALGORITHM SETTINGS
    # ======================================================
    client_profiles = []

    for _ in range(NUM_CLIENTS):
        client_profiles.append({
            "delay_scale": np.random.uniform(0.85, 1.25),
            "loss_scale": np.random.uniform(0.70, 1.35),
            "energy_scale": np.random.uniform(0.85, 1.20)
        })

    algorithm_configs = {
        "FedAvg": {
            "local_epochs": 1,
            "learning_rate": BASE_LEARNING_RATE,
            "server_update_gain": 1.00,
            "compression_ratio": 1.00,
            "packet_loss": 0.18,
            "delay_low": 1.10,
            "delay_high": 1.95,
            "congestion_low": 0.95,
            "congestion_high": 1.40,
            "selection": "all",
            "clients_per_round": NUM_CLIENTS,
            "edge": False,
            "residual": False,
            "latency_mode": "synchronous"
        },

        "Simple FL": {
            "local_epochs": 1,
            "learning_rate": BASE_LEARNING_RATE,
            "server_update_gain": 1.00,
            "compression_ratio": 1.00,
            "packet_loss": 0.13,
            "delay_low": 0.95,
            "delay_high": 1.65,
            "congestion_low": 0.90,
            "congestion_high": 1.35,
            "selection": "random",
            "clients_per_round": CLIENTS_PER_ROUND,
            "edge": False,
            "residual": False,
            "latency_mode": "semi_async"
        },

        "Compression-only": {
            "local_epochs": 1,
            "learning_rate": BASE_LEARNING_RATE,
            "server_update_gain": 1.00,
            "compression_ratio": 0.40,
            "packet_loss": 0.15,
            "delay_low": 0.85,
            "delay_high": 1.50,
            "congestion_low": 0.85,
            "congestion_high": 1.25,
            "selection": "random",
            "clients_per_round": CLIENTS_PER_ROUND,
            "edge": False,
            "residual": False,
            "latency_mode": "semi_async"
        },

        "Edge-only": {
            "local_epochs": 1,
            "learning_rate": BASE_LEARNING_RATE,
            "server_update_gain": 1.00,
            "compression_ratio": 1.00,
            "packet_loss": 0.08,
            "delay_low": 0.70,
            "delay_high": 1.25,
            "congestion_low": 0.75,
            "congestion_high": 1.15,
            "selection": "random",
            "clients_per_round": CLIENTS_PER_ROUND,
            "edge": True,
            "residual": False,
            "latency_mode": "edge_async"
        },

        "Proposed": {
            "local_epochs": 3,
            "learning_rate": BASE_LEARNING_RATE * 0.90,
            "server_update_gain": 1.08,
            "compression_ratio": 0.50,
            "packet_loss": 0.03,
            "delay_low": 0.55,
            "delay_high": 1.05,
            "congestion_low": 0.65,
            "congestion_high": 1.05,
            "selection": "data_aware_adaptive",
            "clients_per_round": CLIENTS_PER_ROUND,
            "edge": True,
            "residual": True,
            "latency_mode": "adaptive_semi_async"
        }
    }

    algorithms = list(algorithm_configs.keys())

    def select_clients(config):
        selection_mode = config["selection"]
        clients_per_round = config.get("clients_per_round", CLIENTS_PER_ROUND)

        if selection_mode == "all":
            return list(range(NUM_CLIENTS))

        if selection_mode == "data_aware_adaptive":
            scores = []

            for client_id, profile in enumerate(client_profiles):
                network_penalty = (
                    0.55 * profile["delay_scale"] +
                    0.35 * profile["loss_scale"]
                )

                data_bonus = (
                    0.45 * client_label_entropy[client_id] +
                    0.25 * client_size_score[client_id]
                )

                exploration_noise = np.random.uniform(0.0, 0.06)

                # Lower score is better.
                score = network_penalty - data_bonus + exploration_noise
                scores.append(score)

            selected = np.argsort(scores)[:clients_per_round]

            return selected.tolist()

        if selection_mode == "adaptive":
            scores = []

            for client_id, profile in enumerate(client_profiles):
                score = (
                    profile["delay_scale"] +
                    profile["loss_scale"] +
                    np.random.uniform(0.0, 0.15)
                )

                scores.append(score)

            selected = np.argsort(scores)[:clients_per_round]

            return selected.tolist()

        return np.random.choice(
            NUM_CLIENTS,
            size=clients_per_round,
            replace=False
        ).tolist()

    def simulate_network(client_id, config, transmitted_parameter_count):
        profile = client_profiles[client_id]

        delay = np.random.uniform(
            config["delay_low"],
            config["delay_high"]
        ) * profile["delay_scale"]

        congestion = np.random.uniform(
            config["congestion_low"],
            config["congestion_high"]
        )

        packet_loss_probability = min(
            0.95,
            config["packet_loss"] * profile["loss_scale"] * congestion
        )

        received = np.random.rand() > packet_loss_probability
        bandwidth_kb = (transmitted_parameter_count * 4.0) / 1024.0
        communication_cost = bandwidth_kb * delay * congestion

        energy = (
            0.006 * bandwidth_kb +
            0.035 * delay +
            0.004 * config["local_epochs"]
        ) * profile["energy_scale"]

        return {
            "delay": float(delay),
            "congestion": float(congestion),
            "packet_loss_probability": float(packet_loss_probability),
            "received": bool(received),
            "bandwidth_kb": float(bandwidth_kb),
            "communication_cost": float(communication_cost),
            "energy": float(energy)
        }

    def compute_sync_latency(delays, selected_count, mode):
        if len(delays) == 0:
            return 0.0

        delays = np.array(delays)

        if mode == "synchronous":
            return float(np.max(delays) + 0.015 * selected_count)

        if mode == "edge_async":
            return float(np.percentile(delays, 80) + 0.008 * selected_count)

        if mode == "adaptive_semi_async":
            return float(np.percentile(delays, 65) + 0.004 * selected_count)

        return float(np.mean(delays) + 0.010 * selected_count)

    # ======================================================
    # RUN FEDERATED SIMULATION
    # ======================================================
    initial_model_vector = initialize_model_vector()
    results = {}

    for algorithm_name in algorithms:
        print(f"\nRunning algorithm: {algorithm_name} on {dataset_display_name}")

        config = algorithm_configs[algorithm_name]
        global_model = initial_model_vector.copy()
        residual_buffers = [
            np.zeros(model_size, dtype=np.float32)
            for _ in range(NUM_CLIENTS)
        ]

        results[algorithm_name] = {
            "test_loss": [],
            "test_accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "delay": [],
            "sync_latency": [],
            "packet_loss": [],
            "packet_success": [],
            "received_updates": [],
            "dropped_updates": [],
            "bandwidth": [],
            "communication_cost": [],
            "energy": [],
            "compression_ratio": [],
            "size_reduction": [],
            "update_variance": [],
            "update_norm": []
        }

        for round_id in range(ROUNDS):
            selected_clients = select_clients(config)
            received_items = []
            round_updates_for_variance = []

            delays = []
            bandwidths = []
            communication_costs = []
            energies = []
            compression_ratios = []

            received_count = 0
            dropped_count = 0

            for client_id in selected_clients:
                client_data = clients[client_id]

                raw_update = local_train(
                    global_model,
                    client_data["x"],
                    client_data["y"],
                    config["local_epochs"],
                    config["learning_rate"]
                )

                if config["residual"]:
                    update_for_compression = raw_update + residual_buffers[client_id]
                else:
                    update_for_compression = raw_update

                compressed_update, transmitted_count = top_k_compress(
                    update_for_compression,
                    config["compression_ratio"]
                )

                network_result = simulate_network(
                    client_id,
                    config,
                    transmitted_count
                )

                delays.append(network_result["delay"])
                bandwidths.append(network_result["bandwidth_kb"])
                communication_costs.append(network_result["communication_cost"])
                energies.append(network_result["energy"])
                compression_ratios.append(transmitted_count / model_size)

                if network_result["received"]:
                    received_count += 1

                    if config["residual"]:
                        residual_buffers[client_id] = (
                            update_for_compression - compressed_update
                        ).astype(np.float32)

                    received_items.append({
                        "update": compressed_update.astype(np.float32),
                        "weight": client_data["size"],
                        "edge_id": client_id % NUM_EDGE_SERVERS
                    })

                    round_updates_for_variance.append(compressed_update)

                else:
                    dropped_count += 1

                    if config["residual"]:
                        residual_buffers[client_id] = update_for_compression.astype(np.float32)

            if config["edge"]:
                aggregated_update = edge_aggregate(received_items)
            else:
                aggregated_update = weighted_average_updates(
                    [item["update"] for item in received_items],
                    [item["weight"] for item in received_items]
                )

            global_model = global_model + config.get("server_update_gain", 1.0) * aggregated_update

            eval_metrics = evaluate_model(global_model, x_test, y_test)

            if len(round_updates_for_variance) > 1:
                update_matrix = np.vstack(round_updates_for_variance)
                update_variance_value = float(np.mean(np.var(update_matrix, axis=0)))
            else:
                update_variance_value = 0.0

            selected_count = len(selected_clients)
            packet_loss_rate = (dropped_count / selected_count) * 100.0
            packet_success_rate = (received_count / selected_count) * 100.0

            avg_delay = float(np.mean(delays)) if delays else 0.0
            avg_bandwidth = float(np.sum(bandwidths))
            avg_communication_cost = float(np.sum(communication_costs))
            avg_energy = float(np.sum(energies))
            avg_compression_ratio = float(np.mean(compression_ratios)) if compression_ratios else 0.0
            size_reduction = (1.0 - avg_compression_ratio) * 100.0

            sync_latency = compute_sync_latency(
                delays,
                selected_count,
                config["latency_mode"]
            )

            results[algorithm_name]["test_loss"].append(eval_metrics["loss"])
            results[algorithm_name]["test_accuracy"].append(eval_metrics["accuracy"])
            results[algorithm_name]["precision"].append(eval_metrics["precision"])
            results[algorithm_name]["recall"].append(eval_metrics["recall"])
            results[algorithm_name]["f1_score"].append(eval_metrics["f1_score"])
            results[algorithm_name]["delay"].append(avg_delay)
            results[algorithm_name]["sync_latency"].append(sync_latency)
            results[algorithm_name]["packet_loss"].append(packet_loss_rate)
            results[algorithm_name]["packet_success"].append(packet_success_rate)
            results[algorithm_name]["received_updates"].append(received_count)
            results[algorithm_name]["dropped_updates"].append(dropped_count)
            results[algorithm_name]["bandwidth"].append(avg_bandwidth)
            results[algorithm_name]["communication_cost"].append(avg_communication_cost)
            results[algorithm_name]["energy"].append(avg_energy)
            results[algorithm_name]["compression_ratio"].append(avg_compression_ratio)
            results[algorithm_name]["size_reduction"].append(size_reduction)
            results[algorithm_name]["update_variance"].append(update_variance_value)
            results[algorithm_name]["update_norm"].append(float(np.linalg.norm(aggregated_update)))

            print(
                f"Round {round_id + 1:02d} | "
                f"Loss: {eval_metrics['loss']:.4f} | "
                f"Acc: {eval_metrics['accuracy']:.2f}% | "
                f"P: {eval_metrics['precision']:.2f}% | "
                f"R: {eval_metrics['recall']:.2f}% | "
                f"F1: {eval_metrics['f1_score']:.2f}% | "
                f"Delay: {avg_delay:.3f} | "
                f"LossRate: {packet_loss_rate:.1f}%"
            )

    # ======================================================
    # SCIENTIFIC SUMMARY TABLE AND FIGURES
    # ======================================================
    def final_value(algorithm_name, metric_name):
        return float(results[algorithm_name][metric_name][-1])

    def average_value(algorithm_name, metric_name):
        return float(np.mean(results[algorithm_name][metric_name]))

    def loss_stability_value(algorithm_name, window=8):
        recent_losses = np.array(results[algorithm_name]["test_loss"][-window:])
        return float(np.var(recent_losses))

    def communication_efficiency_value(algorithm_name):
        # Higher is better: accuracy obtained per unit communication cost.
        acc = final_value(algorithm_name, "test_accuracy")
        cost = average_value(algorithm_name, "communication_cost")
        return float(acc / (cost + 1e-12))

    def metric_for_score(algorithm_name, metric_name, use_final=False):
        if metric_name == "loss_stability":
            return loss_stability_value(algorithm_name)

        if metric_name == "communication_efficiency":
            return communication_efficiency_value(algorithm_name)

        if use_final:
            return final_value(algorithm_name, metric_name)

        return average_value(algorithm_name, metric_name)

    def normalized_scores(metric_name, higher_is_better, use_final=False):
        values = np.array([
            metric_for_score(algorithm_name, metric_name, use_final=use_final)
            for algorithm_name in algorithms
        ], dtype=np.float64)

        min_value = np.min(values)
        max_value = np.max(values)

        if abs(max_value - min_value) < 1e-12:
            return np.ones_like(values) * 0.5

        if higher_is_better:
            return (values - min_value) / (max_value - min_value)

        return (max_value - values) / (max_value - min_value)

    def composite_efficiency_scores():
        score_specs = [
            ("test_accuracy", True, True),
            ("precision", True, True),
            ("recall", True, True),
            ("f1_score", True, True),
            ("test_loss", False, True),
            ("delay", False, False),
            ("sync_latency", False, False),
            ("energy", False, False),
            ("bandwidth", False, False),
            ("communication_cost", False, False),
            ("packet_loss", False, False),
            ("packet_success", True, False),
            ("size_reduction", True, False),
            ("loss_stability", False, False)
        ]

        all_scores = []

        for metric_name, higher_is_better, use_final in score_specs:
            all_scores.append(
                normalized_scores(
                    metric_name,
                    higher_is_better=higher_is_better,
                    use_final=use_final
                )
            )

        score_matrix = np.vstack(all_scores)
        scores = np.mean(score_matrix, axis=0) * 100.0

        return {
            algorithm_name: float(score)
            for algorithm_name, score in zip(algorithms, scores)
        }

    composite_scores = composite_efficiency_scores()

    # ======================================================
    # SAVE SUMMARY TABLE
    # ======================================================
    summary_path = os.path.join(data_output_dir, f"{dataset_tag}_summary_results.csv")

    with open(summary_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Dataset",
            "Algorithm",
            "Final Test Accuracy (%)",
            "Final Macro Precision (%)",
            "Final Macro Recall (%)",
            "Final Macro F1-score (%)",
            "Final Test Loss",
            "Average Delay",
            "Average Synchronization Latency",
            "Average Energy Consumption",
            "Average Bandwidth (KB)",
            "Average Communication Cost",
            "Average Packet Loss (%)",
            "Average Packet Success (%)",
            "Average Update Size Reduction (%)",
            "Loss Stability (Variance of Last Rounds)",
            "Communication Efficiency",
            "Composite Efficiency Score"
        ])

        for algorithm_name in algorithms:
            writer.writerow([
                dataset_display_name,
                algorithm_name,
                final_value(algorithm_name, "test_accuracy"),
                final_value(algorithm_name, "precision"),
                final_value(algorithm_name, "recall"),
                final_value(algorithm_name, "f1_score"),
                final_value(algorithm_name, "test_loss"),
                average_value(algorithm_name, "delay"),
                average_value(algorithm_name, "sync_latency"),
                average_value(algorithm_name, "energy"),
                average_value(algorithm_name, "bandwidth"),
                average_value(algorithm_name, "communication_cost"),
                average_value(algorithm_name, "packet_loss"),
                average_value(algorithm_name, "packet_success"),
                average_value(algorithm_name, "size_reduction"),
                loss_stability_value(algorithm_name),
                communication_efficiency_value(algorithm_name),
                composite_scores[algorithm_name]
            ])

    print(f"\nSummary table saved to: {summary_path}")

    # ======================================================
    # FIGURE SETTINGS
    # ======================================================
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.75,
        "figure.dpi": 120,
        "savefig.dpi": 600
    })

    display_names = {
        "FedAvg": "FedAvg",
        "Simple FL": "Simple FL",
        "Compression-only": "Comp.-only",
        "Edge-only": "Edge-only",
        "Proposed": "Proposed"
    }

    colors = {
        "FedAvg": "#6F6F6F",
        "Simple FL": "#A7A7A7",
        "Compression-only": "#4E79A7",
        "Edge-only": "#59A14F",
        "Proposed": "#D55E00"
    }

    markers = {
        "FedAvg": "o",
        "Simple FL": "s",
        "Compression-only": "^",
        "Edge-only": "D",
        "Proposed": "*"
    }

    # ======================================================
    # FIGURE HELPER FUNCTIONS
    # ======================================================
    def clean_axis(ax, grid_axis="x"):
        ax.set_facecolor("white")

        if grid_axis is not None:
            ax.grid(axis=grid_axis, color="#E2E2E2", linewidth=0.65, alpha=0.9)
            ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.tick_params(axis="both", colors="#303030", length=3, width=0.75)

    def save_figure(fig, filename):
        png_path = os.path.join(output_dir, filename + ".png")
        pdf_path = os.path.join(output_dir, filename + ".pdf")
        svg_path = os.path.join(output_dir, filename + ".svg")

        fig.savefig(png_path, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")

        plt.close(fig)

    def format_value(value, decimals=2, scientific=False, suffix=""):
        if scientific:
            return f"{value:.2e}{suffix}"
        return f"{value:.{decimals}f}{suffix}"

    def plot_horizontal_bar(values_dict, xlabel, filename, decimals=2, scientific=False, suffix=""):
        ordered_algorithms = list(reversed(algorithms))
        labels = [display_names[name] for name in ordered_algorithms]
        values = np.array([values_dict[name] for name in ordered_algorithms], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(4.8, 2.85), facecolor="white")

        y_pos = np.arange(len(ordered_algorithms))
        bar_colors = [colors[name] for name in ordered_algorithms]

        ax.barh(
            y_pos,
            values,
            height=0.56,
            color=bar_colors,
            edgecolor="#333333",
            linewidth=0.45
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)

        clean_axis(ax, grid_axis="x")

        max_value = float(np.max(values))

        if max_value <= 0:
            ax.set_xlim(0, 1)
            label_shift = 0.02
        else:
            ax.set_xlim(0, max_value * 1.18)
            label_shift = max_value * 0.025

        for y, value in zip(y_pos, values):
            ax.text(
                value + label_shift,
                y,
                format_value(value, decimals=decimals, scientific=scientific, suffix=suffix),
                va="center",
                ha="left",
                fontsize=7,
                color="#222222"
            )

        fig.tight_layout()
        save_figure(fig, filename)

    def plot_dot_metric(values_dict, xlabel, filename, decimals=2, suffix=""):
        ordered_algorithms = list(reversed(algorithms))
        labels = [display_names[name] for name in ordered_algorithms]
        values = np.array([values_dict[name] for name in ordered_algorithms], dtype=np.float64)

        fig, ax = plt.subplots(figsize=(4.8, 2.85), facecolor="white")
        y_pos = np.arange(len(ordered_algorithms))

        for y, algorithm_name, value in zip(y_pos, ordered_algorithms, values):
            ax.scatter(
                value,
                y,
                s=52 if algorithm_name != "Proposed" else 78,
                color=colors[algorithm_name],
                edgecolor="#222222",
                linewidth=0.45,
                zorder=3,
                marker=markers[algorithm_name]
            )

            ax.text(
                value,
                y + 0.18,
                format_value(value, decimals=decimals, suffix=suffix),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222"
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)

        value_range = float(np.max(values) - np.min(values))

        if value_range < 1e-9:
            margin = max(0.5, abs(float(np.mean(values))) * 0.01)
        else:
            margin = value_range * 0.28

        ax.set_xlim(float(np.min(values)) - margin, float(np.max(values)) + margin)
        clean_axis(ax, grid_axis="x")

        fig.tight_layout()
        save_figure(fig, filename)

    def plot_line(metric_name, ylabel, filename):
        fig, ax = plt.subplots(figsize=(5.2, 3.15), facecolor="white")
        rounds_axis = np.arange(1, ROUNDS + 1)

        for algorithm_name in algorithms:
            ax.plot(
                rounds_axis,
                results[algorithm_name][metric_name],
                label=display_names[algorithm_name],
                color=colors[algorithm_name],
                marker=markers[algorithm_name],
                linewidth=1.25,
                markersize=3.0,
                markevery=3
            )

        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        clean_axis(ax, grid_axis="y")

        ax.legend(
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            handlelength=2.0,
            columnspacing=1.2
        )

        fig.tight_layout()
        save_figure(fig, filename)

    def plot_improvement_over_fedavg(filename):
        baseline = "FedAvg"
        proposed = "Proposed"

        comparison_specs = [
            ("Accuracy", "test_accuracy", True, True),
            ("Precision", "precision", True, True),
            ("Recall", "recall", True, True),
            ("F1-score", "f1_score", True, True),
            ("Test loss", "test_loss", False, True),
            ("Delay", "delay", False, False),
            ("Latency", "sync_latency", False, False),
            ("Energy", "energy", False, False),
            ("Bandwidth", "bandwidth", False, False),
            ("Cost", "communication_cost", False, False),
            ("Packet loss", "packet_loss", False, False)
        ]

        labels = []
        values = []

        for label, metric_name, higher_is_better, use_final in comparison_specs:
            labels.append(label)
            base_value = metric_for_score(baseline, metric_name, use_final=use_final)
            proposed_value = metric_for_score(proposed, metric_name, use_final=use_final)

            if abs(base_value) < 1e-12:
                improvement = 0.0
            elif higher_is_better:
                improvement = ((proposed_value - base_value) / base_value) * 100.0
            else:
                improvement = ((base_value - proposed_value) / base_value) * 100.0

            values.append(improvement)

        values_dict = {label: value for label, value in zip(labels, values)}
        ordered_labels = list(reversed(labels))
        ordered_values = np.array([values_dict[label] for label in ordered_labels])

        fig, ax = plt.subplots(figsize=(5.1, 3.35), facecolor="white")
        y_pos = np.arange(len(ordered_labels))
        bar_colors = ["#D55E00" if value >= 0 else "#777777" for value in ordered_values]

        ax.barh(
            y_pos,
            ordered_values,
            height=0.55,
            color=bar_colors,
            edgecolor="#333333",
            linewidth=0.45
        )

        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_labels)
        ax.set_xlabel("Improvement over FedAvg (%)")
        clean_axis(ax, grid_axis="x")

        max_abs = float(np.max(np.abs(ordered_values)))
        if max_abs < 1e-9:
            max_abs = 1.0

        ax.set_xlim(-max_abs * 0.25, max_abs * 1.20)

        for y, value in zip(y_pos, ordered_values):
            ha = "left" if value >= 0 else "right"
            shift = max_abs * 0.025 if value >= 0 else -max_abs * 0.025
            ax.text(
                value + shift,
                y,
                f"{value:.1f}",
                va="center",
                ha=ha,
                fontsize=7,
                color="#222222"
            )

        fig.tight_layout()
        save_figure(fig, filename)

    # ======================================================
    # PREPARE FIGURE DATA
    # ======================================================
    final_accuracy = {algorithm_name: final_value(algorithm_name, "test_accuracy") for algorithm_name in algorithms}
    final_precision = {algorithm_name: final_value(algorithm_name, "precision") for algorithm_name in algorithms}
    final_recall = {algorithm_name: final_value(algorithm_name, "recall") for algorithm_name in algorithms}
    final_f1 = {algorithm_name: final_value(algorithm_name, "f1_score") for algorithm_name in algorithms}
    final_loss = {algorithm_name: final_value(algorithm_name, "test_loss") for algorithm_name in algorithms}
    avg_delay = {algorithm_name: average_value(algorithm_name, "delay") for algorithm_name in algorithms}
    avg_latency = {algorithm_name: average_value(algorithm_name, "sync_latency") for algorithm_name in algorithms}
    avg_energy = {algorithm_name: average_value(algorithm_name, "energy") for algorithm_name in algorithms}
    avg_bandwidth = {algorithm_name: average_value(algorithm_name, "bandwidth") for algorithm_name in algorithms}
    avg_comm_cost = {algorithm_name: average_value(algorithm_name, "communication_cost") for algorithm_name in algorithms}
    avg_packet_loss = {algorithm_name: average_value(algorithm_name, "packet_loss") for algorithm_name in algorithms}
    avg_packet_success = {algorithm_name: average_value(algorithm_name, "packet_success") for algorithm_name in algorithms}
    avg_size_reduction = {algorithm_name: average_value(algorithm_name, "size_reduction") for algorithm_name in algorithms}
    loss_stability = {algorithm_name: loss_stability_value(algorithm_name) for algorithm_name in algorithms}
    communication_efficiency = {algorithm_name: communication_efficiency_value(algorithm_name) for algorithm_name in algorithms}

    # ======================================================
    # SAVE FIGURES
    # ======================================================
    plot_line("test_accuracy", "Test accuracy (%)", "fig01_accuracy_convergence")
    plot_line("test_loss", "Test loss", "fig02_loss_convergence")
    plot_line("f1_score", "Macro F1-score (%)", "fig03_f1_convergence")

    plot_dot_metric(final_accuracy, "Final test accuracy (%)", "fig04_final_test_accuracy", decimals=2, suffix="%")
    plot_dot_metric(final_precision, "Final macro precision (%)", "fig05_final_macro_precision", decimals=2, suffix="%")
    plot_dot_metric(final_recall, "Final macro recall (%)", "fig06_final_macro_recall", decimals=2, suffix="%")
    plot_dot_metric(final_f1, "Final macro F1-score (%)", "fig07_final_macro_f1_score", decimals=2, suffix="%")
    plot_dot_metric(final_loss, "Final test loss", "fig08_final_test_loss", decimals=4)

    plot_horizontal_bar(avg_delay, "Average communication delay", "fig09_average_communication_delay", decimals=3)
    plot_horizontal_bar(avg_latency, "Average synchronization latency", "fig10_average_synchronization_latency", decimals=3)
    plot_horizontal_bar(avg_energy, "Average energy consumption", "fig11_average_energy_consumption", decimals=3)
    plot_horizontal_bar(avg_bandwidth, "Average transmitted data (KB)", "fig12_average_bandwidth_usage", decimals=2)
    plot_horizontal_bar(avg_comm_cost, "Average communication cost", "fig13_average_communication_cost", decimals=2)
    plot_horizontal_bar(avg_packet_loss, "Average packet loss (%)", "fig14_average_packet_loss", decimals=2, suffix="%")
    plot_horizontal_bar(avg_packet_success, "Average packet success (%)", "fig15_average_packet_success", decimals=2, suffix="%")
    plot_horizontal_bar(avg_size_reduction, "Average update size reduction (%)", "fig16_average_update_size_reduction", decimals=1, suffix="%")
    plot_horizontal_bar(loss_stability, "Loss stability variance", "fig17_loss_stability", scientific=True)
    plot_horizontal_bar(communication_efficiency, "Accuracy per communication cost", "fig18_communication_efficiency", decimals=3)
    plot_horizontal_bar(composite_scores, "Composite efficiency score", "fig19_composite_efficiency_score", decimals=1)
    plot_improvement_over_fedavg("fig20_proposed_improvement_over_fedavg")

    print(f"\nAll figures saved successfully for {dataset_display_name}.")
    print(f"Figures folder: {output_dir}")
    print(f"Summary CSV: {summary_path}")

    summary_rows = []
    for algorithm_name in algorithms:
        summary_rows.append([
            dataset_display_name,
            algorithm_name,
            final_value(algorithm_name, "test_accuracy"),
            final_value(algorithm_name, "precision"),
            final_value(algorithm_name, "recall"),
            final_value(algorithm_name, "f1_score"),
            final_value(algorithm_name, "test_loss"),
            average_value(algorithm_name, "delay"),
            average_value(algorithm_name, "sync_latency"),
            average_value(algorithm_name, "energy"),
            average_value(algorithm_name, "bandwidth"),
            average_value(algorithm_name, "communication_cost"),
            average_value(algorithm_name, "packet_loss"),
            average_value(algorithm_name, "packet_success"),
            average_value(algorithm_name, "size_reduction"),
            loss_stability_value(algorithm_name),
            communication_efficiency_value(algorithm_name),
            composite_scores[algorithm_name]
        ])

    return summary_rows



# ==========================================================
# ABLATION FIGURES IN A SEPARATE FOLDER
# ==========================================================
def save_ablation_outputs(all_summary_rows):
    """
    Builds additional ablation figures from the final summary rows.

    These figures are saved separately from the main result figures in:
        ablation_figures_mnist_femnist/

    The original figures are not removed or changed.
    """
    os.makedirs(BASE_ABLATION_OUTPUT_DIR, exist_ok=True)

    header = [
        "Dataset",
        "Algorithm",
        "Final Test Accuracy (%)",
        "Final Macro Precision (%)",
        "Final Macro Recall (%)",
        "Final Macro F1-score (%)",
        "Final Test Loss",
        "Average Delay",
        "Average Synchronization Latency",
        "Average Energy Consumption",
        "Average Bandwidth (KB)",
        "Average Communication Cost",
        "Average Packet Loss (%)",
        "Average Packet Success (%)",
        "Average Update Size Reduction (%)",
        "Loss Stability (Variance of Last Rounds)",
        "Communication Efficiency",
        "Composite Efficiency Score"
    ]

    ablation_summary_path = os.path.join(
        BASE_ABLATION_OUTPUT_DIR,
        "ablation_summary_results.csv"
    )

    with open(ablation_summary_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(all_summary_rows)

    records = []
    for row in all_summary_rows:
        records.append({
            "Dataset": row[0],
            "Algorithm": row[1],
            "Final Test Accuracy (%)": float(row[2]),
            "Final Macro Precision (%)": float(row[3]),
            "Final Macro Recall (%)": float(row[4]),
            "Final Macro F1-score (%)": float(row[5]),
            "Final Test Loss": float(row[6]),
            "Average Delay": float(row[7]),
            "Average Synchronization Latency": float(row[8]),
            "Average Energy Consumption": float(row[9]),
            "Average Bandwidth (KB)": float(row[10]),
            "Average Communication Cost": float(row[11]),
            "Average Packet Loss (%)": float(row[12]),
            "Average Packet Success (%)": float(row[13]),
            "Average Update Size Reduction (%)": float(row[14]),
            "Loss Stability (Variance of Last Rounds)": float(row[15]),
            "Communication Efficiency": float(row[16]),
            "Composite Efficiency Score": float(row[17])
        })

    method_order = [
        "FedAvg",
        "Simple FL",
        "Compression-only",
        "Edge-only",
        "Proposed"
    ]

    display_names = {
        "FedAvg": "FedAvg",
        "Simple FL": "Simple FL",
        "Compression-only": "Comp.-only",
        "Edge-only": "Edge-only",
        "Proposed": "Proposed"
    }

    colors = {
        "FedAvg": "#6F6F6F",
        "Simple FL": "#A7A7A7",
        "Compression-only": "#4E79A7",
        "Edge-only": "#59A14F",
        "Proposed": "#D55E00"
    }

    markers = {
        "FedAvg": "o",
        "Simple FL": "s",
        "Compression-only": "^",
        "Edge-only": "D",
        "Proposed": "*"
    }

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.75,
        "figure.dpi": 120,
        "savefig.dpi": 600
    })

    def clean_axis(ax):
        ax.set_facecolor("white")
        ax.grid(axis="x", color="#E2E2E2", linewidth=0.65, alpha=0.9)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444444")
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_linewidth(0.75)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.tick_params(axis="both", colors="#303030", length=3, width=0.75)

    def save_figure(fig, filename):
        png_path = os.path.join(BASE_ABLATION_OUTPUT_DIR, filename + ".png")
        pdf_path = os.path.join(BASE_ABLATION_OUTPUT_DIR, filename + ".pdf")
        svg_path = os.path.join(BASE_ABLATION_OUTPUT_DIR, filename + ".svg")

        fig.savefig(png_path, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def safe_filename(text_value):
        return (
            text_value
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("compatible_", "")
            .replace("__", "_")
        )

    def get_dataset_records(dataset_name):
        dataset_records = [record for record in records if record["Dataset"] == dataset_name]
        return {
            record["Algorithm"]: record
            for record in dataset_records
        }

    def format_value(value, decimals=2, suffix=""):
        return f"{value:.{decimals}f}{suffix}"

    def plot_dot_panel(ax, dataset_records, metric_name, xlabel, decimals=2, suffix=""):
        ordered_methods = list(reversed(method_order))
        y_pos = np.arange(len(ordered_methods))
        values = []

        for method in ordered_methods:
            value = dataset_records[method][metric_name]
            values.append(value)

            ax.scatter(
                value,
                y_pos[ordered_methods.index(method)],
                s=52 if method != "Proposed" else 78,
                color=colors[method],
                edgecolor="#222222",
                linewidth=0.45,
                zorder=3,
                marker=markers[method]
            )

            ax.text(
                value,
                y_pos[ordered_methods.index(method)] + 0.18,
                format_value(value, decimals=decimals, suffix=suffix),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222"
            )

        values = np.array(values, dtype=np.float64)
        value_range = float(np.max(values) - np.min(values))
        if value_range < 1e-9:
            margin = max(0.5, abs(float(np.mean(values))) * 0.01)
        else:
            margin = value_range * 0.28

        ax.set_xlim(float(np.min(values)) - margin, float(np.max(values)) + margin)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([display_names[method] for method in ordered_methods])
        ax.set_xlabel(xlabel)
        clean_axis(ax)

    def plot_three_panel_ablation(dataset_name, figure_type, metric_specs, filename):
        dataset_records = get_dataset_records(dataset_name)

        missing_methods = [
            method for method in method_order
            if method not in dataset_records
        ]

        if missing_methods:
            print(
                f"WARNING: Skipping {filename}; missing methods for {dataset_name}: "
                f"{missing_methods}"
            )
            return

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12.8, 3.15),
            facecolor="white"
        )

        for ax, spec in zip(axes, metric_specs):
            metric_name, xlabel, decimals, suffix = spec
            plot_dot_panel(
                ax,
                dataset_records,
                metric_name,
                xlabel,
                decimals=decimals,
                suffix=suffix
            )

        fig.tight_layout(w_pad=2.0)
        save_figure(fig, filename)

    dataset_names = []
    for row in all_summary_rows:
        if row[0] not in dataset_names:
            dataset_names.append(row[0])

    learning_specs = [
        ("Final Test Accuracy (%)", "Final test accuracy (%)", 2, "%"),
        ("Final Test Loss", "Final test loss", 4, ""),
        ("Final Macro F1-score (%)", "Final macro F1-score (%)", 2, "%")
    ]

    communication_specs = [
        ("Average Communication Cost", "Average communication cost", 2, ""),
        ("Average Packet Loss (%)", "Average packet loss (%)", 2, "%"),
        ("Composite Efficiency Score", "Composite efficiency score", 1, "")
    ]

    for dataset_name in dataset_names:
        dataset_filename = safe_filename(dataset_name)

        plot_three_panel_ablation(
            dataset_name=dataset_name,
            figure_type="learning",
            metric_specs=learning_specs,
            filename=f"ablation_learning_{dataset_filename}"
        )

        plot_three_panel_ablation(
            dataset_name=dataset_name,
            figure_type="communication",
            metric_specs=communication_specs,
            filename=f"ablation_communication_{dataset_filename}"
        )

    print(f"\nAblation figures saved successfully.")
    print(f"Ablation folder: {BASE_ABLATION_OUTPUT_DIR}")
    print(f"Ablation summary CSV: {ablation_summary_path}")



# ==========================================================
# RUN ALL DATASETS AND SAVE COMBINED SUMMARY
# ==========================================================
all_summary_rows = []
for config in DATASET_CONFIGS:
    all_summary_rows.extend(run_experiment(config))

combined_summary_path = os.path.join(BASE_DATA_OUTPUT_DIR, "all_datasets_summary_results.csv")
with open(combined_summary_path, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Dataset",
        "Algorithm",
        "Final Test Accuracy (%)",
        "Final Macro Precision (%)",
        "Final Macro Recall (%)",
        "Final Macro F1-score (%)",
        "Final Test Loss",
        "Average Delay",
        "Average Synchronization Latency",
        "Average Energy Consumption",
        "Average Bandwidth (KB)",
        "Average Communication Cost",
        "Average Packet Loss (%)",
        "Average Packet Success (%)",
        "Average Update Size Reduction (%)",
        "Loss Stability (Variance of Last Rounds)",
        "Communication Efficiency",
        "Composite Efficiency Score"
    ])
    writer.writerows(all_summary_rows)

save_ablation_outputs(all_summary_rows)

print("\nAll MNIST and FEMNIST-compatible EMNIST experiments completed successfully.")
print(f"Base figures folder: {BASE_OUTPUT_DIR}")
print(f"Base data folder: {BASE_DATA_OUTPUT_DIR}")
print(f"Ablation figures folder: {BASE_ABLATION_OUTPUT_DIR}")
print(f"Combined summary CSV: {combined_summary_path}")
