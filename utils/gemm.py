import mlflow
import torch
from torch import float32, int8, int16, int32

bits = 8
alpha_q, beta_q = -2 ** (bits - 1), 2 ** (bits - 1) - 1


def quantization(x, s, z):
    x_q = torch.round(1 / s * x + z)
    x_q = torch.clamp(x_q, alpha_q, beta_q)

    return x_q


def quantization_int8(x, s, z):
    x_q = quantization(x, s, z)
    x_q = x_q.to(int8)
    return x_q


def dequantization(x_q, s, z):
    # x_q - z might go outside the quantization range.
    x_q = x_q.int()
    x = s * (x_q - z)
    x = x.to(float32)

    return x


def generate_quantization_constants(alpha, beta):
    # Affine quantization mapping
    s = (beta - alpha) / (beta_q - alpha_q)
    z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))

    return s, z


def generate_quantization_int8_constants(alpha, beta):
    b = 8

    s, z = generate_quantization_constants(alpha=alpha, beta=beta)
    return s, z


def quantization_matrix_multiplication_int8(A_q, B_q, s_A, z_A, s_B, z_B, s_Y, z_Y):
    p = B_q.shape[0]

    # Y_q_simulated is FP32
    Y_q_simulated = torch.zeros(A_q.shape[0], B_q.shape[1], dtype=int32)
    # outer produce

    for k in range(p):
        Y_q_simulated += torch.einsum("i,j->ij", (A_q[:, k].to(int) - z_A), (B_q[k, :].to(int) - z_B))

    Y_q_simulated = s_A * s_B * Y_q_simulated / s_Y + z_Y

    Y_q_simulated = torch.round(Y_q_simulated)
    Y_q_simulated = torch.clamp(Y_q_simulated, min=alpha_q, max=beta_q)
    Y_q_simulated = Y_q_simulated.to(int8)
    return Y_q_simulated


def generate_matrices(n, alpha_a, beta_a, alpha_b, beta_b, dist_type="uniform"):
    if dist_type == "uniform":
        a = torch.rand(n, n) * (beta_a - alpha_a) + alpha_a
        b = torch.rand(n, n) * (beta_b - alpha_b) + alpha_b
    elif dist_type == "normal":
        a = torch.randn(n, n) * beta_a + alpha_a
        b = torch.randn(n, n) * beta_b + alpha_b
    else:
        raise ValueError("Invalid dist_type")
    return a, b


def get_range(x, coverage=0.99):
    x_min = torch.quantile(x, 1 - coverage)
    x_max = torch.quantile(x, coverage)
    return x_min, x_max


def experiment(n=100, alpha_a=-1, beta_a=1, alpha_b=-1, beta_b=1, cover=0.99, random=0, dist_type="uniform"):
    # Set random seed for reproducibility
    torch.random.manual_seed(random)
    params = dict()

    # Generate Matrices
    A, B = generate_matrices(n, alpha_a, beta_a, alpha_b, beta_b, dist_type="uniform")
    Y = torch.matmul(A, B)

    # Find the range of matrices
    a_min, a_max = get_range(A, coverage=cover)
    b_min, b_max = get_range(B, coverage=cover)
    y_min, y_max = get_range(Y, coverage=cover)
    params["a_min"] = a_min.item()
    params["a_max"] = a_max.item()
    params["b_min"] = b_min.item()
    params["b_max"] = b_max.item()
    params["y_min"] = y_min.item()
    params["y_max"] = y_max.item()

    # Compute Quantization Constants
    s_A, z_A = generate_quantization_int8_constants(alpha=a_min, beta=a_max)
    s_B, z_B = generate_quantization_int8_constants(alpha=b_min, beta=b_max)
    s_Y, z_Y = generate_quantization_int8_constants(alpha=y_min, beta=y_max)
    params["s_A"] = s_A
    params["z_A"] = z_A
    params["s_B"] = s_B
    params["z_B"] = z_B
    params["s_Y"] = s_Y
    params["z_Y"] = z_Y

    # Compute Quantized Matrix
    A_q = quantization_int8(A, s_A, z_A)
    B_q = quantization_int8(B, s_B, z_B)
    Y_q = quantization_matrix_multiplication_int8(A_q, B_q, s_A, z_A, s_B, z_B, s_Y, z_Y)
    Y_q_expected = quantization_int8(Y, s_Y, z_Y)

    # Compute Dequantized Matrix
    Y_dq = dequantization(Y_q, s_Y, z_Y)
    Y_dq_expected = dequantization(Y_q_expected, s_Y, z_Y)

    # Store Results
    params["Ground Truth Error"] = (torch.norm(Y - Y_dq.float()) / torch.norm(Y)).item()
    params["Expected Error"] = (torch.norm(Y_dq_expected - Y_dq) / torch.norm(Y_dq_expected)).item()
    params["Simulated Error"] = (torch.norm(Y_q_expected.float() - Y_q.float()) / torch.norm(Y_q_expected.float())).item()

    return params


if __name__ == "__main__":
    client = mlflow.MlflowClient()

    # Ensure MLFlow server is running, otherwise start it locally
    tracking_uri = mlflow.get_tracking_uri()
    if tracking_uri is None or tracking_uri == "":
        mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Create a new experiment
    experiment_name = "gemm_matrix_size_outer_product"

    exp = client.search_experiments(filter_string="name='{}'".format(experiment_name))
    if len(exp) == 0:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = exp[0].experiment_id

    # Run the experiment
    for dist_type in ["uniform", "normal"]:
        for cover in [0.7, 0.9, 0.95, 0.99]:
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"dist_type={dist_type}, cover={cover}") as run:
                params = {"cover": cover, "dist_type": dist_type, "alpha_a": -1, "beta_a": 1, "alpha_b": -1, "beta_b": 1}
                mlflow.log_params(params)
                for n in [10, 100, 1000, 3000]:
                    params = experiment(n, alpha_a=-1, beta_a=1, alpha_b=-1, beta_b=1, cover=cover, random=0,
                                        dist_type=dist_type)
                    expected_error = params["Expected Error"]
                    simulated_error = params["Simulated Error"]
                    ground_truth_error = params["Ground Truth Error"]

                    # Log the results
                    mlflow.log_metric("s_A", params["s_A"], step=n)
                    mlflow.log_metric("z_A", params["z_A"], step=n)
                    mlflow.log_metric("s_B", params["s_B"], step=n)
                    mlflow.log_metric("z_B", params["z_B"], step=n)
                    mlflow.log_metric("s_Y", params["s_Y"], step=n)
                    mlflow.log_metric("z_Y", params["z_Y"], step=n)
                    mlflow.log_metric("Expected Error", expected_error, step=n)
                    mlflow.log_metric("Simulated Error", simulated_error, step=n)
                    mlflow.log_metric("Ground Truth Error", ground_truth_error, step=n)
                    print(f"Experiment completed for n={n}, dist_type={dist_type}, cover={cover} "
                          f"Expected Error={expected_error}, Simulated Error={simulated_error}, "
                          f"Ground Truth Error={ground_truth_error}")
            print(f"Experiment completed for dist_type={dist_type}, cover={cover}")