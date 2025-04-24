def train_pytorch():
    import os

    import torch
    from torch import nn
    import torch.nn.functional as F

    from torchvision import datasets, transforms
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    # [1] Configure CPU/GPU device and distributed backend.
    # Kubeflow Trainer will automatically configure the distributed environment.
    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    dist.init_process_group(backend=backend)

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    print(
        "Distributed Training with WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}.".format(
            dist.get_world_size(),
            dist.get_rank(),
            local_rank,
        )
    )

    # [2] Define PyTorch CNN Model to be trained.
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # [3] Attach model to the correct device.
    device = torch.device(f"{device}:{local_rank}")
    model = nn.parallel.DistributedDataParallel(Net().to(device))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # [4] Get the Fashion-MNIST dataset and distributed it across all available devices.
    dataset = datasets.FashionMNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=100,
        sampler=DistributedSampler(dataset),
    )

    # [5] Define the training loop.
    for epoch in range(3):
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Attach tensors to the device.
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

    # Wait for the training to complete and destroy to PyTorch distributed process group.
    dist.barrier()
    if dist.get_rank() == 0:
        print("Training is finished")
    dist.destroy_process_group()



from kubeflow.trainer.job_runners.podman_job_runner import PodmanJobRunner

# Constants
JOB_STARTUP_WAIT_SECONDS = 5  # Time to wait for job to start before getting logs

import platform

# TODO: This could be improved
def find_podman_socket():
    system = platform.system()

    if system == "Linux":
        uid = os.getuid()
        return f"unix:///run/user/{uid}/podman/podman.sock"

    elif system == "Darwin":  # macOS
        user_home = os.path.expanduser("~")
        return f"unix://{user_home}/.local/share/containers/podman/machine/podman.sock"

    elif system == "Windows":
        # Windows usually uses a named pipe
        return "npipe:////./pipe/podman-machine-default"

    else:
        raise RuntimeError(f"Unsupported operating system: {system}")


import os
import podman
import docker

from kubeflow.trainer import LocalTrainerClient, TrainerClient, CustomTrainer
from kubeflow.trainer.types import types

exec_mode = os.getenv("KUBEFLOW_TRAINER_EXEC_MODE", "local")

podman_socket = find_podman_socket()
print("socket: " + podman_socket)
client = LocalTrainerClient(
    job_runner=PodmanJobRunner(podman_client=podman.PodmanClient(
        base_url=podman_socket
    ))
)

# Get and print available runtimes
runtimes = client.list_runtimes()
print("\nAvailable runtimes:")
for runtime in runtimes:
    print(f"- {runtime.name}")

# Get a specific runtime
runtime = client.get_runtime("torch-distributed")
print(f"\nSelected runtime: {runtime.name}")

# Create a training job
job_name = client.train(
    runtime=runtime,
    trainer=CustomTrainer(
        func=train_pytorch,
        num_nodes=6,
    )
)
print(f"\nJob Running: {job_name}")

# Get detailed job information
print("\nGetting detailed job information:")
job = client.get_job(job_name)
print(f"Found Job: {job.name}")
print(f"Listing steps: {job.name}")
for step in job.steps:
    print(f"    - {step.name} ({step.status})")

# List and print all jobs for the runtime
jobs = client.list_jobs(runtime=runtime)
print("\nCurrent jobs:")
for job in jobs:
    print(f"Listing all Jobs: {job.name}")
    print(f"  Steps:")
    for step in job.steps:
        print(f"    - {step.name} ({step.status})")
    print()

# Get logs for the job
print("\nGetting job logs:")
# Wait for job to start and initialize
import time
time.sleep(JOB_STARTUP_WAIT_SECONDS)
logs = client.get_job_logs(job_name, follow=False, step="node", node_rank=0)
print("Logs for node 0:")
for step, log in logs.items():
    print(f"{step}:")
    print(log)

# Get logs with follow=True to see real-time output
print("\nFollowing job logs (press Ctrl+C to stop):")
try:
    client.get_job_logs(job_name, follow=True, step="node", node_rank=0)
except KeyboardInterrupt:
    print("\nStopped following logs")


time.sleep(120)
# Delete the job
print(f"\nDeleting job: {job_name}")
client.delete_job(job_name)

# Print jobs after deletion to verify
print("\nJobs after deletion:")
remaining_jobs = client.list_jobs(runtime=runtime)
for job in remaining_jobs:
    print(f"Job: {job.name}")
    print(f"  Status: {job.status}")
    print(f"  Runtime: {job.runtime.name}")
    print()


