# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime
from typing import Dict, List, Optional
import time

import podman
from kubeflow.trainer.constants import constants
from kubeflow.trainer.job_runners.job_runner import JobRunner
from kubeflow.trainer.types import types
from kubeflow.trainer.utils import utils


class PodmanJobRunner(JobRunner):
    """Job runner implementation using Podman.

    This class implements the JobRunner interface using Podman as the container runtime.
    It provides functionality to create, manage, and monitor training jobs running in
    Podman containers.
    """

    def __init__(self, podman_client: Optional[podman.PodmanClient] = None):
        if podman_client is None:
            self.podman_client = podman.PodmanClient()
        else:
            self.podman_client = podman_client

    def create_job(
            self,
            image: str,
            entrypoint: List[str],
            command: List[str],
            num_nodes: int,
            framework: types.Framework,
            runtime_name: str,
    ) -> str:
        if framework != types.Framework.TORCH:
            raise RuntimeError(f"Framework '{framework}' is not currently supported.")

        train_job_name = (
            f"{constants.LOCAL_TRAIN_JOB_NAME_PREFIX}{utils.generate_train_job_name()}"
        )

        # Create a pod
        pod = self.podman_client.pods.create(
            name=train_job_name,
            publish=str(constants.TORCH_HEAD_NODE_PORT) + ":" + str(constants.TORCH_HEAD_NODE_PORT),
            labels={
                constants.CONTAINER_TRAIN_JOB_NAME_LABEL: train_job_name,
                constants.CONTAINER_RUNTIME_LABEL: runtime_name,
            }
        )
        print(f"Created pod: {train_job_name}")

        # Start the pod
        pod.start()
        # time.sleep(2)  # Wait for pod to be ready

        # Create containers within the pod
        containers = []
        container_ips = {}  # Store container IPs for extra_hosts

        for i in range(num_nodes):
            container_name = f"{train_job_name}-{i}"
            try:
                env = self.__get_container_environment(
                    framework=framework,
                    head_node_address=f"{train_job_name}-0",  # Use container name
                    num_nodes=num_nodes,
                    node_rank=i,
                )

                container = self.podman_client.containers.create(
                    name=container_name,
                    pod=train_job_name,
                    image=image,
                    entrypoint=entrypoint,
                    command=command,
                    labels={
                        constants.CONTAINER_TRAIN_JOB_NAME_LABEL: train_job_name,
                        constants.LOCAL_NODE_RANK_LABEL: str(i),
                        constants.CONTAINER_RUNTIME_LABEL: runtime_name,
                    },
                    environment=env,
                    detach=True,
                )
                print(f"Created container: {container_name}")
                container.start()
                # containers.append(container)
                
                # # Get container IP and store it
                # container.reload()
                # container_ip = container.attrs["NetworkSettings"]["IPAddress"]
                # container_ips[container_name] = container_ip
                # print(f"Container {container_name} IP: {container_ip}")
                # 
                # # Wait for container to be ready
                # time.sleep(2)
                
            except Exception as e:
                print(f"Error creating container {container_name}: {str(e)}")
                self.delete_job(train_job_name)
                raise

        return train_job_name

    def get_job(self, job_name: str) -> types.ContainerJob:
        try:
            pod = self.podman_client.pods.get(job_name)
        except Exception as e:
            raise RuntimeError(f"Could not find pod for job '{job_name}': {str(e)}")

        podman_containers = self.podman_client.containers.list(
            filters={"label": f"{constants.CONTAINER_TRAIN_JOB_NAME_LABEL}={job_name}"},
            all=True,
        )

        containers = []
        for container in podman_containers:
            status = "unknown"
            try:
                if isinstance(container.attrs, dict):
                    status = container.attrs.get("State", {}).get("Status", "unknown")
                else:
                    status = container.status
            except Exception:
                pass
            containers.append(
                types.Container(
                    name=container.name,
                    status=status,
                ),
            )

        return types.ContainerJob(
            name=job_name,
            creation_timestamp=datetime.now(),
            runtime_name=pod.attrs["Labels"][constants.CONTAINER_RUNTIME_LABEL],
            containers=containers,
            status=self.__get_job_status(containers),
        )

    def get_job_logs(
        self,
        job_name: str,
        follow: bool = False,
        step: str = constants.NODE,
        node_rank: int = 0,
    ) -> Dict[str, str]:
        containers = self.podman_client.containers.list(
            filters={"label": f"{constants.CONTAINER_TRAIN_JOB_NAME_LABEL}={job_name}"},
            all=True,
        )
        if len(containers) == 0:
            raise RuntimeError(f"Could not find container with job label '{job_name}'")

        # Sort containers by node rank to ensure consistent ordering
        containers.sort(key=lambda c: int(c.labels.get(constants.LOCAL_NODE_RANK_LABEL, "0")))

        logs: Dict[str, str] = {}
        for container in containers:
            container_rank = int(container.labels.get(constants.LOCAL_NODE_RANK_LABEL, "0"))
            if container_rank == node_rank:
                if follow:
                    for line in container.logs(stream=True):
                        decoded = line.decode("utf-8")
                        print(decoded)
                        logs[f"{step}-{node_rank}"] = (
                            logs.get(f"{step}-{node_rank}", "") + decoded + "\n"
                        )
                else:
                    # Get logs as a generator and process frames efficiently
                    log_generator = container.logs()
                    # Use list comprehension to collect decoded logs
                    decoded_logs = [frame.decode("utf-8") for frame in log_generator]
                    # Join all logs with newlines
                    log_content = "".join(decoded_logs)
                    # Print the logs
                    print(log_content, end="")
                    logs[f"{step}-{node_rank}"] = log_content
        return logs

    def list_jobs(
        self,
        runtime_name: Optional[str] = None,
    ) -> List[types.ContainerJob]:
        jobs = []
        for name in self.__list_job_names(runtime_name):
            jobs.append(self.get_job(name))
        return jobs

    def delete_job(self, job_name: str) -> None:
        try:
            # Stop and remove containers
            containers = self.podman_client.containers.list(
                all=True,
                filters={"label": f"{constants.CONTAINER_TRAIN_JOB_NAME_LABEL}={job_name}"},
            )
            for c in containers:
                try:
                    c.stop()
                    c.remove(force=True)
                    print(f"Removed container: {c.name}")
                except Exception as e:
                    print(f"Error removing container {c.name}: {str(e)}")

            # Stop and remove pod
            try:
                pod = self.podman_client.pods.get(job_name)
                pod.stop()
                pod.remove(force=True)
                print(f"Removed pod: {pod.name}")
            except podman.errors.exceptions.NotFound:
                print(f"Pod {job_name} not found, skipping pod removal")
            except Exception as e:
                print(f"Error removing pod {job_name}: {str(e)}")

        except Exception as e:
            print(f"Error during job deletion: {str(e)}")

    def __list_job_names(
        self,
        runtime_name: Optional[str] = None,
    ) -> List[str]:
        filters = {"label": [constants.CONTAINER_TRAIN_JOB_NAME_LABEL]}
        if runtime_name is not None:
            filters["label"].append(
                f"{constants.CONTAINER_RUNTIME_LABEL}={runtime_name}"
            )

        networks = self.podman_client.networks.list(filters=filters)

        job_names = []
        for n in networks:
            job_names.append(n.name)
        return job_names

    @staticmethod
    def __get_container_environment(
        framework: types.Framework,
        head_node_address: str,
        num_nodes: int,
        node_rank: int,
    ) -> Dict[str, str]:
        env = {
            "PET_NNODES": str(num_nodes),
            "PET_NPROC_PER_NODE": "1",
            "PET_NODE_RANK": str(node_rank),
            "PET_MASTER_ADDR": head_node_address,
            "PET_MASTER_PORT": str(constants.TORCH_HEAD_NODE_PORT),
        }
        return env

    @staticmethod
    def __get_job_status(containers: List[types.Container]) -> str:
        """Get the status of a training job.

        Args:
            containers: List of containers in the job.

        Returns:
            str: The status of the job.
        """
        # TODO: Implement proper status reporting
        return "Running"

