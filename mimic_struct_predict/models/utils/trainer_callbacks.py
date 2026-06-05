from typing import Any, Optional

import GPUtil
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.integrations import TensorBoardCallback


class MultiGpuTensorBoardCallback(TensorBoardCallback):
    def __init__(self, tb_writer: Optional[SummaryWriter] = None) -> None:
        super().__init__(tb_writer)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # Call parent method for standard metrics
        super().on_step_end(args, state, control, **kwargs)

        # CRITICAL: Only fetch and log on the main process
        if state.is_world_process_zero and self.tb_writer:
            gpus = GPUtil.getGPUs()
            if gpus:
                total_load: float = 0.0
                total_mem: float = 0.0

                # Log each GPU individually
                for i, gpu in enumerate(gpus):
                    self.tb_writer.add_scalar(f"Hardware/GPU_{i}_Compute", gpu.load, state.global_step)
                    self.tb_writer.add_scalar(f"Hardware/GPU_{i}_Memory", gpu.memoryUtil, state.global_step)

                    total_load += gpu.load
                    total_mem += gpu.memoryUtil

                # Log the cluster average
                self.tb_writer.add_scalar("Hardware/Cluster_Avg_Compute", total_load / len(gpus), state.global_step)
                self.tb_writer.add_scalar("Hardware/Cluster_Avg_Memory", total_mem / len(gpus), state.global_step)
