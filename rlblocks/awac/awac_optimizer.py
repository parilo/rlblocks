from typing import Dict, Union

import torch as t
import torch.optim as optim
import torch.nn.functional as F

from rlblocks.data.replay_buffer import Batch
from rlblocks.model.actor import Actor
from rlblocks.model.q_func import QFunc


class AWACOptimizer:

    def __init__(
            self,
            actor: Actor,
            q_func: QFunc,
            alambda: float,
            lr: float
    ):
        self._actor = actor
        self._actor_opt = optim.Adam(self._actor.model.parameters(), lr=lr)
        self._q_func = q_func
        self._alambda = alambda

    def train_step_actor(self, batch: Batch):
        action = self._actor(batch.state)

        # add implementation

        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        return {
            'actor_loss': actor_loss.item()
        }

    def train_step(self, actor_batch: Batch) -> Dict[str, float]:
        train_info = {}
        train_info.update(self.train_step_actor(actor_batch))
        return train_info
