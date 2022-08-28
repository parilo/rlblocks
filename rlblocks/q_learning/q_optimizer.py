from typing import Dict, Union

import torch as t
import torch.optim as optim
import torch.nn.functional as F

from rlblocks.data.replay_buffer import Batch
from rlblocks.model.actor import Actor
from rlblocks.model.q_func import QFunc


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model (PyTorch model): weights will be copied from
        target_model (PyTorch model): weights will be copied to
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class QOptimizer:

    def __init__(
            self,
            q_func: QFunc,
            q_target_func: QFunc,
            actor: Actor,
            lr: float,
            gamma: float,
            update_target_each: int,
            update_target_tau: float,
    ):
        self._q_func = q_func
        self._q_target_func = q_target_func
        self._actor = actor
        self._gamma = gamma
        self._update_target_each = update_target_each
        self._update_target_tau = update_target_tau

        self._q_opt = optim.Adam(self._q_func.model.parameters(), lr=lr)
        self._step_ind = 0

    def _update_target(self):
        if self._step_ind % self._update_target_each == 0:
            soft_update(
                self._q_func.model,
                self._q_target_func.model,
                self._update_target_tau
            )

    def train_step(self, batch: Batch) -> Dict[str, float]:
        q_value = self._q_func(batch.state, batch.action)
        with t.no_grad():
            next_action = self._actor(batch.next_state)
            q_value_target = self._q_target_func(batch.next_state, next_action)
            q_target = batch.reward + self._gamma * q_value_target

        # print(f'--- train_step q: state {batch.state.shape}')
        # print(f'--- train_step q: next state {batch.next_state.shape}')
        # print(f'--- train_step q: batch action {batch.action.shape}')
        # print(f'--- train_step q: actor action {action.shape}')
        # print(f'--- train_step q: q value  {q_value.shape}')
        # print(f'--- train_step q: q target {q_target.shape}')

        q_loss = F.mse_loss(q_value, q_target)

        self._q_opt.zero_grad()
        q_loss.backward()
        self._q_opt.step()

        self._update_target()
        self._step_ind += 1

        return {
            'q_value': q_value.mean().item(),
            'q_target': q_value_target.mean().item(),
            'q_loss': q_loss.item()
        }
