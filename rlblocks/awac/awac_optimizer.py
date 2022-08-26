from typing import Dict, Union

import torch as t
import torch.optim as optim
import torch.nn.functional as F

from lm_rl.data.replay_buffer import Batch
from lm_rl.model.actor import Actor
from lm_rl.model.q_func import QFunc


class AWACOptimizer:

    def __init__(
            self,
            actor: Actor,
            q_func: QFunc,
            lr: float
    ):
        self._actor = actor
        self._actor_opt = optim.Adam(self._actor.model.parameters(), lr=lr)
        self._q_func = q_func

    def train_step_actor(self, batch: Batch):
        action = self._actor(batch.state)

        # print(f'--- train_step_actor: state {batch.state.shape}')
        # print(f'--- train_step_actor: batch action {batch.action.shape}')
        # print(f'--- train_step_actor: actor action {action.shape}')

        with t.no_grad():
            action_q_batch = self._q_func(batch.state, batch.action)
            action_q_actor = self._q_func(batch.state, action)
            adv = action_q_batch - action_q_actor
            score = t.where(adv > 0, 1., 0.)

        # print(f'--- train_step_actor: action_q_batch {action_q_batch.shape}')
        # print(f'--- train_step_actor: action_q_actor {action_q_actor.shape}')
        # print(f'--- train_step_actor: adv {adv.shape}')
        # print(f'--- train_step_actor: score {score.shape}')

        action_diff = F.mse_loss(action, batch.action, reduction='none')
        # print(f'--- train_step_actor: action_diff {action_diff.shape}')
        scored_action_diff = action_diff * score
        # print(f'--- train_step_actor: scored_action_diff {scored_action_diff.shape}')
        actor_loss = scored_action_diff.mean()
        # print(f'--- train_step_actor: actor_loss {actor_loss.shape}')

        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        return {
            'action_q_batch': action_q_batch.mean().item(),
            'action_q_actor': action_q_actor.mean().item(),
            'adv': adv.mean().item(),
            'adv clipped': adv.clamp(min=0).mean().item(),
            'score': score.mean().item(),
            'action_diff': action_diff.mean().item(),
            'scored_action_diff': scored_action_diff.mean().item(),
            'actor_loss': actor_loss.item()
        }

    def train_step(self, actor_batch: Batch) -> Dict[str, float]:
        train_info = {}
        train_info.update(self.train_step_actor(actor_batch))
        return train_info
