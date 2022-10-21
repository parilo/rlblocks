from typing import Dict, Union

import torch as t
import torch.optim as optim
import torch.nn.functional as F

from rlblocks.data.replay_buffer import Batch
from rlblocks.model.actor import Actor
from rlblocks.model.q_func import QFunc
from rlblocks.model.stochastic_actor import StochasticActor


class AWACOptimizer:

    def __init__(
            self,
            actor: StochasticActor,
            q_func: QFunc,
            alambda: float,
            lr: float
    ):
        self._actor = actor
        self._actor_opt = optim.Adam(self._actor.model.parameters(), lr=lr)
        self._q_func = q_func
        self._alambda = alambda

    def train_step(self, batch: Batch) -> Dict[str, float]:
        # print(f'--- train_step_actor: state {batch.state.shape}')
        # print(f'--- train_step_actor: batch action {batch.action.shape}')
        # print(f'--- train_step_actor: actor action {action.shape}')

        with t.no_grad():
            action_q_batch = self._q_func(batch.state, batch.action)
            action_policy, info = self._actor(batch.state, add_info=True)
            action_q_actor = self._q_func(batch.state, action_policy)
            adv = action_q_batch - action_q_actor
            adv_clipped = adv.clamp(min=0)
            # adv = adv.clamp(max=5)
            # score = t.exp(adv / self._alambda)
            score = adv / self._alambda
            score_clipped = t.where(adv > 0, score, 0.)

        # print(f'--- train_step_actor: action_q_batch {action_q_batch.shape}')
        # print(f'--- train_step_actor: action_q_actor {action_q_actor.shape}')
        # print(f'--- train_step_actor: adv {adv.shape}')
        # print(f'--- train_step_actor: score {score.shape}')

        log_prob = self._actor.log_prob(batch.state, batch.action)
        # print(f'--- train_step_actor: log_prob {log_prob.shape}')
        log_prob_scored = log_prob * score_clipped
        # print(f'--- train_step_actor: scored log_prob_scored {log_prob_scored.shape}')
        actor_loss = -log_prob_scored.mean()
        # print(f'--- train_step_actor: actor_loss {actor_loss.shape}')

        self._actor_opt.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()

        train_info = {
            'actor_loss': actor_loss.item(),
            'actor_adv': adv.mean().item(),
            'actor_adv_clipped': adv_clipped.mean().item(),
            'actor_score': score.mean().item(),
            'actor_score_clipped': score_clipped.mean().item(),
            'actor_log_prob': log_prob.mean().item(),
            # 'action_mu': info['mu'].mean().item(),
            # 'action_std': info['std'].mean().item(),
        }
        for name, val in info.items():
            train_info[f'action_{name}'] = val.mean().item()

        return train_info
