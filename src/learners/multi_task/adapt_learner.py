import copy
import random
from sre_compile import dis

from numpy import log
from components.episode_buffer import EpisodeBatch
from modules.mixers.multi_task.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.multi_task.qattn import QMixer as MTAttnQMixer
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
import math

import os


class ADAPTLearner:
    def __init__(self, mac, logger, main_args):
        self.main_args = main_args
        self.mac = mac
        self.logger = logger

        # get some attributes from mac
        self.task2args = mac.task2args
        self.task2n_agents = mac.task2n_agents
        self.surrogate_decomposer = mac.surrogate_decomposer
        self.task2decomposer = mac.task2decomposer

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if main_args.mixer is not None:
            if main_args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif main_args.mixer == "mt_qattn":
                self.mixer = MTAttnQMixer(self.surrogate_decomposer, main_args)
            else:
                raise ValueError(f"Mixer {main_args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self._reset_optimizer()

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # define attributes for each specific task
        self.task2train_info, self.task2encoder_params, self.task2encoder_optimiser = (
            {},
            {},
            {},
        )
        for task in self.task2args:
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = (
                -task_args.learner_log_interval - 1
            )

        self.c = main_args.c_step
        self.skill_dim = main_args.skill_dim
        self.beta = main_args.beta
        self.alpha = main_args.coef_conservative
        self.phi = main_args.coef_dist
        self.kl_weight = main_args.coef_kl
        self.entity_embed_dim = main_args.entity_embed_dim
        self.device = None
        self.ssl_type = main_args.ssl_type
        self.ssl_tw = main_args.ssl_time_window
        self.double_neg = main_args.double_neg
        self.td_weight = main_args.td_weight
        # self.adaptation = main_args.adaptation

        self.pretrain_steps = 0
        self.training_steps = 0
        self.reset_last_batch()

    def _reset_optimizer(self):
        if self.main_args.optim_type.lower() == "rmsprop":
            self.pre_optimiser = RMSprop(
                params=self.params,
                lr=self.main_args.lr,
                alpha=self.main_args.optim_alpha,
                eps=self.main_args.optim_eps,
                weight_decay=self.main_args.weight_decay,
            )
            self.optimiser = RMSprop(
                params=self.params,
                lr=self.main_args.lr,
                alpha=self.main_args.optim_alpha,
                eps=self.main_args.optim_eps,
                weight_decay=self.main_args.weight_decay,
            )
        elif self.main_args.optim_type.lower() == "adam":
            self.pre_optimiser = Adam(
                params=self.params,
                lr=self.main_args.critic_lr,
                weight_decay=self.main_args.weight_decay,
            )
            self.optimiser = Adam(
                params=self.params,
                lr=self.main_args.critic_lr,
                weight_decay=self.main_args.weight_decay,
            )
        else:
            raise ValueError("Invalid optimiser type", self.main_args.optim_type)
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def zero_grad(self):
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def update(self, pretrain=True):
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.main_args.grad_norm_clip
        )
        if pretrain:
            self.pre_optimiser.step()
            self.pre_optimiser.zero_grad()
        else:
            self.optimiser.step()
            self.optimiser.zero_grad()

    def l2_loss(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    def contrastive_loss(self, obs, obs_pos, obs_neg):
        # obs & obs_pos: 1, dim; obs_neg: bs, dim
        obs_ = th.cat([obs, obs_neg.detach()], dim=0)
        obs_pos_ = th.cat([obs_pos, obs_neg], dim=0).detach()
        logits = self.mac.forward_contrastive(obs_, obs_pos_)
        labels = th.zeros(logits.shape[0]).long().to(self.device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def reset_last_batch(self):
        self.last_task = ""
        # self.last_batch = th.tensor(0.)
        self.last_batch = {}

    def update_last_batch(self, cur_task, cur_batch):
        if not self.double_neg:
            if cur_task != self.last_task:
                self.reset_last_batch()
                self.last_batch[cur_task] = cur_batch
        else:
            self.last_batch[cur_task] = cur_batch
        self.last_task = cur_task
        # if cur_task != self.last_task:
        #     self.last_batch = cur_batch
        #     self.last_task = cur_task

    def compute_neg_sample(self, batch, task):
        target_outs = []
        agent_random = random.randint(0, self.task2n_agents[task] - 1)
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length - self.c):
            with th.no_grad():
                target_mac_out, _ = self.target_mac.forward_discriminator(
                    batch, t=t, task=task
                )
                target_mac_out = target_mac_out[:, agent_random]
            target_outs.append(target_mac_out)
        target_outs = th.cat(target_outs, dim=1).reshape(
            -1, self.main_args.entity_embed_dim
        )

        return target_outs

    def train_vae(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        task: str,
        ssl_loss=None,
    ):
        # print("training vae!!")
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        dec_loss = 0.0
        b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        self.mac.init_hidden(batch.batch_size, task)
        # self.target_mac.init_hidden(batch.batch_size, task)
        t = 0
        while t < batch.max_seq_length - self.c:
            # for t in range(batch.max_seq_length - self.c):
            act_outs = []
            agent_outs, _ = self.mac.forward_planner(
                batch, t=t, task=task, actions=actions[:, t], hrl=True
            )
            act_agent_outs = self.mac.forward_planner_feedforward(agent_outs)
            for i in range(self.c):
                act_out, _ = self.mac.forward_global_action(
                    batch, act_agent_outs, t + i, task
                )
                act_outs.append(act_out)
            act_outs = th.stack(act_outs, dim=1)
            _, _, n, a = act_out.shape
            dec_loss += (
                F.cross_entropy(
                    act_outs.reshape(-1, a),
                    # actions[:, t: t + self.c].squeeze(-1).reshape(-1),
                    # reduction="sum") / mask[:, t:t + self.c].sum()
                    actions[:, t : t + self.c].squeeze(-1).reshape(-1),
                    reduction="sum",
                )
                / mask[:, t : t + self.c].sum()
            ) / n
            t += self.c

        # vae_loss = dec_loss / (batch.max_seq_length - self.c) + self.main_args.beta * enc_loss
        vae_loss = dec_loss / (batch.max_seq_length - self.c)
        # cls_loss = cls_loss / (batch.max_seq_length - self.c)
        loss = vae_loss
        if ssl_loss is not None:
            loss += self.beta * ssl_loss
        # loss += self.main_args.coef_sim * (sim_loss)

        # self.mac.agent.value.requires_grad_(False)
        # self.mac.agent.planner.requires_grad_(False)
        # self.optimiser.zero_grad()
        loss.backward()
        # self.optimiser.step()
        # return vae_loss

        # if (t_env - self.last_target_update_episode) / self.main_args.target_update_interval >= 1.0:
        # 	self._update_targets()
        # 	self.last_target_update_episode = t_env

        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            # self.logger.log_stat(f"pretrain/{task}/grad_norm", grad_norm.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/dist_loss", dist_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/enc_loss", enc_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/dec_loss", dec_loss.item(), t_env)
            self.logger.log_stat(f"train/{task}/vae_loss", vae_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/cls_loss", cls_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/rew_loss", v_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/ensemble_loss", ensemble_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/ent_loss", ent_loss.item(), t_env)

            # for i in range(self.skill_dim):
            #     skill_dist = seq_skill_input.reshape(-1, self.skill_dim).mean(dim=0)
            #     self.logger.log_stat(f"pretrain/{task}/skill_class{i+1}", skill_dist[i].item(), t_env)

            self.task2train_info[task]["log_stats_t"] = t_env

    def test_vae(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # # Calculate estimated Q-Values
        # mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        # for t in range(batch.max_seq_length):
        #     agent_outs = self.mac.forward_skill(batch, t=t, task=task, actions=actions[:, t, :])
        #     mac_out.append(agent_outs)
        # mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # ######## beta-vae loss
        # # prior loss
        # seq_skill_input = F.gumbel_softmax(mac_out[:, :-self.c, :, :], dim=-1)
        # kl_seq_skill = seq_skill_input * (th.log(seq_skill_input) - math.log(1 / self.main_args.skill_dim))
        # enc_loss = kl_seq_skill.mean()

        dec_loss = 0.0  ### batch time agent skill
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length - self.c):
            # seq_action_output = self.mac.forward_seq_action(batch, seq_skill_input[:, t, :, :], t, task=task)
            seq_action_output = self.mac.forward_seq_action(batch, t, task=task)
            b, c, n, a = seq_action_output.size()
            dec_loss += (
                F.cross_entropy(
                    seq_action_output.reshape(-1, a),
                    actions[:, t : t + self.c].squeeze(-1).reshape(-1),
                    reduction="sum",
                )
                / mask[:, t : t + self.c].sum()
            ) / n

        # vae_loss = dec_loss / (batch.max_seq_length - self.c) + self.main_args.beta * enc_loss
        vae_loss = dec_loss / (batch.max_seq_length - self.c)
        loss = vae_loss

        # self.logger.log_stat(f"pretrain/{task}/test_vae_loss", loss.item(), t_env)
        # self.logger.log_stat(f"pretrain/{task}/test_enc_loss", enc_loss.item(), t_env)
        # self.logger.log_stat(f"pretrain/{task}/test_dec_loss", dec_loss.item(), t_env)
        self.logger.log_stat(f"train/{task}/test_vae_loss", loss.item(), t_env)

        # for i in range(self.skill_dim):
        #     skill_dist = seq_skill_input.reshape(-1, self.skill_dim).mean(dim=0)
        #     self.logger.log_stat(f"pretrain/{task}/test_skill_class{i+1}", skill_dist[i].item(), t_env)

    def train_value(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]

        # # if len(self.last_task) != 0:
        # if self.last_task != '':
        #     ssl_loss = 0.
        #     target_outs = []
        #     self.mac.init_hidden(batch.batch_size, task)
        #     self.target_mac.init_hidden(batch.batch_size, task)
        #
        #     cur_random, last_random = random.randint(0, self.task2n_agents[task]-1), \
        #         random.randint(0, self.task2n_agents[self.last_task]-1)
        #     pos_random = random.randint(0, self.task2n_agents[task]-1)
        #     while pos_random == cur_random:
        #         pos_random = random.randint(0, self.task2n_agents[task]-1)
        #     cur_t = random.randint(0, batch.max_seq_length-self.c-1)
        #
        #     mac_out, _ = self.mac.forward_discriminator(batch, t=cur_t, task=task)
        #     cur_out, pos_out = mac_out[:, cur_random], mac_out[:, pos_random]
        #     for t in range(self.last_batch.max_seq_length - self.c):
        #         with th.no_grad():
        #             target_mac_out, _ = self.target_mac.forward_discriminator(
        #             self.last_batch, t=t, task=self.last_task)
        #             target_mac_out = target_mac_out[:, last_random]
        #         target_outs.append(target_mac_out)
        #
        #     target_outs = th.cat(target_outs, dim=1).reshape(-1, self.main_args.entity_embed_dim)
        #
        #     for _ in range(cur_out.shape[0]):
        #         ssl_loss += self.contrastive_loss(cur_out, pos_out.detach(), target_outs.detach())
        #     ssl_loss = ssl_loss / cur_out.shape[0]
        # else:
        #     ssl_loss = th.tensor(0.)

        #### value net inference
        values = []
        target_values = []
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            # out_h, _ = self.mac.forward_planner(batch, t=t, task=task, actions=actions[:, t])
            # value_out_h = self.mac.forward_planner_feedforward(out_h, forward_type='value')
            # value = self.mac.forward_value_skill(batch, value_out_h, task=task)
            value = self.mac.forward_value(batch, t=t, task=task)
            values.append(value)
            with th.no_grad():
                # target_out_h, _ = self.target_mac.forward_planner(batch, t=t, task=task, actions=actions[:, t])
                # target_value = self.target_mac.forward_value_skill(batch, target_out_h, task=task)
                target_value = self.target_mac.forward_value(batch, t=t, task=task)
                target_values.append(target_value)

        # bs, t_len, n_agents, 1
        values = th.stack(values, dim=1)
        target_values = th.stack(target_values, dim=1)
        rewards = rewards.reshape(-1, batch.max_seq_length, 1)

        if self.mixer is not None:
            mixed_values = self.mixer(
                values, batch["state"][:, :], self.task2decomposer[task]
            )
            with th.no_grad():
                target_mixed_values = self.target_mixer(
                    target_values, batch["state"][:, :], self.task2decomposer[task]
                ).detach()
        else:
            mixed_values = values.sum(dim=2)
            target_mixed_values = target_values.sum(dim=2).detach()

        cs_rewards = batch["reward"]
        discount = self.main_args.gamma
        for i in range(1, self.c):
            cs_rewards[:, : -self.c] += discount * rewards[:, i : -(self.c - i)]
            discount *= self.main_args.gamma

        td_error = (
            mixed_values[:, : -self.c]
            - cs_rewards[:, : -self.c]
            - discount
            * (1 - terminated[:, self.c - 1 : -1])
            * target_mixed_values[:, self.c :].detach()
        )
        mask = mask.expand_as(mixed_values)
        masked_td_error = td_error * mask[:, : -self.c]

        if self.main_args.adaptation:
            value_loss = th.mean((masked_td_error**2).sum()) / mask[:, : -self.c].sum()
        else:
            value_loss = (
                th.mean(
                    th.abs(0.9 - (masked_td_error < 0).float()).mean()
                    * (masked_td_error**2).sum()
                )
                / mask[:, : -self.c].sum()
            )

        # loss = ssl_loss + value_loss
        loss = value_loss

        self.mac.agent.value.requires_grad_(True)
        loss.backward()

        # return value_loss, ssl_loss
        return value_loss
        ####

    def train_planner(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        task: str,
        v_loss=None,
        dec_loss=None,
        cls_loss=None,
        ssl_loss=None,
    ):
        # Get the relevant quantities
        # print("training planner!!")
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]
        # task2label = {
        # 	3: 0,
        # 	5: 1,
        # 	9: 2,
        # }

        #### planner inference
        # values = []
        # values_next = []
        mac_out_h, target_mac_out_h = [], []
        # mac_out_own = []
        # mac_out_enemy = []
        # mac_out_ally = []
        # mac_out_own_h = []
        # mac_out_enemy_h = []
        # mac_out_ally_h = []
        # target_mac_value = []
        # dec_loss = 0.
        planner_loss = 0.0
        b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        # task_label = th.tensor(task2label[n]).reshape(1, 1, 1, 1).repeat(b, t, n, 1).to(actions.device)
        # cls_loss = 0.

        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length - self.c):
            out_h, obs_loss = self.mac.forward_planner(
                batch,
                t=t,
                task=task,
                actions=actions[:, t],
                training=True,
                loss_out=True,
            )
            planner_loss += obs_loss
            # mac_out_h.append(th.cat(out_h, dim=-2))
            # with th.no_grad():
            #     target_out_h, _ = self.target_mac.forward_planner(
            #         batch,
            #         t=t,
            #         task=task,
            #         actions=actions[:, t],
            #         training=True,
            #         loss_out=False,
            #     )
            #     target_mac_out_h.append(th.cat(target_out_h, dim=-2))

        # t = batch.max_seq_length - self.c
        # for i in range(self.c):
        #     out_h, obs_loss = self.mac.forward_planner(
        #         batch,
        #         t=t + i,
        #         task=task,
        #         actions=actions[:, t + i],
        #         training=True,
        #         loss_out=True,
        #     )
        #     planner_loss += obs_loss
        #     mac_out_h.append(th.cat(out_h, dim=-2))
        #     with th.no_grad():
        #         target_out_h, _ = self.target_mac.forward_planner(
        #             batch,
        #             t=t + i,
        #             task=task,
        #             actions=actions[:, t + i],
        #             training=True,
        #             loss_out=False,
        #         )
        #         target_mac_out_h.append(th.cat(target_out_h, dim=-2))
        #
        # mac_out_h = th.stack(mac_out_h, dim=1)
        # target_mac_out_h = th.stack(target_mac_out_h, dim=1)

        planner_loss = planner_loss / (batch.max_seq_length - self.c)
        loss = planner_loss

        # Do RL Learning
        # self.mac.agent.state_encoder.requires_grad_(False)
        # self.mac.agent.decoder.requires_grad_(False)
        # self.mac.agent.value.requires_grad_(False)
        # self.optimiser.zero_grad()
        loss.backward()
        # self.mac.agent.value.requires_grad_(True)

        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        # self.optimiser.step()

        # episode_num should be pulic
        # if (
        #     t_env - self.last_target_update_episode
        # ) / self.main_args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_episode = t_env

        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            if dec_loss is not None:
                self.logger.log_stat(f"{task}/dec_loss", dec_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/td_error", td_error.item(), t_env)
            if v_loss is not None:
                self.logger.log_stat(f"{task}/value_loss", v_loss.item(), t_env)
            self.logger.log_stat(f"{task}/plan_loss", planner_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/class_loss", cls_loss.item(), t_env)
            if ssl_loss is not None:
                self.logger.log_stat(f"{task}/ssl_loss", ssl_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/grad_norm", grad_norm.item(), t_env)
            # mask_elems = mask.sum().item()
            # self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            # self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item() / (
            #     mask_elems * self.task2args[task].n_agents), t_env)
            # self.logger.log_stat(f"{task}/target_mean",
            #                      (targets * mask[:, :-self.c]).sum().item() / (mask_elems * self.task2args[task].n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def train_ssl(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        ssl_loss = 0.0
        # mac_outs, mac_outs_h = [], []
        target_outs, target_outs_h = [], []
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)

        cur_random, last_random = random.randint(
            0, self.task2n_agents[task] - 1
        ), random.randint(0, self.task2n_agents[self.last_task] - 1)
        pos_random = random.randint(0, self.task2n_agents[task] - 1)
        while pos_random == cur_random:
            pos_random = random.randint(0, self.task2n_agents[task] - 1)
        cur_t = random.randint(0, batch.max_seq_length - self.c - 1)

        mac_out, _ = self.mac.forward_discriminator(batch, t=cur_t, task=task)
        cur_out, pos_out = mac_out[:, cur_random], mac_out[:, pos_random]
        for t in range(self.last_batch.max_seq_length - self.c):
            with th.no_grad():
                target_mac_out, _ = self.target_mac.forward_discriminator(
                    self.last_batch, t=t, task=self.last_task
                )
                target_mac_out = target_mac_out[:, last_random]
            target_outs.append(target_mac_out)

        # for t in range(batch.max_seq_length - self.c):
        #     mac_out, mac_out_h = self.mac.forward_discriminator(batch, t=t, task=task)
        #     n = mac_out.shape[1]
        # for t in range(self.last_batch.max_seq_length - self.c):
        #     with th.no_grad():
        #         target_out, target_out_h = self.target_mac.forward_discriminator(
        # self.last_batch, t=t, task=task)

        #         mac_out, mac_out_h, target_out, target_out_h = mac_out.reshape(-1, self.main_args.entity_embed_dim), \
        #    mac_out_h.reshape(-1, self.main_args.entity_embed_dim), target_out.reshape(-1, self.main_args.entity_embed_dim), \
        #        target_out_h.reshape(-1, self.main_args.entity_embed_dim)

        # mac_outs.append(mac_out)
        # mac_outs_h.append(mac_out_h)
        # target_outs.append(target_out)
        # target_outs_h.append(target_out_h)

        target_outs = th.cat(target_outs, dim=1).reshape(
            -1, self.main_args.entity_embed_dim
        )
        # target_outs_h = th.stack(target_outs_h, dim=1)

        for _ in range(cur_out.shape[0]):
            ssl_loss += self.contrastive_loss(
                cur_out, pos_out.detach(), target_outs.detach()
            )
        ssl_loss = ssl_loss / cur_out.shape[0]

        #     ssl_loss += self.contrastive_loss(mac_outs, target_outs_h.detach()).sum() + \
        #    self.contrastive_loss(mac_outs_h, target_outs.detach()).sum()
        #     ssl_loss = ssl_loss / (mask.sum() * n)
        ssl_loss.backward()

        return ssl_loss

    def pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.pretrain_steps == 0:
            self._reset_optimizer()
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = (
                    -task_args.learner_log_interval - 1
                )

        # self.train_vae(batch, t_env, episode_num, task)
        self.train_planner(
            batch,
            t_env,
            episode_num,
            task,
            # dec_loss=dec_loss,
        )
        # v_loss = self.train_value(batch, t_env, episode_num, task)
        # self.train_planner(batch, t_env, episode_num, task, v_loss)
        self.pretrain_steps += 1

    def test_pretrain(
        self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str
    ):
        self.test_vae(batch, t_env, episode_num, task)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.training_steps == 0:
            self._reset_optimizer()
            self.device = batch.device
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = (
                    -task_args.learner_log_interval - 1
                )

        # self.train_policy(batch, t_env, episode_num, task)
        # if self.last_task != '':
        #     ssl_loss = self.train_ssl(batch, t_env, episode_num, task)
        #     self.update(pretrain=False)
        # else:
        #     ssl_loss = th.tensor(0.)
        if self.main_args.adaptation:
            v_loss = self.train_value(batch, t_env, episode_num, task)
            self.train_planner(
                batch,
                t_env,
                episode_num,
                task,
                v_loss=th.tensor(0.0),
                dec_loss=th.tensor(0.0),
                ssl_loss=th.tensor(0.0),
            )
        else:
            self.train_vae(batch, t_env, episode_num, task)
            # dec_loss = self.train_vae(batch, t_env, episode_num, task)
            # self.update(pretrain=False)

        self.training_steps += 1

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load(
                    "{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
