import copy
from email import policy
import random
from sre_compile import dis
from collections import deque

import numpy as np
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


class ADAPTMOCOLearner:
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

    def contrastive_loss(self, obs, obs_pos):
        logits = self.mac.forward_contrastive(obs, obs_pos).to(self.device)
        labels = (
            th.zeros(logits.shape[1])
            .unsqueeze(0)
            .repeat(logits.shape[0], 1)
            .long()
            .to(logits.device)
        )
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
        value_loss=None,
    ):
        # print("training vae!!")
        # rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]

        task2decomposer = self.task2decomposer[task]
        n_agents = task2decomposer.n_agents
        n_enenmies = task2decomposer.n_enemies
        n_entities = n_agents + n_enenmies
        n_actions_no_attack = task2decomposer.n_actions_no_attack
        act_dim = n_actions_no_attack + n_enenmies

        policy_loss = 0.0
        reg_loss = 0.0
        # b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)
        t = 0
        act_outs = []
        merge_outs = []
        while t < batch.max_seq_length - 1:
            # with th.no_grad():
            # for t in range(batch.max_seq_length - self.c):
            out_h = self.mac.forward_global_hidden(
                batch, t=t, task=task, actions=actions[:, t]
            )
            out_h = out_h.repeat(1, n_agents, 1)
            # out_h = self.mac.forward_encoder(batch, t=t, task=task)
            # out_h = self.mac.forward_latent_planner(
            #     out_h,
            #     rewards=None,
            #     actions=None,
            #     t=t,
            #     # end_t=t,
            #     task=task,
            #     test_mode=False,
            # )
            # merge_h, _ = self.mac.forward_reconstruction(
            #     batch, out_h, t=t, task=task, actions=actions[:, t]
            # )
            # merge_outs.append(merge_h)

            act_out = self.mac.forward_global_action(
                batch,
                out_h,
                # th.zeros(
                #     batch.batch_size, 1, n_agents, n_entities, self.entity_embed_dim
                # ).to(actions.device),
                t=t,
                task=task,
            )
            act_outs.append(act_out)
            t += 1

        act_outs = th.stack(act_outs, dim=1)
        policy_loss = (
            F.cross_entropy(
                act_outs.reshape(-1, act_dim),
                actions[:, :-1].reshape(-1),
                reduction="sum",
            )
            / mask[:, :-1].sum()
            / self.task2n_agents[task]
        )
        reg_loss += (
            0.5 * out_h.reshape(-1, self.entity_embed_dim).pow(2).sum(1)
        ).mean()

        # reg_loss += (
        #     0.5 * act_agent_outs.reshape(-1, self.entity_embed_dim).pow(2).sum(1)
        # ).mean()
        # ally_out, enemy_out = self.mac.forward_global_decoder(
        #     act_agent_outs, t, task
        # )
        # al_dim, en_dim = ally_out.shape[-1], enemy_out.shape[-1]
        # ally_target, enemy_target = self.mac.global_state_process(
        #     batch, t=t, task=task, actions=actions[:, t]
        # )

        # dec_loss += (
        #     F.mse_loss(
        #         ally_out.reshape(-1, al_dim),
        #         ally_target.reshape(-1, al_dim),
        #         reduction="sum",
        #     )
        #     / mask[:, t : t + 1].sum()
        # )

        # vae_loss = dec_loss / (batch.max_seq_length - self.c) + self.main_args.beta * enc_loss
        policy_loss = policy_loss / (batch.max_seq_length - 1)
        loss = policy_loss + 1e-6 * reg_loss
        # loss = vae_loss + 1e-6 * reg_loss

        # self.mac.agent.value.requires_grad_(False)
        # self.mac.agent.encoder.requires_grad_(False)
        # self.mac.agent.planner.requires_grad_(False)
        # self.mac.agent.merge.requires_grad_(False)
        # self.optimiser.zero_grad()
        loss.backward()
        # self.optimiser.step()
        # return vae_loss
        if (
            t_env - self.last_target_update_episode
        ) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = t_env

        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            # self.logger.log_stat(f"pretrain/{task}/grad_norm", grad_norm.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/dist_loss", dist_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/enc_loss", enc_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/dec_loss", dec_loss.item(), t_env)
            self.logger.log_stat(f"train/{task}/vae_loss", policy_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/reg_loss", reg_loss.item(), t_env)
            if value_loss is not None:
                self.logger.log_stat(
                    f"train/{task}/value_loss", value_loss.item(), t_env
                )
            # self.logger.log_stat(f"train/{task}/ensemble_loss", ensemble_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/ent_loss", ent_loss.item(), t_env)

            # for i in range(self.skill_dim):
            #     skill_dist = seq_skill_input.reshape(-1, self.skill_dim).mean(dim=0)
            #     self.logger.log_stat(f"pretrain/{task}/skill_class{i+1}", skill_dist[i].item(), t_env)

            self.task2train_info[task]["log_stats_t"] = t_env
        # return policy_loss

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

        vae_loss = dec_loss / (batch.max_seq_length - self.c)
        loss = vae_loss

        # self.logger.log_stat(f"pretrain/{task}/test_vae_loss", loss.item(), t_env)
        # self.logger.log_stat(f"pretrain/{task}/test_enc_loss", enc_loss.item(), t_env)
        # self.logger.log_stat(f"pretrain/{task}/test_dec_loss", dec_loss.item(), t_env)
        self.logger.log_stat(f"train/{task}/test_vae_loss", loss.item(), t_env)

    def train_value(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]

        task2decomposer = self.task2decomposer[task]
        n_agents = task2decomposer.n_agents
        n_enemies = task2decomposer.n_enemies
        n_entities = n_agents + n_enemies

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
            # emb_out = self.mac.forward_encoder(
            #     batch, t=t, task=task, actions=actions[:, t]
            # )
            # skill_out = self.mac.forward_global_hidden(
            #     batch, t=t, task=task, actions=actions[:, t]
            # )
            # emb_out = self.mac.forward_encoder_mlp(emb_out, t=t, task=task).reshape(
            #     -1, n_entities, self.entity_embed_dim
            # )
            value, _ = self.mac.forward_value(batch, t, task)
            values.append(value)
            with th.no_grad():
                target_value, _ = self.target_mac.forward_value(batch, t, task)
                target_values.append(target_value)

            # value = self.mac.forward_value(batch, t=t, task=task)
            # values.append(value)
            # with th.no_grad():
            #     # target_out_h, _ = self.target_mac.forward_planner(batch, t=t, task=task, actions=actions[:, t])
            #     # target_value = self.target_mac.forward_value_skill(batch, target_out_h, task=task)
            #     target_value = self.target_mac.forward_value(batch, t=t, task=task)
            #     target_values.append(target_value)

        # bs, t_len, n_agents, 1
        values = th.stack(values, dim=1)
        # values = values.gather(
        #     -1, actions.reshape(bs, batch.max_seq_length, n_agents, 1)
        # )
        target_values = th.stack(target_values, dim=1)
        # target_values = target_values.max(-1)[0]
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
        assert self.c == 1
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

        self.mac.agent.value.requires_grad_(True)
        loss = value_loss
        loss.backward()

        # if (
        #     t_env - self.last_target_update_episode
        # ) / self.main_args.target_update_interval >= 1.0:
        #     self._update_targets()
        #     self.last_target_update_episode = t_env

        # if (
        #     t_env - self.task2train_info[task]["log_stats_t"]
        #     >= self.task2args[task].learner_log_interval
        # ):
        #     self.logger.log_stat(f"train/{task}/value_loss", value_loss.item(), t_env)
        #     self.task2train_info[task]["log_stats_t"] = t_env

        # return value_loss, ssl_loss
        return value_loss
        ####

    def train_planner(
        self,
        batch: EpisodeBatch,
        t_env: int,
        episode_num: int,
        task: str,
        value_loss=None,
        dec_loss=None,
        cls_loss=None,
        ssl_loss=None,
    ):
        # Get the relevant quantities
        # print("training planner!!")
        states = batch["state"][:, :]
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]

        task2decomposer = self.task2decomposer[task]
        n_agents = task2decomposer.n_agents
        n_enenmies = task2decomposer.n_enemies
        n_entities = n_agents + n_enenmies
        n_actions_no_attack = task2decomposer.n_actions_no_attack
        device = actions.device

        #### planner inference
        enc_out_h = []
        t_mask = []
        values = []
        merge_outs = []
        target_values = []
        mac_out_h, target_mac_out_h = [], []
        act_outs = []
        global_outs = []
        kl_loss = th.tensor(0.0).to(device)
        infer_loss = th.tensor(0.0).to(device)
        policy_loss = th.tensor(0.0).to(device)
        b, t, n = actions.shape[0], actions.shape[1], actions.shape[2]
        act_dim = n_actions_no_attack + n_enenmies

        self.mac.init_hidden(batch.batch_size, task)
        self.target_mac.init_hidden(batch.batch_size, task)

        # t = random.randint(0, batch.max_seq_length - 1)
        # if (t + self.main_args.context_len) < batch.max_seq_length:
        #     end_t = t + self.main_args.context_len - 1
        #     t_len = self.main_args.context_len
        # else:
        #     end_t = batch.max_seq_length - 1
        #     t_len = batch.max_seq_length - t

        for t in range(batch.max_seq_length - 1):
            # for i in range(t_len):
            value, _ = self.mac.forward_value(
                batch,
                t=t,
                task=task,
                actions=actions[:, t],
            )
            with th.no_grad():
                target_value, _ = self.target_mac.forward_value(
                    batch, t=t, task=task, actions=actions[:, t]
                )

            out_h = self.mac.forward_encoder(batch, t=t, task=task)

            values.append(value)
            target_values.append(target_value)
            enc_out_h.append(out_h)
            # in_actions.append(actions[:, t + i])
            t_mask.append(1)

            # ### trajectory modeling
            # infer_h = self.mac.forward_latent_planner(
            #     out_h,
            #     rewards=None,
            #     actions=None,
            #     t=t,
            #     # end_t=t,
            #     task=task,
            #     test_mode=False,
            # )
            # ### action decoding
            # # for i in range(t_len):

            # # compute trajectory modeling loss
            # merge_h, rec_loss = self.mac.forward_reconstruction(
            #     batch, infer_h, t=t, task=task, actions=actions[:, t]
            # )
            # merge_outs.append(merge_h)
            # infer_loss += rec_loss / mask[:, t].sum() / self.task2n_agents[task]

            act_out = self.mac.forward_global_action(
                batch,
                out_h,
                # th.zeros(batch.batch_size, n_agents, self.entity_embed_dim),
                t=t,
                task=task,
            )
            act_outs.append(act_out)

            with th.no_grad():
                global_h = self.mac.forward_global_hidden(
                    batch, t=t, task=task, actions=actions[:, t]
                )
                global_outs.append(global_h)

        # ### zero padding
        # if t_len < self.main_args.context_len:
        #     for _ in range(self.main_args.context_len - t_len):
        #         enc_out_h.insert(0, th.zeros(out_h.shape).to(out_h.device))
        #         values.insert(0, th.zeros(v_h.shape).to(v_h.device))
        #         in_actions.insert(
        #             0, th.zeros(actions[:, 0].shape).long().to(actions.device)
        #         )
        #         t_mask.insert(0, 0)
        ### list -> tensor
        values = th.stack(values, dim=1)  ### (bs, t_len, n_agents, emb_dim)
        target_values = th.stack(
            target_values, dim=1
        )  ### (bs, t_len, n_agents, emb_dim)
        enc_out_h = th.stack(enc_out_h, dim=1)  ### (bs, t_len, n_agents, emb_dim)
        global_outs = th.stack(global_outs, dim=1)
        # merge_outs = th.stack(merge_outs, dim=1)
        t_mask = th.tensor(t_mask).to(device)  ### (bs, t_len)

        ### zero padding
        # if t_len < self.main_args.context_len:
        #     for _ in range(self.main_args.context_len - t_len):
        #         act_outs.insert(0, th.zeros(act_out.shape).to(act_out.device))
        ### list -> tensor
        act_outs = th.stack(act_outs, dim=1)  ### (bs, t_len, n_agents, act_dim)
        ### add time mask
        # infer_h = th.einsum("ijkl,j->ijkl", infer_h, t_mask)
        # act_outs = th.einsum("ijkl,j->ijkl", act_outs, t_mask)

        # ### value estimation
        # values = self.mac.forward_value(batch, enc_out_h, t, task).reshape(
        #     b, self.main_args.context_len, -1, 1
        # )[:, :, :n_agents]
        # batch_state = batch["state"][:, t:end_t]
        # if t_len < self.main_args.context_len:
        #     tmp = th.stack(
        #         [
        #             th.zeros(batch_state[:, 0].shape)
        #             for _ in range(self.main_args.context_len - t_len)
        #         ],
        #         dim=1,
        #     ).to(device)
        #     batch_state = th.cat([tmp, batch_state], dim=1)
        # # with th.no_grad():
        # #     target_values = self.target_mac.forward_value(
        # #         batch, enc_out_h, t, task
        # #     ).reshape(b, -1, n_agents, 1)

        # compute value residual
        if self.mixer is not None:
            with th.no_grad():
                mixed_values = self.mixer(
                    values,
                    batch["state"][:, :-1],
                    self.task2decomposer[task],
                )  ### (bs, t_len, 1)
                # with th.no_grad():
                target_mixed_values = self.target_mixer(
                    target_values,
                    batch["state"][:, :-1],
                    self.task2decomposer[task],
                )
        else:
            mixed_values = values.sum(dim=2).detach()
            target_mixed_values = target_values.sum(dim=2).detach()
        # ### add time mask
        # enc_out_h = th.einsum("ijkl,j->ijkl", enc_out_h, t_mask)
        # mixed_values = th.einsum("ijk,j->ijk", mixed_values, t_mask)

        cs_rewards = batch["reward"][:, :-1]
        # cs_actions = actions[:, t : t + t_len]
        cs_mask = mask[:, :-1]

        # if t_len < self.main_args.context_len:
        #     tmp_r = th.stack(
        #         [
        #             th.zeros(cs_rewards[:, 0].shape)
        #             for _ in range(self.main_args.context_len - t_len)
        #         ],
        #         dim=1,
        #     ).to(device)
        #     tmp_a = th.stack(
        #         [
        #             th.zeros(cs_actions[:, 0].shape).long()
        #             for _ in range(self.main_args.context_len - t_len)
        #         ],
        #         dim=1,
        #     ).to(device)
        #     tmp_m = th.stack(
        #         [
        #             th.zeros(cs_mask[:, 0].shape).long()
        #             for _ in range(self.main_args.context_len - t_len)
        #         ],
        #         dim=1,
        #     ).to(device)
        #     cs_rewards = th.cat([tmp_r, cs_rewards], dim=1)
        #     cs_actions = th.cat([tmp_a, cs_actions], dim=1)
        #     cs_mask = th.cat([tmp_m, cs_mask], dim=1)

        discount = self.main_args.gamma

        td_error = (
            discount * target_mixed_values[:, 1:].detach()
            + cs_rewards[:, :-1]
            - mixed_values[:, :-1].detach()
        )
        td_error = (td_error * cs_mask[:, 1:]).sum() / cs_mask[:, 1:].sum()
        weight = th.exp(td_error * self.td_weight)
        weight = th.clamp_max(weight, 100.0).detach()

        policy_loss = (
            F.cross_entropy(
                act_outs.reshape(-1, act_dim),
                actions[:, :-1].reshape(-1).detach(),
                reduction="sum",
            )
            / cs_mask.sum()
            / self.task2n_agents[task]
        )
        for i in range(n_agents):
            kl_loss += (
                self.l2_loss(
                    enc_out_h[:, :, i].reshape(-1, self.entity_embed_dim),
                    global_outs.reshape(-1, self.entity_embed_dim).detach(),
                )
                * cs_mask[:, i]
            ).mean()
        kl_loss = kl_loss / n_agents
        # plan_loss = (
        #     F.mse_loss(infer_h[:, :-1], enc_out_h[:, 1:].detach(), reduction="sum")
        #     / cs_mask[:, 1:].sum()
        #     / n_entities
        # )
        # infer_loss = weight * infer_loss
        policy_loss = policy_loss / (batch.max_seq_length - 1)
        # infer_loss = infer_loss / (batch.max_seq_length - 1)
        # loss = infer_loss + 1e-6 * reg_loss + policy_loss
        loss = policy_loss + self.beta * kl_loss

        # target_mac_out_h = th.stack(target_mac_out_h, dim=1)

        # last_mac_out_h = th.stack(last_mac_out_h, dim=1)

        # out_h = out_h.reshape(-1, 1, self.entity_embed_dim)
        # target_out_h = target_out_h.reshape(-1, 1, self.entity_embed_dim)

        # indices = th.randperm(out_h.shape[0])
        # pos_out_h = (
        #     out_h.reshape(out_h.shape[0], -1)[indices]
        #     .reshape(out_h.shape)
        #     .detach()
        # )

        # last_out_h = (
        #     last_mac_out_h.reshape(-1, 1, self.entity_embed_dim)
        #     .repeat(1, out_h.shape[0], 1)
        #     .permute(1, 0, 2)
        # )

        # i = random.randint(0, batch.max_seq_length - 1)
        # for i in range(batch.max_seq_length - 1):
        # for i in range(i, i + time_window):
        # z = th.cat([out_h, last_out_h], dim=1)
        # with th.no_grad():
        #     z_pos = th.cat([pos_out_h, last_out_h], dim=1)

        # max_loss = self.contrastive_loss(z, z_pos.detach())
        # min_loss = self.mac.agent.planner.kl_loss(
        #     out_h.squeeze(-2), target_out_h.squeeze(-2)
        # )

        # planner_loss += max_loss + self.main_args.beta * min_loss
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

        # if len(self.last_batch) != 0:
        #     planner_loss = planner_loss / (batch.max_seq_length - 1)
        #     loss = planner_loss

        # Do RL Learning
        # self.mac.agent.encoder.requires_grad_(False)
        self.mac.agent.value.requires_grad_(False)
        # self.optimiser.zero_grad()
        loss.backward()
        # self.optimiser.step()

        # self.mac.agent.value.requires_grad_(True)

        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        # self.optimiser.step()

        # episode_num should be pulic
        if (
            t_env - self.last_target_update_episode
        ) / self.main_args.target_update_interval >= 1.0:
            # print("Update target network!")
            self._update_targets()
            self.last_target_update_episode = t_env

        if (
            t_env - self.task2train_info[task]["log_stats_t"]
            >= self.task2args[task].learner_log_interval
        ):
            if dec_loss is not None:
                self.logger.log_stat(f"{task}/dec_loss", dec_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/td_error", td_error.item(), t_env)
            if value_loss is not None:
                self.logger.log_stat(f"{task}/value_loss", value_loss.item(), t_env)
            self.logger.log_stat(f"{task}/reg_loss", kl_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/infer_loss", infer_loss.item(), t_env)
            self.logger.log_stat(f"{task}/infer_loss", policy_loss.item(), t_env)
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

    def pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.pretrain_steps == 0:
            self._reset_optimizer()
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = (
                    -task_args.learner_log_interval - 1
                )

        # value_loss = self.train_value(batch, t_env, episode_num, task)
        self.train_vae(batch, t_env, episode_num, task)
        # self.train_planner(
        #     batch,
        #     t_env,
        #     episode_num,
        #     task,
        #     # dec_loss=dec_loss,
        # )
        # self.update_last_batch(task, batch)
        # v_loss = self.train_value(batch, t_env, episode_num, task)
        # value_loss = self.train_value(batch, t_env, episode_num, task)
        # value_loss = None
        # self.train_planner(batch, t_env, episode_num, task, value_loss=value_loss)
        self.pretrain_steps += 1

    def test_pretrain(
        self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str
    ):
        self.test_vae(batch, t_env, episode_num, task)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.training_steps == 0:
            self._reset_optimizer()
            # self.device = batch.device
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
            # self.train_vae(batch, t_env, episode_num, task)
            # value_loss = None
            # value_loss = self.train_value(batch, t_env, episode_num, task)
            # bc_loss = self.train_vae(batch, t_env, episode_num, task)
            # self.update(pretrain=False)
            self.train_planner(
                batch,
                t_env,
                episode_num,
                task,
                value_loss=None,
            )
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
