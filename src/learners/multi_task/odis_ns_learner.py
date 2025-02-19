import copy
import random

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


class ODISNsLearner:
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
        self.task2train_info, self.task2encoder_params, self.task2encoder_optimiser = {}, {}, {}
        for task in self.task2args:
            task_args = self.task2args[task]
            self.task2train_info[task] = {}
            self.task2train_info[task]["log_stats_t"] = -task_args.learner_log_interval - 1

        self.c = main_args.c_step
        self.skill_dim = main_args.skill_dim
        self.beta = main_args.beta
        self.alpha = main_args.coef_conservative
        self.phi = main_args.coef_dist
        # self.kl_weight = main_args.coef_kl
        self.entity_embed_dim = main_args.entity_embed_dim
        self.mask_flag = main_args.mask

        self.pretrain_steps = 0
        self.training_steps = 0

    def _reset_optimizer(self):
        if self.main_args.optim_type.lower() == "rmsprop":
            self.pre_optimiser = RMSprop(
                params=self.params, 
                lr=self.main_args.lr, 
                alpha=self.main_args.optim_alpha, 
                eps=self.main_args.optim_eps, 
                weight_decay=self.main_args.weight_decay)
            self.optimiser = RMSprop(
                params=self.params, 
                lr=self.main_args.lr,
                alpha=self.main_args.optim_alpha, 
                eps=self.main_args.optim_eps, 
                weight_decay=self.main_args.weight_decay)
        elif self.main_args.optim_type.lower() == "adam":
            self.pre_optimiser = Adam(params=self.params, lr=self.main_args.lr, weight_decay=self.main_args.weight_decay)
            self.optimiser = Adam(params=self.params, lr=self.main_args.critic_lr, weight_decay=self.main_args.weight_decay)
        else:
            raise ValueError("Invalid optimiser type", self.main_args.optim_type)
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def zero_grad(self):
        self.pre_optimiser.zero_grad()
        self.optimiser.zero_grad()

    def update(self, pretrain=True):
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        if pretrain:
            self.pre_optimiser.step()
            self.pre_optimiser.zero_grad()
        else:
            self.optimiser.step()
            self.optimiser.zero_grad()

    def cosine_similarity(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)

        return 2 - 2*(x*y).sum(dim=-1)

    def train_vae(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        mac_out_mask = []

        # self.mac.init_hidden(batch.batch_size, task)
        # for t in range(batch.max_seq_length):
        #     agent_outs = self.mac.forward_obs_skill(batch, t=t, task=task)
        #     mac_out.append(agent_outs)
        #     # if self.mask_flag:
        #     agent_outs = self.mac.forward_obs_skill(batch, t=t, task=task, mask_flag=True)
        #     mac_out_mask.append(agent_outs)
        # mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # # if self.mask_flag:
        # mac_out_mask = th.stack(mac_out_mask, dim=1)

        ######## beta-vae loss
        # prior loss
        # seq_skill_input = F.gumbel_softmax(mac_out[:, :-self.c, :, :], dim=-1)
        # kl_seq_skill = seq_skill_input * (th.log(seq_skill_input) - math.log(1 / self.main_args.entity_embed_dim))
        # enc_loss = kl_seq_skill.mean()

        dec_loss = 0.   ### batch time agent skill
        ensemble_loss = 0.
        ent_loss = 0.
        self.mac.init_hidden(batch.batch_size, task)
        # self.target_mac.init_hidden(batch.batch_size, task)

        # seq_skill = mac_out[:, :-self.c]
        # seq_skill_mask = mac_out_mask[:, :-self.c]

        seq_skill, seq_skill_mask = [], []
        for t in range(batch.max_seq_length-self.c):
            # seq_action_output = self.mac.forward_seq_action(batch, seq_skill_input[:, t, :, :], t, task=task)
            # mask_flag = False
            # p = random.random()
            # if p > self.main_args.random and self.main_args.mask:
            #     mask_flag = True

            seq_action_output, seq_skill_output = self.mac.forward_seq_action(batch, t, task=task, mask=False)
            _, seq_skill_output_mask = self.mac.forward_seq_action(batch, t, task=task, mask=True)
            b, c, n, a = seq_action_output.size()
            dec_loss += (
                F.cross_entropy(
                    seq_action_output.reshape(-1, a), 
                    # actions[:, t: t + self.c].squeeze(-1).reshape(-1), 
                    # reduction="sum") / mask[:, t:t + self.c].sum()
                    actions[:, t:t+self.c].squeeze(-1).reshape(-1), 
                    reduction="sum") / mask[:, t:t+self.c].sum()
            ) / n

            seq_skill.append(seq_skill_output)
            seq_skill_mask.append(seq_skill_output_mask)

        # CQL start

        assert self.c == 1
        seq_skill = th.stack(seq_skill, dim=1).squeeze(2)   # b, t_len, n_agent, entity_embed_dim
        seq_skill_mask = th.stack(seq_skill_mask, dim=1).squeeze(2)   # b, t_len, n_agent, entity_embed_dim
        # seq_obs_input = seq_obs_input.squeeze(2)
        # seq_q_output = self.mac.forward_qvalue(batch, task=task, attn_skill=seq_skill, attn_obs=seq_obs_input)
        # # seq_skill: b, len, n_a, dim; seq_obs_input: b, len, n_a, 1+n_enemy, dim
        # # seq_q_output: b, len, n_a, act_dim
        #
        # # Pick the Q-Values for the actions taken by each agent
        # chosen_action_qvals = th.gather(
        #     seq_q_output, 
        #     dim=-1, 
        #     index=actions[:, :-self.c]
        # ).squeeze(-1)  # Remove the last dim

        # # Calculate the Q-Values necessary for the target

        # self consistant and team consistant,
        # self consistant is achieved by constrastive learning,
        # team consistant is achieved by KL constrain.

        target_skill = []
        target_skill_mask = []
        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size, task)
            for t in range(batch.max_seq_length-self.c):
                _, target_skill_outs = self.target_mac.forward_seq_action(batch, t=t, task=task)
                target_skill.append(target_skill_outs)
                _, target_skill_outs = self.target_mac.forward_seq_action(batch, t=t, task=task, mask=True)
                target_skill_mask.append(target_skill_outs)

            target_seq_skill = th.stack(target_skill, dim=1).unsqueeze(2)  # Concat over time
            target_seq_skill_mask = th.stack(target_skill_mask, dim=1).unsqueeze(2)  # Concat over time
            # target_seq_skill = target_mac_out[:, :-self.c]
            # target_seq_skill_mask = target_mac_out_mask[:, :-self.c]
            target_seq_skill_ = self.target_mac.forward_contrastive(target_seq_skill)
            target_seq_skill_mask_ = self.target_mac.forward_contrastive(target_seq_skill_mask)
            target_seq_skill_ = target_seq_skill_.reshape(-1, self.skill_dim)
            target_seq_skill_mask_ = target_seq_skill_mask_.reshape(-1, self.skill_dim)

        seq_skill_ = self.mac.forward_contrastive(seq_skill)
        seq_skill_mask_ = self.mac.forward_contrastive(seq_skill_mask)
        seq_skill_ = seq_skill_.reshape(-1, self.skill_dim)
        seq_skill_mask_ = seq_skill_mask_.reshape(-1, self.skill_dim)

        # bs = seq_skill.shape[0]

        c_loss_1 = self.cosine_similarity(seq_skill_, target_seq_skill_mask_.detach())
        c_loss_2 = self.cosine_similarity(seq_skill_mask_, target_seq_skill_.detach())
        contrastive_loss = c_loss_1 + c_loss_2
        contrastive_loss = contrastive_loss.mean()

        #     target_seq_obs = []
        #     self.target_mac.init_hidden(batch.batch_size, task)
        #     for t in range(batch.max_seq_length-self.c):
        #         _, target_obs_input = self.mac.forward_seq_action(batch, t, task=task, attn_skill=target_seq_skill[:, t])
        #         target_seq_obs.append(target_obs_input)
        #
        #     assert self.c == 1
        #     target_seq_obs = th.stack(target_seq_obs, dim=1)
        #     target_seq_obs = target_seq_obs.squeeze(2)
        #     target_q_output = self.target_mac.forward_qvalue(batch, task=task, attn_skill=target_seq_skill, attn_obs=target_seq_obs)
        #
        # # We don't need the first timesteps Q-Value estimate for calculating targets
        # # Mask out unavailable actions
        # # target_mac_out[avail_actions[:, self.c:] == 0] = -9999999
        #
        # # Max over target Q-Values
        # if self.main_args.double_q:
        #     # Get actions that maximise live Q (for double q-learning)
        #     seq_q_detach = seq_q_output.clone().detach()
        #     # mac_out_detach[avail_actions == 0] = -9999999
        #     cur_max_actions = seq_q_detach[:, :].max(dim=-1, keepdim=True)[1]
        #     target_max_qvals = th.gather(target_q_output, -1, cur_max_actions).squeeze(-1)
        #     cons_max_qvals = th.gather(target_q_output, -1, cur_max_actions).squeeze(-1)
        #     # cons_error = (cons_max_qvals - chosen_action_qvals).mean(dim=-1, keepdim=True)
        # else:
        #     target_max_qvals = target_mac_out.max(dim=-1)[0]
        #     cons_error = None
        #
        # # Mix
        # # bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
        # # task_repre = self.mac.get_task_repres(task, require_grad=False)[None, None, ...].repeat(bs, seq_len, 1, 1)
        # # task_repre = self.mac.sample_task_repres(task, require_grad=False, shape=(bs, seq_len)).to(chosen_action_qvals.device)
        # if self.mixer is not None:
        #     chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-self.c],
        #                                      self.task2decomposer[task])
        #     target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, :-self.c],
        #                                          self.task2decomposer[task])
        #     cons_max_qvals = self.mixer(cons_max_qvals, batch["state"][:, :-self.c],
        #                                 self.task2decomposer[task])
        #
        # # Calculate c-step Q-Learning targets
        # cs_rewards = batch["reward"][:, :-self.c]
        # for i in range(1, self.c):
        #     cs_rewards[:, :-self.c] += rewards[:, i:-(self.c - i)]
        # # cs_rewards /= self.c
        #
        # # print(cs_rewards[:, :-self.c].shape, terminated[:, self.c:-1].shape, target_max_qvals[:, self.c:].shape)
        # targets = cs_rewards[:, :-self.c] + self.main_args.gamma * (1 - terminated[:, self.c:-1]) * target_max_qvals[:, self.c:]
        #
        # # Td-error
        # td_error = (chosen_action_qvals[:, :-self.c] - targets.detach())
        #
        # # # Cons-error
        # cons_error = (cons_max_qvals - chosen_action_qvals)
        #
        # mask = mask[:, :-self.c].expand_as(cons_error)
        #
        # # 0-out the targets that came from padded data
        # masked_td_error = td_error * mask[:, :-self.c]
        # masked_cons_error = cons_error * mask
        #
        # # Normal L2 loss, take mean over actual data
        # td_loss = (masked_td_error ** 2).sum() / mask[:, :-self.c].sum()
        # cons_loss = masked_cons_error.sum() / mask.sum()
        # q_loss = td_loss + self.alpha * cons_loss

        # CQL end

        # p = F.softmax(seq_skill_output[:, :-1], dim=-1)
        # q = F.softmax(seq_skill_output[:, 1:], dim=-1)
        # kl = F.kl_div(th.log(p), q, reduction="sum") / n
        # kl_loss += kl

        # # maximum entropy
        # seq_skill = self.mac.forward_global(seq_skill)
        # ent_loss = -th.log(F.softmax(seq_skill, dim=-1)).mean()
        # b, t, n, _ = seq_skill.shape
        #
        # with th.no_grad():
        #     target_seq_skill = F.softmax(self.target_mac.forward_global(target_seq_skill), dim=-1)
        #     target_seq_skill = th.max(target_seq_skill, dim=-2)[0]   # b, c, skill_dim
        #     target_seq_skill = th.max(target_seq_skill, dim=-1)[1].reshape(b, t, 1, 1).repeat(1, 1, n, 1)
        #
        # ensemble_loss += (
        #     F.cross_entropy(
        #         seq_skill.reshape(-1, 5), 
        #         target_seq_skill.squeeze(-1).reshape(-1), 
        #         # reduction="sum") / mask[:, t:t + self.c].sum()
        #         reduction="sum") / mask.sum()
        # ) / n
        # ensemble_loss += self.main_args.coef_ent * ent_loss
        # print(ensemble_loss.shape, ent.shape)

        # vae_loss = dec_loss / (batch.max_seq_length - self.c) + self.main_args.beta * enc_loss
        # vae_loss = dec_loss / (batch.max_seq_length - self.c)
        # vae_loss = dec_loss + self.main_args.coef_ensemble * ensemble_loss
        vae_loss = dec_loss
        vae_loss /= (batch.max_seq_length - self.c)
        # vae_loss += self.main_args.coef_ent * ent_loss
        # vae_loss = dec_loss + self.kl_weight * kl_loss 
        # loss = vae_loss / (batch.max_seq_length - self.c)
        loss = vae_loss + self.main_args.coef_sim * contrastive_loss
        # loss = vae_loss
        # loss += q_loss

        # self.optimiser.zero_grad()
        loss.backward()
        # self.optimiser.step()

        if (t_env - self.last_target_update_episode) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = t_env

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            # self.logger.log_stat(f"pretrain/{task}/grad_norm", grad_norm.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/vae_loss", vae_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/dist_loss", dist_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/enc_loss", enc_loss.item(), t_env)
            # self.logger.log_stat(f"pretrain/{task}/dec_loss", dec_loss.item(), t_env)
            self.logger.log_stat(f"train/{task}/vae_loss", vae_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/sim_loss", contrastive_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/q_loss", q_loss.item(), t_env)
            # self.logger.log_stat(f"train/{task}/dec_loss", dec_loss.item(), t_env)
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

        dec_loss = 0.   ### batch time agent skill
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length-self.c):
            # seq_action_output = self.mac.forward_seq_action(batch, seq_skill_input[:, t, :, :], t, task=task)
            seq_action_output = self.mac.forward_seq_action(batch, t, task=task)
            b, c, n, a = seq_action_output.size()
            dec_loss += (F.cross_entropy(seq_action_output.reshape(-1, a), actions[:, t:t + self.c].squeeze(-1).reshape(-1), reduction="sum") / mask[:, t:t + self.c].sum()) / n

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

    def train_policy(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        # Get the relevant quantities
        rewards = batch["reward"][:, :]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :].float()
        mask = batch["filled"][:, :].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # avail_actions = batch["avail_actions"]

        #### encode action
        with th.no_grad():
            new_actions = []
            self.mac.init_hidden(batch.batch_size, task)
            for t in range(batch.max_seq_length):
                action = self.mac.forward_skill(batch, t=t, task=task, actions=actions[:, t, :])
                label_action = action.max(dim=-1)[1].unsqueeze(-1)
                new_actions.append(label_action)
            actions = th.stack(new_actions, dim=1)
        ####

        #### representation
        mac_out_obs = []
        mac_out = []
        self.mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            agent_outs, pri_outs = self.mac.forward_both(batch, t=t, task=task)
            mac_out.append(agent_outs)
            mac_out_obs.append(pri_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        _, _, n_agents, _ = mac_out.size()
        mac_out_obs = th.stack(mac_out_obs, dim=1)  # Concat over time
        dist_loss = F.cross_entropy(mac_out_obs.reshape(-1, self.skill_dim),
                                    actions.reshape(-1), reduction="sum") / mask.sum() / n_agents

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :], dim=3, index=actions[:, :]).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size, task)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward_qvalue(batch, t=t, task=task)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

        # Mask out unavailable actions
        # target_mac_out[avail_actions[:, self.c:] == 0] = -9999999

        # Max over target Q-Values
        if self.main_args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, :].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            cons_max_qvals = th.gather(mac_out, 3, cur_max_actions).squeeze(3)
            # cons_error = (cons_max_qvals - chosen_action_qvals).mean(dim=-1, keepdim=True)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            cons_error = None

        # Mix
        bs, seq_len = chosen_action_qvals.size(0), chosen_action_qvals.size(1)
        # task_repre = self.mac.get_task_repres(task, require_grad=False)[None, None, ...].repeat(bs, seq_len, 1, 1)
        # task_repre = self.mac.sample_task_repres(task, require_grad=False, shape=(bs, seq_len)).to(chosen_action_qvals.device)
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :],
                                             self.task2decomposer[task])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, :],
                                                 self.task2decomposer[task])

            cons_max_qvals = self.mixer(cons_max_qvals, batch["state"][:, :],
                                        self.task2decomposer[task])

        # Calculate c-step Q-Learning targets
        cs_rewards = batch["reward"]
        for i in range(1, self.c):
            cs_rewards[:, :-self.c] += rewards[:, i:-(self.c - i)]
        # cs_rewards /= self.c

        targets = cs_rewards[:, :-self.c] + self.main_args.gamma * (1 - terminated[:, self.c - 1:-1]) * target_max_qvals[:, self.c:]

        # Td-error
        td_error = (chosen_action_qvals[:, :-self.c] - targets.detach())

        # # Cons-error
        cons_error = (cons_max_qvals - chosen_action_qvals)

        mask = mask[:, :].expand_as(cons_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask[:, :-self.c]
        masked_cons_error = cons_error * mask

        # Normal L2 loss, take mean over actual data
        td_loss = (masked_td_error ** 2).sum() / mask[:, :-self.c].sum()
        cons_loss = masked_cons_error.sum() / mask.sum()
        loss = td_loss + self.alpha * cons_loss + self.phi * dist_loss

        # Do RL Learning
        self.mac.agent.encoder.requires_grad_(False)
        self.mac.agent.state_encoder.requires_grad_(False)
        self.mac.agent.decoder.requires_grad_(False)
        # self.optimiser.zero_grad()
        loss.backward()
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.main_args.grad_norm_clip)
        # self.optimiser.step()

        # episode_num should be pulic
        if (t_env - self.last_target_update_episode) / self.main_args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = t_env

        if t_env - self.task2train_info[task]["log_stats_t"] >= self.task2args[task].learner_log_interval:
            self.logger.log_stat(f"{task}/loss", loss.item(), t_env)
            self.logger.log_stat(f"{task}/td_loss", td_loss.item(), t_env)
            self.logger.log_stat(f"{task}/cons_loss", cons_loss.item(), t_env)
            self.logger.log_stat(f"{task}/dist_loss", dist_loss.item(), t_env)
            # self.logger.log_stat(f"{task}/grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(f"{task}/td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat(f"{task}/q_taken_mean", (chosen_action_qvals * mask).sum().item() / (
                mask_elems * self.task2args[task].n_agents), t_env)
            self.logger.log_stat(f"{task}/target_mean",
                                 (targets * mask[:, :-self.c]).sum().item() / (mask_elems * self.task2args[task].n_agents), t_env)
            self.task2train_info[task]["log_stats_t"] = t_env

    def pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.pretrain_steps == 0:
            self._reset_optimizer()
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = -task_args.learner_log_interval - 1

        self.train_vae(batch, t_env, episode_num, task)
        self.pretrain_steps += 1

    def test_pretrain(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        self.test_vae(batch, t_env, episode_num, task)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, task: str):
        if self.training_steps == 0:
            self._reset_optimizer()
            for t in self.task2args:
                task_args = self.task2args[t]
                self.task2train_info[t]["log_stats_t"] = -task_args.learner_log_interval - 1

        # self.train_policy(batch, t_env, episode_num, task)
        self.train_vae(batch, t_env, episode_num, task)
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
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
