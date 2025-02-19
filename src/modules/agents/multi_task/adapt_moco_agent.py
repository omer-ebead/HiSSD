import collections
from re import I, L
from token import N_TOKENS
import numpy as np
import torch as th
from torch.cuda import device_of
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
import h5py
import collections

import test
from utils.embed import polynomial_embed, binary_embed
from utils.transformer import Transformer
from .vq_skill import SkillModule, MLPNet
from .net_utils import TruncatedNormal


class ADAPTMOCOAgent(nn.Module):
    """sotax agent for multi-task learning"""

    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(ADAPTMOCOAgent, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.c = args.c_step
        self.skill_dim = args.skill_dim

        self.q = Qnet(args)
        self.state_encoder = StateEncoder(
            task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        )
        # self.obs_encoder = ObsEncoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.value = ValueNet(
            task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        )
        self.encoder = Encoder(
            task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        )
        self.decoder = Decoder(
            # self.decoder = InverseDynamics(
            task2input_shape_info,
            task2decomposer,
            task2n_agents,
            decomposer,
            args,
        )
        self.planner = SequentialLatentPlanner(
            task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        )
        self.merge = MergeRec(
            task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        )
        # self.discr = Discriminator(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        # self.skill_encoder = SkillEncoder(args)

        self.last_out_h = None
        self.last_h_plan = None
        self.last_action = None

        if self.args.evaluate:
            self.tSNE_data = Eval()
            self.task_count = 0
        self.coordination = []
        self.specific = []
        # self.adaptation = args.adaptation
        self.noise_weight = args.noise_weight

    def init_hidden(self):
        # make hidden states on the same device as model
        return (
            self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
            self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
            self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
            self.encoder.q_skill.weight.new(1, self.args.entity_embed_dim).zero_(),
        )

    def forward_seq_action(
        self,
        seq_inputs,
        hidden_state_dec,
        hidden_state_plan,
        task,
        mask=False,
        t=0,
        actions=None,
    ):
        seq_act = []
        # seq_obs = []
        # hidden_state = None
        for i in range(self.c):
            act, hidden_state_dec, hidden_state_plan = self.forward_action(
                seq_inputs[:, i, :],
                hidden_state_dec,
                hidden_state_plan,
                task,
                mask,
                t,
                actions[:, i],
            )
            if i == 0:
                hidden_state = hidden_state_dec
                h_plan = hidden_state_plan
            seq_act.append(act)
            # seq_obs.append(obs)
        seq_act = th.stack(seq_act, dim=1)
        # seq_obs = th.stack(seq_obs, dim=1)

        return seq_act, hidden_state, h_plan

    # def forward_global_hidden(self, inputs, task, hidden_state_enc=None, actions=None):
    #     total_hidden, h_enc = self.state_encoder(inputs, hidden_state_enc, task, actions=actions)
    #     return total_hidden, h_enc

    def forward_action(
        self,
        inputs,
        emb_inputs,
        hidden_state_dec,
        task,
        mask=False,
        t=0,
        actions=None,
    ):
        act, h_dec = self.decoder(
            emb_inputs, inputs, hidden_state_dec, task, mask, actions
        )
        # _, out_h, h_plan = self.forward_planner(inputs, hidden_state_plan, t, task)
        # pre_own, pre_enemy, pre_ally = out_h
        # pre_state = th.cat([pre_own.unsqueeze(1), pre_enemy, pre_ally], dim=1)
        # act, h_dec = self.decoder(inputs, hidden_state_dec, pre_state, task, mask, actions)
        return act, h_dec

    def forward_global_hidden(
        self, inputs, hidden_state_global, task, t=0, actions=None
    ):
        attn_out, hidden_state_global = self.state_encoder(
            inputs, hidden_state_global, task, t=t, actions=actions
        )
        return attn_out, hidden_state_global

    def forward_encoder(self, inputs, hidden_state_local, task, t=0, actions=None):
        attn_out, hidden_state_local = self.encoder(
            inputs, hidden_state_local, task, t=t, actions=actions
        )
        # attn_out = self.merge(attn_out, task)
        # attn_out = self.encoder.mlp(attn_out)
        return attn_out, hidden_state_local

    def forward_global_decoder(
        self, emb_inputs, hidden_state_global_dec, task, actions=None
    ):
        ally_out, enemy_out = self.state_decoder(
            emb_inputs, hidden_state_global_dec, task=task, actions=actions
        )
        return ally_out, enemy_out, hidden_state_global_dec

    def global_state_process(self, inputs, task, actions=None):
        ally_out, enemy_out = self.state_decoder.global_process(
            inputs, task=task, actions=actions
        )
        return ally_out, enemy_out

    def forward_value(
        self, inputs, hidden_state_value, task, actions=None, emb_inputs=None
    ):
        # hidden_state_value = hidden_state_value.reshape(-1, 1, self.args.entity_embed_dim)
        # attn_out, hidden_state_value = self.value(inputs, hidden_state_value, task)
        attn_out, attn_h, hidden_state_value = self.value(
            inputs, hidden_state_value, task, emb_inputs=emb_inputs
        )
        return attn_out, attn_h, hidden_state_value

    def forward_latent_planner(
        self,
        emb_inputs,
        task,
        actions=None,
        rewards=None,
        hidden_state=None,
        test_mode=True,
    ):
        out_emb, hidden_state = self.planner(
            emb_inputs,
            task,
            actions=actions,
            rewards=rewards,
            hidden_state=hidden_state,
            test_mode=test_mode,
        )
        return out_emb, hidden_state

    def forward_planner(
        self,
        inputs,
        hidden_state_plan,
        t,
        task,
        actions=None,
        next_inputs=None,
        loss_out=False,
    ):
        # h_plan = hidden_state_plan.reshape(-1, 1, self.args.entity_embed_dim)
        out_h, h = self.planner(
            inputs,
            hidden_state_plan,
            t,
            task,
            next_inputs=next_inputs,
            actions=actions,
            loss_out=loss_out,
        )
        return out_h, h

    def forward_reconstruction(self, emb_inputs, states, task, t=0, actions=None):
        emb_out, loss = self.merge(emb_inputs, states, task=task, t=t, actions=actions)
        return emb_out, loss

    def forward_contrastive(self, inputs, inputs_pos):
        logits = self.planner.compute_logits(inputs, inputs_pos)
        return logits

    def reset_planner(self, task, device, bs):
        self.planner.init_seq_obs()
        self.planner.reset_seq_obs(task, device, bs)

    def forward(
        self,
        inputs,
        hidden_state_enc,
        hidden_state_dec,
        hidden_state_plan,
        t,
        task,
        skill=None,
        mask=False,
        actions=None,
        local_obs=None,
        test_mode=None,
    ):
        device = inputs.device

        out_h, h_enc = self.forward_encoder(inputs, hidden_state_enc, task)
        out_h = out_h.reshape(
            # 1, self.task2n_agents[task], -1, self.args.entity_embed_dim
            1,
            -1,
            self.args.entity_embed_dim,
        )
        # if t == 0:
        #     self.reset_planner(task, device, bs=out_h.shape[0])
        #     # self.last_action = th.zeros(out_h.shape[0], 1).long().to(device)
        # infer_h, h_plan = self.forward_latent_planner(
        #     out_h, task, hidden_state=hidden_state_plan, actions=None
        # )
        # infer_h = infer_h.reshape(
        #     1,
        #     self.args.context_len,
        #     self.task2n_agents[task],
        #     -1,
        #     self.args.entity_embed_dim,
        # )
        # merge_h, _ = self.forward_reconstruction(infer_h, None, task, t, None)
        h_plan = hidden_state_plan
        self.last_out_h, self.last_h_plan = out_h, h_enc

        # out_h, h_plan = self.forward_global_hidden(inputs, task, hidden_state_plan, actions)
        # out_h = out_h.reshape(-1, 1, self.args.entity_embed_dim)
        # out_h = th.cat(out_h, dim=2)

        if not test_mode and self.args.adaptation:
            own_d, enemy_d, ally_d = (
                self.last_out_h[0].shape[1],
                self.last_out_h[1].shape[1],
                self.last_out_h[2].shape[1],
            )
            high_hidden = th.cat(self.last_out_h, dim=1)
            noise = 2 * th.rand_like(high_hidden) - 1
            high_hidden += noise
            own_hidden, enemy_hidden, ally_hidden = (
                high_hidden[:, :own_d],
                high_hidden[:, own_d : own_d + enemy_d],
                high_hidden[:, -ally_d:],
            )
            self.last_out_h = [own_hidden, enemy_hidden, ally_hidden]

        act, h_dec = self.decoder(out_h, inputs, hidden_state_dec, task, mask, actions)
        # pre_state = th.cat([pre_own.unsqueeze(1), pre_enemy, pre_ally], dim=1)
        # act, h_dec = self.decoder(inputs, hidden_state_dec, out_h, task)

        if self.args.evaluate:
            if task != self.tSNE_data.get_last_task():
                print(f"Task: {self.tSNE_data.get_last_task()} done!")
                self.task_count += 1
                if len(self.coordination) != 0:
                    self.tSNE_data.write_data(self.coordination, self.specific)
                self.tSNE_data.write_task(task)
                self.coordination, self.specific = [], []
            # out_h = th.cat(self.last_out_h, dim=1)
            out_h = self.last_out_h
            self.coordination.append(out_h.cpu())
            # self.specific.append(discr_h.cpu())
            # print(out_h.cpu().shape)
            # print(discr_h.cpu().shape)

            if self.task_count == 6:
                coordination, specific = self.tSNE_data.get_data()
                # data = {'coordination': coordination,
                #         'specific': specific,
                #         }
                with h5py.File("tSNE_coordination.h5", "w") as h5file:
                    for name, value in coordination.items():
                        # print(name)
                        value = th.stack(value, dim=0)
                        # print(value.shape)
                        h5file.create_dataset(name, data=value)
                with h5py.File("tSNE_specific", "w") as h5file:
                    for name, value in specific.items():
                        # print(name)
                        value = th.stack(value, dim=0)
                        # print(value.shape)
                        h5file.create_dataset(name, data=value)
                print("Representation has been saved!")
                # print('=' * 50)

        return act, h_enc, h_dec, h_plan, skill


class StateEncoder(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(StateEncoder, self).__init__()

        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape information
        state_nf_al, state_nf_en, timestep_state_dim = (
            task2decomposer_.state_nf_al,
            task2decomposer_.state_nf_en,
            task2decomposer_.timestep_number_state_dim,
        )
        self.state_last_action, self.state_timestep_number = (
            task2decomposer_.state_last_action,
            task2decomposer_.state_timestep_number,
        )

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_encoder = nn.Linear(
                state_nf_al + (self.n_actions_no_attack + 1) * 2, self.entity_embed_dim
            )
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)
            # state_nf_al += self.n_actions_no_attack + 1
        else:
            self.ally_encoder = nn.Linear(
                state_nf_al + (self.n_actions_no_attack + 1), self.entity_embed_dim
            )
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)

        # we ought to do attention
        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

        self.ln = nn.LayerNorm(self.entity_embed_dim)
        self.ally_to_ally = nn.Linear(self.entity_embed_dim * 2, self.entity_embed_dim)
        self.ally_to_enemy = nn.Linear(self.entity_embed_dim * 2, self.entity_embed_dim)
        self.representation = None

    def forward(self, states, hidden_state, task, t=0, actions=None):
        states = states.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        bs = states.size(0)
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies
        if t == 0:
            self.representation = nn.Parameter(
                th.randn(1, 1, self.entity_embed_dim).to(states.device)
            ).repeat(bs, 1, 1)

        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = (
            task_decomposer.decompose_state(states)
        )
        ally_states = th.stack(ally_states, dim=0)  # [n_agents, bs, 1, state_nf_al]

        _, current_attack_action_info, current_compact_action_states = (
            task_decomposer.decompose_action_info(
                F.one_hot(
                    actions.reshape(-1), num_classes=self.task2last_action_shape[task]
                )
            )
        )
        current_compact_action_states = (
            current_compact_action_states.reshape(bs, n_agents, -1)
            .permute(1, 0, 2)
            .unsqueeze(2)
        )
        ally_states = th.cat([ally_states, current_compact_action_states], dim=-1)

        current_attack_action_info = current_attack_action_info.reshape(
            bs, n_agents, n_enemies
        ).sum(dim=1)
        attack_action_states = (
            (current_attack_action_info > 0)
            .type(ally_states.dtype)
            .reshape(bs, n_enemies, 1, 1)
            .permute(1, 0, 2, 3)
        )
        enemy_states = th.stack(enemy_states, dim=0)  # [n_enemies, bs, 1, state_nf_en]
        enemy_states = th.cat([enemy_states, attack_action_states], dim=-1)

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = task_decomposer.decompose_action_info(
                last_action_states
            )
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        # do inference and get entity_embed
        ally_embed = self.ally_encoder(ally_states)
        enemy_embed = self.enemy_encoder(enemy_states)

        # we ought to do self-attention
        hidden_state = hidden_state.reshape(-1, 1, 1, self.entity_embed_dim).permute(
            1, 0, 2, 3
        )
        self.representation = self.representation.unsqueeze(1).permute(1, 0, 2, 3)
        entity_embed = th.cat([ally_embed, enemy_embed, self.representation], dim=0)

        # do attention
        proj_query = (
            self.query(entity_embed)
            .permute(1, 2, 0, 3)
            .reshape(bs, n_entities + 1, self.attn_embed_dim)
        )
        proj_key = (
            self.key(entity_embed)
            .permute(1, 2, 3, 0)
            .reshape(bs, self.attn_embed_dim, n_entities + 1)
        )
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = entity_embed.permute(1, 2, 3, 0).reshape(
            bs, self.entity_embed_dim, n_entities + 1
        )
        # attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)[:, :n_agents, :]
        attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)
        # .reshape(bs, n_entities, self.entity_embed_dim)[:, :n_agents, :]

        # hidden_state = attn_out[:, -1].reshape(bs, 1, self.entity_embed_dim)
        hidden_state = hidden_state.reshape(bs, -1, self.entity_embed_dim)
        # attn_out = attn_out[:, :n_agents].reshape(bs, n_agents, self.entity_embed_dim)
        attn_out = attn_out[:, -1].reshape(bs, 1, self.entity_embed_dim)
        self.representation = attn_out
        return attn_out, hidden_state
        # ally_out = attn_out[:, :n_agents]
        # enemy_out = attn_out[:, n_agents:]
        #
        # ally_self_a = ally_out.unsqueeze(2).repeat(1, 1, n_agents, 1)
        # ally_self_e = ally_out.unsqueeze(2).repeat(1, 1, n_enemies, 1)
        # ally_relat = ally_out.unsqueeze(1).repeat(1, n_agents, 1, 1)
        # enemy_out = enemy_out.unsqueeze(1).repeat(1, n_agents, 1, 1)
        #
        # enemy_out = th.cat([ally_self_e, enemy_out], dim=-1)
        # _ally_relat = th.cat([ally_self_a, ally_relat], dim=-1)
        #
        # # ally_self = self.ally_forward(ally_out)
        # enemy_relat = self.ally_to_enemy(enemy_out).reshape(bs, task_n_agents, -1, self.entity_embed_dim)
        # ally_relat = self.ally_to_ally(_ally_relat)[:, :, 1:].reshape(bs, task_n_agents, -1, self.entity_embed_dim)
        # ally_self = ally_out.unsqueeze(2).reshape(bs, task_n_agents, -1, self.entity_embed_dim)

        # return [ally_self, enemy_relat, ally_relat], hidden_state
        # total_hidden = th.cat([ally_self.unsqueeze(2), enemy_relat, ally_relat], dim=2)
        # total_hidden = self.ln(total_hidden)

        # attn_out = attn_out.reshape(bs * n_agents, self.entity_embed_dim)
        # return total_hidden, hidden_state


class SequentialLatentPlanner(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(SequentialLatentPlanner, self).__init__()

        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.entity_embed_dim = args.entity_embed_dim
        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack
        self.context_len = args.context_len

        self.position_encoder = nn.Embedding(self.context_len, self.entity_embed_dim)
        self.action_encoder = nn.Embedding(
            self.n_actions_no_attack + 1, self.entity_embed_dim
        )
        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )

        self.init_seq_obs()

    def init_seq_obs(self):
        self.embeddings = collections.deque([], maxlen=self.context_len)
        self.times = collections.deque([], maxlen=self.context_len)
        self.actions = collections.deque([], maxlen=self.context_len)
        self.rewards = collections.deque([], maxlen=self.context_len)
        self.mask = None
        self.t = 0

    def reset_seq_obs(self, task, device, bs=1):
        task_decomposer = self.task2decomposer[task]
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        # n_tokens = n_entities
        # n_tokens = 3 if self.args.value else 2
        n_tokens = n_entities

        self.t = 0
        for _ in range(self.context_len):
            self.embeddings.append(
                th.zeros(bs, n_agents, n_entities, self.entity_embed_dim).to(device)
            )
            self.times.append(th.tensor(self.t).reshape(1).to(device))
            self.actions.append(th.zeros(bs, n_agents, 1).to(device))
            self.rewards.append(
                th.zeros(bs, n_agents, self.entity_embed_dim).to(device)
            )

        # self.mask = th.tril(
        #     th.ones(n_tokens * self.context_len, n_tokens * self.context_len).to(device)
        # ).detach()

        # self.mask = (
        #     self.mask.reshape(self.context_len, 1, self.context_len, 1)
        #     .repeat(1, n_tokens, 1, n_tokens)
        #     .reshape(1, n_tokens * self.context_len, n_tokens * self.context_len)
        # )

    def append_obs(self, embeddings, bs=1, actions=None, rewards=None):
        self.embeddings.append(embeddings)
        if self.t < self.context_len - 1:
            self.t += 1
            self.times.append(th.tensor(self.t).reshape(1).to(embeddings.device))
        if actions is not None:
            self.actions.append(actions)
        if rewards is not None:
            self.rewards.append(rewards)

    def get_obs(self, device):
        obs = {
            "embeddings": th.stack(list(self.embeddings), dim=1).to(device),
            "actions": (
                th.stack(list(self.actions), dim=1).long().to(device)
                if len(self.actions) != 0
                else None
            ),
            "rewards": (
                th.stack(list(self.rewards), dim=1).to(device)
                if len(self.rewards) != 0
                else None
            ),
            "times": th.stack(list(self.times), dim=1).long().to(device),
        }
        return obs

    def forward(
        self,
        emb_inputs,
        task,
        actions=None,
        rewards=None,
        hidden_state=None,
        test_mode=True,
    ):
        device = emb_inputs.device
        ### emb_inputs: bs, seq_len, context_len * n_tokens, emb_dim
        if test_mode:
            self.append_obs(
                emb_inputs,
                bs=emb_inputs.shape[0],
                actions=actions,
                rewards=rewards,
            )
            obs = self.get_obs(device)
            observations, _actions, rewards, times = (
                obs["embeddings"],
                obs["actions"],
                obs["rewards"],
                obs["times"],
            )
            actions = th.cat(
                [
                    _actions,
                    th.zeros(_actions[:, 0].shape).unsqueeze(1).long().to(device),
                ],
                dim=1,
            )[:, 1:]
            actions = None
        else:
            observations = emb_inputs
            times = th.tensor([i for i in range(self.context_len)]).to(device).long()

        task_decomposer = self.task2decomposer[task]
        last_action_shape = self.task2last_action_shape[task]

        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies
        n_tokens = 1 * observations.shape[-2]

        observations = observations.reshape(
            -1, self.context_len, n_tokens, self.entity_embed_dim
        )
        tf_inputs = [observations]
        if rewards is not None and self.args.value:
            rewards = rewards.reshape(-1, self.context_len, 1, self.entity_embed_dim)
            tf_inputs.append(rewards)
            n_tokens += 1

        if actions is not None:
            actions = th.where(
                actions < self.n_actions_no_attack,
                actions,
                self.n_actions_no_attack,
            ).reshape(-1, self.context_len)
            actions = self.action_encoder(actions).unsqueeze(2)
            tf_inputs.append(actions)
            n_tokens += 1

        n_hidden = 0
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
            #     # tf_inputs.append(hidden_state)
            n_hidden += 1

        time_emb = self.position_encoder(times.long())
        time_emb = time_emb.reshape(
            1, self.context_len, 1, self.entity_embed_dim
        ).repeat(observations.shape[0], 1, n_tokens, 1)

        tf_inputs = th.cat(tf_inputs, dim=2)
        # observations = observations.reshape(
        #     -1, self.context_len, n_tokens, self.entity_embed_dim
        # )
        # tf_inputs += time_emb

        out_emb = self.transformer(
            th.cat(
                [
                    tf_inputs.reshape(
                        -1,
                        n_tokens * self.context_len,
                        self.entity_embed_dim,
                    ),
                    hidden_state,
                ],
                dim=-2,
            ),
            # self.mask,
            None,
        )
        # if test_mode:
        #     out_emb = out_emb[:, -1:]
        # if not test_mode:
        #     out_emb = out_emb.reshape(
        #         -1, self.context_len, n_tokens, self.entity_embed_dim
        #     )[:, :, n_tokens - 2]
        hidden_state = out_emb[:, -1]

        return out_emb[:, :-1], hidden_state


class ValueNet(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(ValueNet, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)

        self.ln = nn.Sequential(nn.LayerNorm(self.entity_embed_dim), nn.Tanh())
        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )

        self.q_skill = nn.Linear(self.entity_embed_dim, self.skill_dim)
        # self.n_actions_no_attack = n_actions_no_attack
        self.reward_fc = nn.Linear(self.entity_embed_dim, 1)
        # self.reward_fc = nn.Sequential(
        #     nn.Linear(self.entity_embed_dim, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 1),
        # )

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.entity_embed_dim).zero_()

    def encode(self, inputs, hidden_state, task, emb_inputs=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        if emb_inputs is not None:
            emb_inputs = emb_inputs.reshape(-1, 1, self.entity_embed_dim)
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = (
            inputs[:, :obs_dim],
            inputs[:, obs_dim : obs_dim + last_action_shape],
            inputs[:, obs_dim + last_action_shape :],
        )

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs
        )  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(
                binary_embed(i + 1, self.args.id_length, self.args.max_agent),
                dtype=own_obs.dtype,
            )
            for i in range(task_n_agents)
        ]
        agent_id_inputs = (
            th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        )
        _, attack_action_info, compact_action_states = (
            task_decomposer.decompose_action_info(last_action_inputs)
        )

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats).permute(1, 0, 2)
        enemy_hidden = self.enemy_value(enemy_feats).permute(1, 0, 2)
        history_hidden = hidden_state

        # own_emb_inputs, enemy_emb_inputs, ally_emb_inputs = emb_inputs
        # own_emb_inputs = self.agent_feedforward(own_emb_inputs)
        # enemy_emb_inputs = self.enemy_feedforward(enemy_emb_inputs)
        # ally_emb_inputs = self.ally_feedforward(ally_emb_inputs)
        # print(own_emb_inputs.shape, enemy_emb_inputs.shape, ally_emb_inputs.shape)
        # emb_hidden = th.cat([own_emb_inputs, enemy_emb_inputs, ally_emb_inputs], dim=1)
        # total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, emb_hidden, history_hidden], dim=1)
        if emb_inputs is not None:
            total_hidden = th.cat(
                [emb_inputs, own_hidden, enemy_hidden, ally_hidden], dim=1
            )
        else:
            total_hidden = th.cat(
                [own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1
            )
        return total_hidden

    def encode_for_skill(self, inputs, hidden_state, task):
        own_obs, enemy_feats, ally_feats = inputs

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)
        history_hidden = hidden_state

        # own_emb_inputs, enemy_emb_inputs, ally_emb_inputs = emb_inputs
        # own_emb_inputs = self.agent_feedforward(own_emb_inputs)
        # enemy_emb_inputs = self.enemy_feedforward(enemy_emb_inputs)
        # ally_emb_inputs = self.ally_feedforward(ally_emb_inputs)
        # print(own_emb_inputs.shape, enemy_emb_inputs.shape, ally_emb_inputs.shape)
        # emb_hidden = th.cat([own_emb_inputs, enemy_emb_inputs, ally_emb_inputs], dim=1)
        # total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, emb_hidden, history_hidden], dim=1)
        total_hidden = th.cat(
            [own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1
        )
        return total_hidden

    def predict(self, total_hidden):
        outputs = self.transformer(total_hidden, None)
        h = outputs[:, -1, :]
        reward_h = outputs[:, 0, :]
        reward = self.reward_fc(reward_h)
        return reward, reward_h, h

    def predict_mlp(self, emb_inputs, n_agents):
        reward = self.reward_fc(emb_inputs)
        return reward

    def forward(self, inputs, hidden_state, task, emb_inputs=None):
        task_n_agents = self.task2n_agents[task]
        total_hidden = self.encode(inputs, hidden_state, task, emb_inputs=emb_inputs)
        reward, reward_h, h = self.predict(total_hidden)
        # reward = self.predict_mlp(emb_inputs, task_n_agents)
        return reward, reward_h, h


class Encoder(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(Encoder, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args
        self.skill_dim = args.skill_dim

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.act_encode = nn.Embedding(2, self.entity_embed_dim)

        # self.own_merge = nn.Linear(2 * self.entity_embed_dim, self.entity_embed_dim)
        # self.enemy_merge = nn.Linear(2 * self.entity_embed_dim, self.entity_embed_dim)
        # self.ally_merge = nn.Linear(2 * self.entity_embed_dim, self.entity_embed_dim)

        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )
        # self.trunk = nn.Sequential(
        #     nn.Linear(self.entity_embed_dim, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, self.entity_embed_dim),
        #     nn.LayerNorm(self.entity_embed_dim),
        #     nn.Tanh(),
        # )

        self.q_skill = nn.Linear(self.entity_embed_dim * 2, n_actions_no_attack)
        self.base_q_skill = MLPNet(
            2 * self.entity_embed_dim, n_actions_no_attack, 128, output_norm=False
        )
        self.ally_q_skill = MLPNet(2 * self.entity_embed_dim, 1, 128, output_norm=False)

        self.representation = None
        self.n_actions_no_attack = n_actions_no_attack

    # def mlp(self, inputs):
    #     out = self.trunk(inputs)
    #     return out

    def forward(self, inputs, hidden_state, task, t=0, actions=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)

        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = (
            inputs[:, :obs_dim],
            inputs[:, obs_dim : obs_dim + last_action_shape],
            inputs[:, obs_dim + last_action_shape :],
        )

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs
        )  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        if t == 0:
            self.representation = (
                nn.Parameter(
                    th.randn(1, 1, self.entity_embed_dim).to(hidden_state.device)
                )
                .repeat(bs, task_n_agents, 1)
                .reshape(-1, 1, self.entity_embed_dim)
            )

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(
                binary_embed(i + 1, self.args.id_length, self.args.max_agent),
                dtype=own_obs.dtype,
            )
            for i in range(task_n_agents)
        ]
        agent_id_inputs = (
            th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        )
        _, attack_action_info, compact_action_states = (
            task_decomposer.decompose_action_info(last_action_inputs)
        )

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        enemy_feats = enemy_feats.permute(1, 0, 2)
        ally_feats = ally_feats.permute(1, 0, 2)
        n_enemy, n_ally = enemy_feats.shape[1], ally_feats.shape[1]
        n_entity = n_enemy + n_ally + 1

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)

        total_hidden = th.cat(
            [own_hidden, enemy_hidden, ally_hidden, self.representation], dim=-2
        )
        # outputs = self.transformer(total_hidden, None)
        outputs = total_hidden

        # h_out = outputs[:, -1]
        h_out = hidden_state
        outputs = outputs[:, -1]

        return outputs, h_out


class StateDecoder(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(StateDecoder, self).__init__()

        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args
        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape information
        state_nf_al, state_nf_en, timestep_state_dim = (
            task2decomposer_.state_nf_al,
            task2decomposer_.state_nf_en,
            task2decomposer_.timestep_number_state_dim,
        )
        self.state_last_action, self.state_timestep_number = (
            task2decomposer_.state_last_action,
            task2decomposer_.state_timestep_number,
        )

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_decoder = nn.Linear(
                self.entity_embed_dim, state_nf_al + (self.n_actions_no_attack + 1) * 2
            )
        else:
            self.ally_decoder = nn.Linear(
                self.entity_embed_dim, state_nf_al + (self.n_actions_no_attack + 1)
            )
        self.enemy_decoder = nn.Linear(self.entity_embed_dim, state_nf_en + 1)

        # decoder
        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )

    def global_process(self, states, task, actions=None):
        states = states.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        bs = states.size(0)
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = (
            task_decomposer.decompose_state(states)
        )
        ally_states = th.stack(ally_states, dim=0)  # [n_agents, bs, 1, state_nf_al]

        _, current_attack_action_info, current_compact_action_states = (
            task_decomposer.decompose_action_info(
                F.one_hot(
                    actions.reshape(-1), num_classes=self.task2last_action_shape[task]
                )
            )
        )
        current_compact_action_states = (
            current_compact_action_states.reshape(bs, n_agents, -1)
            .permute(1, 0, 2)
            .unsqueeze(2)
        )
        ally_states = th.cat([ally_states, current_compact_action_states], dim=-1)

        current_attack_action_info = current_attack_action_info.reshape(
            bs, n_agents, n_enemies
        ).sum(dim=1)
        attack_action_states = (
            (current_attack_action_info > 0)
            .type(ally_states.dtype)
            .reshape(bs, n_enemies, 1, 1)
            .permute(1, 0, 2, 3)
        )
        enemy_states = th.stack(enemy_states, dim=0)  # [n_enemies, bs, 1, state_nf_en]
        enemy_states = th.cat([enemy_states, attack_action_states], dim=-1)

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = task_decomposer.decompose_action_info(
                last_action_states
            )
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        return ally_states, enemy_states

    def forward(self, emb_inputs, hidden_state, task, actions=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)

        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        total_hidden = th.cat([emb_inputs, hidden_state], dim=-2)
        outputs = self.transformer(total_hidden, None)
        ally_out = outputs[:, :task_n_agents]
        enemy_out = outputs[:, task_n_agents:]

        ally_out = self.ally_decoder(ally_out)
        enemy_out = self.enemy_decoder(enemy_out)

        return ally_out, enemy_out


class InverseDynamics(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(self, InverseDynamics).__init__()

        self.fuse = nn.Sequential(nn.Linear(), nn.LayerNorm(), nn.Tanh())
        self.base = nn.Linear()
        self.attack = nn.Linear()

    def forward(self, x):
        return


class Decoder(nn.Module):
    """sotax agent for multi-task learning"""

    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(Decoder, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim
        self.cls_dim = 3

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        # self.task_repre_dim = args.task_repre_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        # self.skill_value = nn.Linear(self.skill_dim, self.entity_embed_dim)

        # self.own_merge = nn.Linear(2 * self.entity_embed_dim, self.entity_embed_dim)
        # self.enemy_merge = nn.Linear(2 * self.entity_embed_dim, self.entity_embed_dim)
        # self.ally_merge = nn.Linear(2 * self.entity_embed_dim, self.entity_embed_dim)

        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )

        self.q_skill = nn.Linear(self.entity_embed_dim * 2, n_actions_no_attack)
        self.base_q_skill = MLPNet(
            2 * self.entity_embed_dim, n_actions_no_attack, 64, output_norm=False
        )
        self.ally_q_skill = MLPNet(2 * self.entity_embed_dim, 1, 64, output_norm=False)
        # self.base_q_skill = nn.Linear(2 * self.entity_embed_dim, n_actions_no_attack)
        # self.ally_q_skill = nn.Linear(2 * self.entity_embed_dim, 1)

        self.n_actions_no_attack = n_actions_no_attack

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def forward(self, emb_inputs, inputs, hidden_state, task, mask=False, actions=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        emb_inputs = emb_inputs.reshape(-1, 1, self.entity_embed_dim)

        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]
        # bs = emb_inputs.shape[0]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = (
            inputs[:, :obs_dim],
            inputs[:, obs_dim : obs_dim + last_action_shape],
            inputs[:, obs_dim + last_action_shape :],
        )

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs
        )  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(
                binary_embed(i + 1, self.args.id_length, self.args.max_agent),
                dtype=own_obs.dtype,
            )
            for i in range(task_n_agents)
        ]
        agent_id_inputs = (
            th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        )
        _, attack_action_info, compact_action_states = (
            task_decomposer.decompose_action_info(last_action_inputs)
        )

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        enemy_feats = enemy_feats.permute(1, 0, 2)
        ally_feats = ally_feats.permute(1, 0, 2)
        n_enemy, n_ally = enemy_feats.shape[1], ally_feats.shape[1]
        n_entity = n_enemy + task_n_agents

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)
        # emb_inputs = emb_inputs.reshape(-1, n_entity, self.entity_embed_dim).repeat(
        #     task_n_agents, 1, 1
        # )
        emb_inputs = emb_inputs.repeat(1, 1 + n_enemy, 1)

        total_hidden = th.cat(
            [
                # emb_inputs,
                own_hidden,
                enemy_hidden,
                ally_hidden,
                # emb_inputs,
                hidden_state,
            ],
            dim=-2,
        )

        outputs = self.transformer(total_hidden, None)
        hidden_state = outputs[:, -1]
        outputs = outputs[:, : 1 + n_enemy]
        outputs = th.cat([outputs, emb_inputs], dim=-1)
        # outputs = self.cross_attn(outputs[:, :-1], emb_inputs, task)

        # own_global = emb_inputs[:, :task_n_agents].reshape(bs, task_n_agents, 1, -1)
        # enemy_global = (
        #     emb_inputs[:, task_n_agents:]
        #     .reshape(bs, 1, -1, self.entity_embed_dim)
        #     .repeat(1, task_n_agents, 1, 1)
        # )
        # n_enemies = enemy_global.shape[2]
        # plan_emb = th.cat([own_global, enemy_global], dim=-2).reshape(
        #     bs * task_n_agents, -1, self.entity_embed_dim
        # )

        # outputs = th.cat([outputs, emb_inputs], dim=-1)
        # hidden_state = outputs[:, -1]
        # outputs = outputs[:, : 1 + n_enemy]
        # outputs = th.cat([plan_emb, next_plan_emb], dim=-1)

        base_action_inputs = outputs[:, 0]
        # q_base = self.q_skill(base_action_inputs)

        # base_action_inputs = self.base_emb(base_action_inputs)
        q_base = self.base_q_skill(base_action_inputs)
        # base_action_inputs = base_action_inputs.unsqueeze(1).repeat(1, enemy_feats.size(1), 1)
        # skill_emb = skill_emb.unsqueeze(1).repeat(1, enemy_feats.size(1), 1)
        attack_action_inputs = outputs[:, 1 : 1 + n_enemy]
        # attack_action_inputs = th.cat(
        # [base_action_inputs, outputs[:, 1: 1+enemy_feats.size(1), :], skill_emb], dim=-1)
        # [base_action_inputs, outputs[:, 1: 1+enemy_feats.size(1), :]], dim=-1)
        # [outputs[:, 1: 1+enemy_feats.size(1), :], skill_emb[:, 1:]], dim=-1)
        # [outputs[:, 1: 1+enemy_feats.size(1), :]], dim=-1)
        # attack_action_inputs = self.ally_emb(attack_action_inputs)
        q_attack = self.ally_q_skill(attack_action_inputs)
        # q_attack = self.ally_q_skill(outputs[:, 1: 1+enemy_feats.size(0), :])
        # for i in range(enemy_feats.size(0)):
        #     # attack_action_inputs = th.cat([outputs[:, 1+i, :], skill_emb], dim=-1)
        #     # attack_action_inputs = outputs[:, 1+i, :]
        #     attack_action_inputs = th.cat([base_action_inputs, outputs[:, 1+i, :]], dim=-1)
        #     q_enemy = self.ally_q_skill(attack_action_inputs)
        #     # q_enemy_mean = th.mean(q_enemy, 1, True)
        #     q_attack_list.append(q_enemy)
        # q_attack = th.stack(q_attack_list, dim=1).squeeze()

        # q = th.cat([q_base, q_attack], dim=-1)
        # print(q_base.shape, q_attack.shape)
        q = th.cat([q_base, q_attack.reshape(-1, n_enemy)], dim=-1)

        return q, hidden_state


class Qnet(nn.Module):

    def __init__(self, args):
        super(Qnet, self).__init__()
        self.args = args

        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim

        self.q_skill = nn.Linear(self.entity_embed_dim * 2, self.skill_dim)
        self.attack_q_skill = nn.Linear(self.entity_embed_dim * 2, 1)

    def forward(self, inputs):
        q = self.q_skill(inputs)

        return q


class InferenceModel(nn.Module):
    """Infer agent embedding"""

    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(InferenceModel, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args
        self.beta = args.beta

        self.skill_dim = args.skill_dim
        self.vq_skill = args.vq_skill

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        # self.task_repre_dim = args.task_repre_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1
        self.belif_dim = 2 * self.entity_embed_dim
        # if self.args.belif_type == "rec":
        #     self.belif_dim = 2 * self.entity_embed_dim

        # Local Encoder
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        # self.value_value = nn.Linear(1, self.entity_embed_dim)
        # self.local_tf = CrossAttention(
        #     task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        # )

        # Centralized Encoder
        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )
        # self.seq_wrapper = SequenceObservationWrapper(
        #     task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        # self.obs_decoder = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        # # Variational
        # self.var_func = nn.Linear(self.entity_embed_dim, self.belif_dim)
        # self.var_func = MLPNet(self.entity_embed_dim, self.belif_dim, 128)
        # Processer
        # self.trunk = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)

        # self.moco_trunk = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        # self.W = nn.Parameter(th.rand(self.entity_embed_dim, self.entity_embed_dim))

        self.base_q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)

        self.n_actions_no_attack = n_actions_no_attack
        # self.cat_dim = args.n_task
        self.reset_last()
        self.last_prior = None
        # self.act_emb = nn.Embedding(2, 64)
        # self.skill_module = SkillModule(args)
        # self.rec_module = MergeRec(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.base_q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def reset_last(self):
        self.last_own = None
        self.last_enemy = None
        self.last_ally = None

    def add_last(self, own, enemy, ally):
        self.last_own = own
        self.last_enemy = enemy
        self.last_ally = ally

    def avg_pooling(self, input):
        return th.mean(input, dim=-2)

    def reparameterization(self, inputs):
        inputs = self.var_func(inputs)
        mu, log_var = inputs.chunk(2, dim=-1)
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return eps * std + mu, mu, log_var

    def kl_loss(self, p, q=None, normal=False):
        target_mu, target_logvar = th.tensor(0.0), th.tensor(1.0)
        _, p_mu, p_logvar = self.reparameterization(p)
        if q is not None and not normal:
            _, target_mu, target_logvar = self.reparameterization(q)
        # print(p_mu.shape, q_mu.shape)
        kl_loss = th.mean(
            th.mean(
                -0.5
                * th.sum(
                    target_mu.detach()
                    + target_logvar.detach()
                    - p_mu**2
                    - p_logvar.exp(),
                    dim=1,
                ),
                dim=0,
            )
        )
        return kl_loss

    def compute_logits(self, z, z_pos):
        _, z, _ = self.reparameterization(z)
        with th.no_grad():
            _, z_pos, _ = self.reparameterization(z_pos)
        Wz = th.matmul(self.W, z_pos.permute(0, 2, 1))
        logits = th.matmul(z, Wz)
        logits = logits - th.max(logits, 1)[0][:, None]
        return logits

    def feedforward(self, inputs, detach=True):
        if detach:
            inputs = inputs.detach()
        inputs = self.trunk(inputs)
        return inputs

    def moco_forward(self, inputs):
        inputs = self.moco_trunk(inputs)
        return inputs

    def forward(
        self,
        inputs,
        hidden_state,
        t,
        task,
        next_inputs=None,
        actions=None,
        loss_out=False,
    ):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)

        # 	self.reset_last()
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = (
            inputs[:, :obs_dim],
            inputs[:, obs_dim : obs_dim + last_action_shape],
            inputs[:, obs_dim + last_action_shape :],
        )

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs
        )  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(
                binary_embed(i + 1, self.args.id_length, self.args.max_agent),
                dtype=own_obs.dtype,
            )
            for i in range(task_n_agents)
        ]
        agent_id_inputs = (
            th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        )
        _, attack_action_info, compact_action_states = (
            task_decomposer.decompose_action_info(last_action_inputs)
        )

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        enemy_feats = enemy_feats.permute(1, 0, 2)
        ally_feats = ally_feats.permute(1, 0, 2)
        n_enemy, n_ally = enemy_feats.shape[1], ally_feats.shape[1]

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)

        b = own_hidden.shape[0]
        local_total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden], dim=1)
        # local_total_hidden = local_total_hidden.reshape(b, -1, self.entity_embed_dim)

        # local_outputs = self.local_tf(local_total_hidden, local_total_hidden, task)[
        #     :, 0
        # ].reshape(bs, task_n_agents, self.entity_embed_dim)
        outputs = self.transformer(
            th.cat([local_total_hidden, hidden_state], dim=-2), None
        )
        h = outputs[:, -1]
        outputs = outputs[:, 0]
        if self.args.task_emb_type == "mean":
            outputs = outputs.mean(dim=1).unsqueeze(1)
        # out_h, out_mu, out_logvar = self.reparameterization(outputs)

        # own_out = self.own_fc(own_out_h).reshape(bs * task_n_agents, -1)
        # enemy_out = self.enemy_fc(enemy_out_h).reshape(bs * task_n_agents, n_enemy, -1)
        # ally_out = self.ally_fc(ally_out_h).reshape(bs * task_n_agents, n_ally, -1)

        # own_out = self.ln(own_out)
        # enemy_out = self.ln(enemy_out)
        # ally_out = self.ln(ally_out)

        # out_loss = th.tensor(0.0).to(inputs.device)

        return outputs, h


class SkillEncoder(nn.Module):
    def __init__(self, args):
        super(SkillEncoder, self).__init__()
        self.args = args

        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim
        self.value = nn.Sequential(
            nn.Linear(self.entity_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.entity_embed_dim),
            nn.LayerNorm(self.entity_embed_dim),
            nn.Tanh(),
        )
        self.skill = nn.Sequential(
            nn.Linear(self.entity_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.entity_embed_dim),
            nn.LayerNorm(self.entity_embed_dim),
            nn.Tanh(),
        )

    def forward(self, inputs):
        out = self.skill(inputs)
        return out

    def value_forward(self, inputs):
        out = self.value(inputs)
        return out


class SequenceObservationWrapper:
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(SequenceObservationWrapper, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.n_actions_no_attack = n_actions_no_attack
        self.skill_dim = args.skill_dim
        self.own_dim = wrapped_obs_own_dim
        self.enemy_dim = obs_en_dim
        self.ally_dim = obs_al_dim

        # time window
        self.num_stack_frames = args.num_stack_frames
        self.own_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.enemy_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.ally_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.value_stack = collections.deque([], maxlen=self.num_stack_frames)
        self.time_steps = collections.deque([], maxlen=self.num_stack_frames)
        self.mask = collections.deque([], maxlen=self.num_stack_frames)
        self.pos_enc = nn.Embedding(self.num_stack_frames, self.entity_embed_dim)
        self.pos = th.tensor([i for i in range(self.num_stack_frames)])

    def _get_obs(self, device):
        obs = {
            "own": th.stack(list(self.own_stack), dim=1).unsqueeze(2).to(device),
            "enemy": th.stack(list(self.enemy_stack), dim=1).to(device),
            "ally": th.stack(list(self.ally_stack), dim=1).to(device),
            "value": th.stack(list(self.value_stack), dim=1).unsqueeze(2).to(device),
            "mask": th.tensor(list(self.mask)).reshape(-1, 1).to(device),
            "time": th.tensor(list(self.time_steps)).reshape(-1, 1).to(device),
        }
        return obs

    def pad_current_episode(self, own, enemy, ally, value, n):
        for _ in range(n):
            self.own_stack.append(th.zeros_like(own))
            self.enemy_stack.append(th.zeros_like(enemy))
            self.ally_stack.append(th.zeros_like(ally))
            self.value_stack.append(th.zeros_like(value))
            self.mask.append(0)
            self.time_steps.append(0)

    def obs_reset(self, own, enemy, ally, value):
        self.pad_current_episode(own, enemy, ally, value, self.num_stack_frames - 1)
        self.time_now = 0
        self.own_stack.append(own)
        self.enemy_stack.append(enemy)
        self.ally_stack.append(ally)
        self.value_stack.append(value)
        self.mask.append(1)
        self.time_steps.append(self.time_now)
        return self._get_obs(own.device)

    def obs_step(self, own, enemy, ally, value):
        self.time_now += 1
        self.own_stack.append(own)
        self.enemy_stack.append(enemy)
        self.ally_stack.append(ally)
        self.value_stack.append(value)
        self.mask.append(1)
        self.time_steps.append(self.time_now)
        return self._get_obs(own.device)

    def position_encoding(self, obs_emb):
        # obs_emb: bs, t_len, n, dim
        bs, _, n, _ = obs_emb.shape
        pos_emb = (
            self.pos_enc(self.pos)
            .reshape(1, self.num_stack_frames, 1, self.entity_embed_dim)
            .repeat(bs, 1, n, 1)
        )
        obs_emb += pos_emb
        return obs_emb


class Discriminator(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(Discriminator, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim
        self.ssl_type = args.ssl_type

        #### define various dimension information
        ## set attributes
        self.entity_embed_dim = args.entity_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        # self.task_repre_dim = args.task_repre_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )
        self.W = nn.Parameter(th.rand(self.entity_embed_dim, self.entity_embed_dim))

        if args.ssl_type == "moco":
            self.act_proj = nn.Sequential(
                nn.Linear(self.entity_embed_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.entity_embed_dim),
                nn.LayerNorm(self.entity_embed_dim),
                nn.Tanh(),
            )
            self.ssl_proj = nn.Sequential(
                nn.Linear(self.entity_embed_dim, 128),
                #   nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.entity_embed_dim),
            )
        elif args.ssl_type == "byol":
            self.act_proj = nn.Sequential(
                nn.Linear(self.entity_embed_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.entity_embed_dim),
            )
            self.ssl_proj = nn.Sequential(
                nn.Linear(self.entity_embed_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.entity_embed_dim),
            )

    def forward(self, inputs, t, task, hidden_state):
        # hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        # get decomposer, last_action_shape and n_agents of this specific task
        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        # decompose inputs into observation inputs, last_action_info, agent_id_info
        obs_dim = task_decomposer.obs_dim
        obs_inputs, last_action_inputs, agent_id_inputs = (
            inputs[:, :obs_dim],
            inputs[:, obs_dim : obs_dim + last_action_shape],
            inputs[:, obs_dim + last_action_shape :],
        )

        # decompose observation input
        own_obs, enemy_feats, ally_feats = task_decomposer.decompose_obs(
            obs_inputs
        )  # own_obs: [bs*self.n_agents, own_obs_dim]
        bs = int(own_obs.shape[0] / task_n_agents)

        # embed agent_id inputs and decompose last_action_inputs
        agent_id_inputs = [
            th.as_tensor(
                binary_embed(i + 1, self.args.id_length, self.args.max_agent),
                dtype=own_obs.dtype,
            )
            for i in range(task_n_agents)
        ]
        agent_id_inputs = (
            th.stack(agent_id_inputs, dim=0).repeat(bs, 1).to(own_obs.device)
        )
        _, attack_action_info, compact_action_states = (
            task_decomposer.decompose_action_info(last_action_inputs)
        )

        # incorporate agent_id embed and compact_action_states
        own_obs = th.cat([own_obs, agent_id_inputs, compact_action_states], dim=-1)

        # incorporate attack_action_info into enemy_feats
        attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
        enemy_feats = th.cat([th.stack(enemy_feats, dim=0), attack_action_info], dim=-1)
        ally_feats = th.stack(ally_feats, dim=0)

        enemy_feats = enemy_feats.permute(1, 0, 2)
        ally_feats = ally_feats.permute(1, 0, 2)
        # n_enemy, n_ally = enemy_feats.shape[1], ally_feats.shape[1]

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)
        history_hidden = hidden_state

        b = own_hidden.shape[0]
        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden], dim=1)
        outputs = self.transformer(total_hidden, None)
        # h = outputs[:, -1]
        # outputs = outputs[:, :-1]
        h = history_hidden

        own_out_h = outputs[:, 0].reshape(-1, self.entity_embed_dim)
        own_out = self.ssl_proj(own_out_h)
        if self.ssl_type == "moco":
            own_out_h = self.act_proj(own_out_h)
        elif self.ssl_type == "byol":
            own_out_h = self.act_proj(own_out)
        # own_out = self.pre(own_out_h)

        return own_out, own_out_h, h

    def compute_logits(self, z, z_pos):
        Wz = th.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = th.matmul(z, Wz)  # (B,B)
        logits = logits - th.max(logits, 1)[0][:, None]
        return logits


class CrossAttention(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(CrossAttention, self).__init__()

        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # # get detailed state shape information
        # state_nf_al, state_nf_en, timestep_state_dim = (
        #     task2decomposer_.state_nf_al,
        #     task2decomposer_.state_nf_en,
        #     task2decomposer_.timestep_number_state_dim,
        # )
        self.state_last_action, self.state_timestep_number = (
            task2decomposer_.state_last_action,
            task2decomposer_.state_timestep_number,
        )

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # # define state information processor
        # if self.state_last_action:
        #     self.ally_encoder = nn.Linear(
        #         state_nf_al + (self.n_actions_no_attack + 1) * 2, self.entity_embed_dim
        #     )
        #     self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)
        #     # state_nf_al += self.n_actions_no_attack + 1
        # else:
        #     self.ally_encoder = nn.Linear(
        #         state_nf_al + (self.n_actions_no_attack + 1), self.entity_embed_dim
        #     )
        #     self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)

        # we ought to do attention
        self.query = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)
        self.key = nn.Linear(self.entity_embed_dim, self.attn_embed_dim)

    def forward(self, dec_emb, skill_emb, task, actions=None):
        # skill_emb = th.cat(skill_emb, dim=1)
        dec_emb, skill_emb = dec_emb.unsqueeze(1), skill_emb.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies
        bs = dec_emb.shape[0]

        # do attention
        proj_query = self.query(dec_emb).reshape(bs, n_entities, self.attn_embed_dim)
        proj_key = (
            self.key(skill_emb)
            .permute(0, 1, 3, 2)
            .reshape(bs, self.attn_embed_dim, n_entities)
        )
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = dec_emb.permute(0, 1, 3, 2).reshape(
            bs, self.entity_embed_dim, n_entities
        )
        # attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)[:, :n_agents, :]
        attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)
        # .reshape(bs, n_entities, self.entity_embed_dim)[:, :n_agents, :]

        attn_out = attn_out.reshape(bs, n_entities, self.entity_embed_dim)
        return attn_out


class MergeRec(nn.Module):
    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
    ):
        super(MergeRec, self).__init__()
        self.task2last_action_shape = {
            task: task2input_shape_info[task]["last_action_shape"]
            for task in task2input_shape_info
        }
        self.task2decomposer = task2decomposer
        for key in task2decomposer.keys():
            task2decomposer_ = task2decomposer[key]
            break

        self.task2n_agents = task2n_agents
        self.args = args

        self.skill_dim = args.skill_dim

        self.embed_dim = args.mixing_embed_dim
        self.attn_embed_dim = args.attn_embed_dim
        self.entity_embed_dim = args.entity_embed_dim

        # get detailed state shape information
        state_nf_al, state_nf_en, timestep_state_dim = (
            task2decomposer_.state_nf_al,
            task2decomposer_.state_nf_en,
            task2decomposer_.timestep_number_state_dim,
        )
        self.state_last_action, self.state_timestep_number = (
            task2decomposer_.state_last_action,
            task2decomposer_.state_timestep_number,
        )

        self.n_actions_no_attack = task2decomposer_.n_actions_no_attack

        # define state information processor
        if self.state_last_action:
            self.ally_encoder = nn.Linear(
                state_nf_al + (self.n_actions_no_attack + 1) * 2, self.entity_embed_dim
            )
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)
            # state_nf_al += self.n_actions_no_attack + 1
        else:
            self.ally_encoder = nn.Linear(
                state_nf_al + (self.n_actions_no_attack + 1), self.entity_embed_dim
            )
            self.enemy_encoder = nn.Linear(state_nf_en + 1, self.entity_embed_dim)

        # we ought to do attention
        self.own_qk = nn.Linear(self.entity_embed_dim, self.attn_embed_dim * 2)
        self.enemy_qk = nn.Linear(self.entity_embed_dim, self.attn_embed_dim * 2)
        # self.ally_qk = nn.Linear(self.entity_embed_dim, self.attn_embed_dim*2)
        self.enemy_ref_qk = nn.Linear(self.entity_embed_dim, self.attn_embed_dim * 2)
        self.norm = nn.Sequential(nn.LayerNorm(self.attn_embed_dim), nn.Tanh())

        self.enemy_hidden = nn.Parameter(
            th.zeros(1, 1, self.entity_embed_dim)
        ).requires_grad_(True)
        self.last_enemy_h = None
        if self.state_last_action:
            self.ally_dec_fc = nn.Linear(
                self.entity_embed_dim,
                state_nf_al + (self.n_actions_no_attack + 1) * 2,
            )
            self.enemy_dec_fc = nn.Linear(self.entity_embed_dim, state_nf_en + 1)
        else:
            self.ally_dec_fc = nn.Linear(
                self.entity_embed_dim, state_nf_al + (self.n_actions_no_attack + 1)
            )
            self.enemy_dec_fc = nn.Linear(self.entity_embed_dim, state_nf_en + 1)

    def global_process(self, states, task, actions=None):
        states = states.unsqueeze(1)

        task_decomposer = self.task2decomposer[task]
        task_n_agents = self.task2n_agents[task]
        last_action_shape = self.task2last_action_shape[task]

        bs = states.size(0)
        n_agents = task_decomposer.n_agents
        n_enemies = task_decomposer.n_enemies
        n_entities = n_agents + n_enemies

        # get decomposed state information
        ally_states, enemy_states, last_action_states, timestep_number_state = (
            task_decomposer.decompose_state(states)
        )
        ally_states = th.stack(ally_states, dim=0)  # [n_agents, bs, 1, state_nf_al]

        _, current_attack_action_info, current_compact_action_states = (
            task_decomposer.decompose_action_info(
                F.one_hot(
                    actions.reshape(-1), num_classes=self.task2last_action_shape[task]
                )
            )
        )
        current_compact_action_states = (
            current_compact_action_states.reshape(bs, n_agents, -1)
            .permute(1, 0, 2)
            .unsqueeze(2)
        )
        ally_states = th.cat([ally_states, current_compact_action_states], dim=-1)

        current_attack_action_info = current_attack_action_info.reshape(
            bs, n_agents, n_enemies
        ).sum(dim=1)
        attack_action_states = (
            (current_attack_action_info > 0)
            .type(ally_states.dtype)
            .reshape(bs, n_enemies, 1, 1)
            .permute(1, 0, 2, 3)
        )
        enemy_states = th.stack(enemy_states, dim=0)  # [n_enemies, bs, 1, state_nf_en]
        enemy_states = th.cat([enemy_states, attack_action_states], dim=-1)

        # stack action information
        if self.state_last_action:
            last_action_states = th.stack(last_action_states, dim=0)
            _, _, compact_action_states = task_decomposer.decompose_action_info(
                last_action_states
            )
            ally_states = th.cat([ally_states, compact_action_states], dim=-1)

        ally_states = ally_states.permute(1, 2, 0, 3).reshape(bs, n_agents, -1)
        enemy_states = enemy_states.permute(1, 2, 0, 3).reshape(bs, n_enemies, -1)
        return [ally_states, enemy_states]

    def attn_process(self, emb_inputs, emb_q, emb_k):
        # do attention
        bs, n, _ = emb_inputs.shape
        proj_query = emb_q
        proj_key = emb_k.permute(0, 2, 1)
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = emb_inputs.permute(0, 2, 1)
        # attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)[:, :n_agents, :]
        attn_out = th.bmm(proj_value, attn_score).permute(0, 2, 1)
        # .reshape(bs, n_entities, self.entity_embed_dim)[:, :n_agents, :]
        attn_out = attn_out.reshape(bs, n, self.entity_embed_dim)

        return attn_out

    def forward(self, emb_inputs, states, task, t=0, actions=None):
        # own_emb, enemy_emb, ally_emb = emb_inputs
        bs = emb_inputs.shape[0]
        task2decomposer = self.task2decomposer[task]
        n_agents = task2decomposer.n_agents
        n_enemies = task2decomposer.n_enemies

        if states is not None:
            ally_states, enemy_states = self.global_process(
                states, task, actions=actions
            )
        own_emb, enemy_emb = (
            emb_inputs[:, :, :, 0],
            emb_inputs[:, :, :, 1 : 1 + n_enemies],
        )
        if t == 0:
            self.last_enemy_h = self.enemy_hidden.repeat(bs, n_enemies, 1).unsqueeze(-2)

        own_emb = own_emb.reshape(bs, n_agents, self.entity_embed_dim)
        enemy_emb = (
            enemy_emb.reshape(bs, n_agents, n_enemies, self.entity_embed_dim)
            .permute(0, 2, 1, 3)
            .reshape(-1, n_agents, self.entity_embed_dim)
        )
        enemy_emb = th.cat(
            [self.last_enemy_h.reshape(-1, 1, self.entity_embed_dim), enemy_emb], dim=-2
        )
        # ally_emb = ally_emb.reshape(bs, n_agents, n_agents-1, self.entity_embed_dim)

        own_q, own_k = self.own_qk(own_emb).chunk(2, -1)
        enemy_q, enemy_k = self.enemy_qk(enemy_emb).chunk(2, -1)
        # ally_q, ally_k = self.ally_q(ally_emb).chunk(2, -1)

        enemy_ref = self.attn_process(enemy_emb, enemy_q, enemy_k)[:, 0].reshape(
            bs, n_enemies, self.entity_embed_dim
        )
        enemy_q, enemy_k = self.enemy_ref_qk(enemy_ref).chunk(2, -1)

        total_emb = th.cat([own_emb, enemy_ref], dim=-2)
        total_q = th.cat([own_q, enemy_q], dim=-2)
        total_k = th.cat([own_k, enemy_k], dim=-2)
        total_out = self.attn_process(total_emb, total_q, total_k)

        ally_out = total_out[:, :n_agents]
        enemy_out = total_out[:, -n_enemies:]
        self.last_enemy_h = enemy_out

        loss = 0.0
        if states is not None:
            al_dim, en_dim = ally_states.shape[-1], enemy_states.shape[-1]
            ally_out = self.ally_dec_fc(ally_out).reshape(-1, al_dim)
            enemy_out = self.enemy_dec_fc(enemy_out).reshape(-1, en_dim)
            # print(enemy_out.shape, enemy_states.shape, n_enemies)
            # enemy_states = enemy_states.unsqueeze(1).repeat(1, n_agents, 1, 1)

            loss = F.mse_loss(
                ally_out, ally_states.reshape(-1, al_dim).detach(), reduction="sum"
            ) + F.mse_loss(
                enemy_out, enemy_states.reshape(-1, en_dim).detach(), reduction="sum"
            )
            loss = loss / 2.0
        return total_out, loss


class Eval:
    def __init__(self):
        super(Eval, self).__init__()
        self.last_task = ""
        self.coordination = {}
        self.specific = {}

    def get_last_task(self):
        return self.last_task

    def get_data(self):
        return self.coordination, self.specific

    def reset_all(self):
        self.last_task = ""
        self.coordination = {}
        self.specific = {}

    def write_task(self, task):
        self.last_task = task

    def write_data(self, coordination, specific):
        self.coordination[self.last_task] = coordination
        self.specific[self.last_task] = specific
