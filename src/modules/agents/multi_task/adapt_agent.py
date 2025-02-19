import collections
import numpy as np
import torch as th
from torch.cuda import device_of
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
import h5py

from utils.embed import polynomial_embed, binary_embed
from utils.transformer import Transformer
from .vq_skill import SkillModule, MLPNet
from .net_utils import TruncatedNormal


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


class ADAPTAgent(nn.Module):
    """sotax agent for multi-task learning"""

    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        ):
        super(ADAPTAgent, self).__init__()
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
        # self.state_encoder = StateEncoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        # self.obs_encoder = ObsEncoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        # self.value = ValueNet(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.encoder = Encoder(args)
        # self.decoder = BasicDecoder(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        self.decoder = Decoder(
            task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        )
        self.planner = InferenceModel(
            task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        )
        # self.discr = Discriminator(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)

        self.last_out_h = None
        self.last_h_plan = None

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
        hidden_state_plan,
        task,
        mask=False,
        t=0,
        actions=None,
        ):
        h_plan = hidden_state_plan
        act, h_dec, cls_out = self.decoder(
            emb_inputs, inputs, hidden_state_dec, task, mask, actions
        )
        # _, out_h, h_plan = self.forward_planner(inputs, hidden_state_plan, t, task)
        # pre_own, pre_enemy, pre_ally = out_h
        # pre_state = th.cat([pre_own.unsqueeze(1), pre_enemy, pre_ally], dim=1)
        # act, h_dec = self.decoder(inputs, hidden_state_dec, pre_state, task, mask, actions)
        return act, h_dec, h_plan, cls_out

    def forward_value(self, inputs, hidden_state_value, task, actions=None):
        # hidden_state_value = hidden_state_value.reshape(-1, 1, self.args.entity_embed_dim)
        # attn_out, hidden_state_value = self.value(inputs, hidden_state_value, task)
        attn_out, hidden_state_value = self.value(inputs, hidden_state_value, task)
        return attn_out, hidden_state_value

    def forward_value_skill(self, inputs, hidden_state_value, task):
        # total_hidden = self.value.encode_for_skill(inputs, hidden_state_value, task)
        total_hidden = th.cat(
            [inputs, hidden_state_value.reshape(-1, 1, self.args.entity_embed_dim)],
            dim=1,
        )
        attn_out, hidden_state_value = self.value.predict(total_hidden)
        return attn_out, hidden_state_value

    def forward_planner(
        self,
        inputs,
        hidden_state_plan,
        t,
        task,
        test_mode=None,
        actions=None,
        next_inputs=None,
        loss_out=False,
        ):
        # h_plan = hidden_state_plan.reshape(-1, 1, self.args.entity_embed_dim)
        out_h, h, obs_loss = self.planner(
            inputs,
            hidden_state_plan,
            t,
            task,
            test_mode,
            next_inputs=next_inputs,
            actions=actions,
            loss_out=loss_out,
        )
        return out_h, h, obs_loss

    def forward_planner_feedforward(self, emb_inputs, forward_type="action"):
        out_h = self.planner.feedforward(emb_inputs, forward_type)
        return out_h

    def forward(
        self,
        inputs,
        hidden_state_plan,
        hidden_state_dec,
        t,
        task,
        skill=None,
        mask=False,
        actions=None,
        local_obs=None,
        test_mode=None,
        ):
        if t % self.c == 0:
            out_h, h_plan, _ = self.forward_planner(
                inputs, hidden_state_plan, t, task, test_mode
            )
            out_h = self.forward_planner_feedforward(out_h)
            self.last_out_h, self.last_h_plan = out_h, h_plan

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

        act, h_dec, _ = self.decoder(
            self.last_out_h, inputs, hidden_state_dec, task, mask, actions
        )
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
            out_h = th.cat(self.last_out_h, dim=1)
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

        return act, self.last_h_plan, h_dec, skill


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

    def forward(self, states, hidden_state, task, actions=None):
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

        # do inference and get entity_embed
        ally_embed = self.ally_encoder(ally_states)
        enemy_embed = self.enemy_encoder(enemy_states)

        # we ought to do self-attention
        entity_embed = th.cat([ally_embed, enemy_embed], dim=0)

        # do attention
        proj_query = (
            self.query(entity_embed)
            .permute(1, 2, 0, 3)
            .reshape(bs, n_entities, self.attn_embed_dim)
        )
        proj_key = (
            self.key(entity_embed)
            .permute(1, 2, 3, 0)
            .reshape(bs, self.attn_embed_dim, n_entities)
        )
        energy = th.bmm(proj_query / (self.attn_embed_dim ** (1 / 2)), proj_key)
        attn_score = F.softmax(energy, dim=1)
        proj_value = entity_embed.permute(1, 2, 3, 0).reshape(
            bs, self.entity_embed_dim, n_entities
        )
        # attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)[:, :n_agents, :]
        attn_out = th.bmm(proj_value, attn_score).squeeze(1).permute(0, 2, 1)
        # .reshape(bs, n_entities, self.entity_embed_dim)[:, :n_agents, :]

        attn_out = attn_out[:, :n_agents].reshape(bs, n_agents, self.entity_embed_dim)
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


class ObsEncoder(nn.Module):
    """sotax agent for multi-task learning"""

    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        ):
        super(ObsEncoder, self).__init__()
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

        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )

    def forward(self):
        return


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
        # self.reward_fc = nn.Linear(self.entity_embed_dim, 1)
        self.reward_fc = nn.Sequential(
            nn.Linear(self.entity_embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.entity_embed_dim).zero_()

    def encode(self, inputs, hidden_state, task):
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
        h = outputs[:, -1:, :]
        reward = outputs[:, 0, :]
        reward = self.reward_fc(reward)
        return reward, h

    def forward(self, inputs, hidden_state, task):
        total_hidden = self.encode(inputs, hidden_state, task)
        reward, h = self.predict(total_hidden)
        return reward, h


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args

        self.skill_dim = args.skill_dim
        self.entity_embed_dim = args.entity_embed_dim

        self.q_skill = nn.Linear(self.entity_embed_dim, self.skill_dim)

    def forward(self, attn_out):
        skill = self.q_skill(attn_out)
        return skill


class BasicDecoder(nn.Module):

    def __init__(
        self, task2input_shape_info, task2decomposer, task2n_agents, decomposer, args
        ):
        super(BasicDecoder, self).__init__()
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
        # self.task_repre_dim = args.task_repre_dim
        ## get obs shape information
        obs_own_dim = decomposer.own_obs_dim
        obs_en_dim, obs_al_dim = decomposer.obs_nf_en, decomposer.obs_nf_al
        n_actions_no_attack = decomposer.n_actions_no_attack
        ## get wrapped obs_own_dim
        wrapped_obs_own_dim = obs_own_dim + args.id_length + n_actions_no_attack + 1
        ## enemy_obs ought to add attack_action_info
        obs_en_dim += 1

        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )
        self.base_q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)
        self.ally_q_skill = nn.Linear(self.entity_embed_dim, 1)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def forward(self, emb_inputs, inputs, hidden_state, task, mask=False, actions=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        n_agents = self.task2n_agents[task]
        bs, n_agents, n_entity, _ = inputs.shape
        n_enemy = n_entity - n_agents

        total_hidden = emb_inputs.reshape(bs * n_agents, -1, self.entity_embed_dim)
        outputs = self.transformer(total_hidden, None)

        h = outputs[:, -1, :]
        base_action_inputs = outputs[:, 0]

        q_base = self.base_q_skill(base_action_inputs)
        attack_action_inputs = outputs[:, 1 : n_enemy + 1]
        q_attack = self.ally_q_skill(attack_action_inputs)
        q = th.cat([q_base, q_attack.reshape(-1, n_enemy)], dim=-1)

        return q, h


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

        # self.q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)
        # self.skill_enc = nn.Linear(self.skill_dim, self.entity_embed_dim)
        # self.q_skill = nn.Linear(self.entity_embed_dim + self.entity_embed_dim, n_actions_no_attack)
        # self.q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)
        self.q_skill = nn.Linear(self.entity_embed_dim * 2, n_actions_no_attack)
        # self.base_q_skill = nn.Linear(self.entity_embed_dim, n_actions_no_attack)
        # self.ally_q_skill = nn.Linear(self.entity_embed_dim, 1)
        self.base_q_skill = MLPNet(
            2 * self.entity_embed_dim, n_actions_no_attack, 128, output_norm=False
        )
        self.ally_q_skill = MLPNet(2 * self.entity_embed_dim, 1, 128, output_norm=False)

        self.n_actions_no_attack = n_actions_no_attack
        # self.skill_hidden = nn.Parameter(th.zeros(1, 1, self.entity_embed_dim))
        # self.pre_own = nn.Linear(self.entity_embed_dim, self.entity_embed_dim)
        # self.pre_enemy = nn.Linear(self.entity_embed_dim, self.entity_embed_dim)
        # self.pre_ally = nn.Linear(self.entity_embed_dim, self.entity_embed_dim)
        self.cls_hidden = nn.Parameter(th.zeros(1, 1, self.entity_embed_dim))
        self.cls_fc = nn.Linear(self.entity_embed_dim, self.cls_dim)
        # self.cross_attn = CrossAttention(task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)

    def init_hidden(self):
        # make hidden states on the same device as model
        return self.q_skill.weight.new(1, self.args.entity_embed_dim).zero_()

    def forward(self, emb_inputs, inputs, hidden_state, task, mask=False, actions=None):
        hidden_state = hidden_state.reshape(-1, 1, self.entity_embed_dim)
        # cls_hidden = self.cls_hidden.repeat(hidden_state.shape[0], 1, 1)

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
        n_entity = n_enemy + n_ally + 1

        # random mask
        if mask and actions is not None:
            actions = actions.reshape(-1)

            b, n, _ = enemy_feats.shape
            mask = th.randint(0, 2, (b, n, 1)).to(enemy_feats.device)
            for i in range(actions.shape[0]):
                if actions[i] > self.n_actions_no_attack - 1:
                    mask[i, actions[i] - self.n_actions_no_attack] = 1
            enemy_feats = enemy_feats * mask

            b, n, _ = ally_feats.shape
            mask = th.randint(0, 2, (b, n, 1)).to(ally_feats.device)
            ally_feats = ally_feats * mask

        # compute key, query and value for attention
        own_hidden = self.own_value(own_obs).unsqueeze(1)
        ally_hidden = self.ally_value(ally_feats)
        enemy_hidden = self.enemy_value(enemy_feats)
        # skill_hidden = self.skill_value(skill).unsqueeze(1)
        # history_hidden = hidden_state
        # emb_hidden = self.ln(self.agent_feedforward(emb_inputs))
        # own_emb_inputs, enemy_emb_inputs, ally_emb_inputs = emb_inputs
        emb_inputs = th.cat(emb_inputs, dim=1)
        # own_emb_inputs = self.agent_feedforward(own_emb_inputs)
        # enemy_emb_inputs = self.enemy_feedforward(enemy_emb_inputs)
        # ally_emb_inputs = self.ally_feedforward(ally_emb_inputs)
        # emb_hidden = th.cat([own_emb_inputs, enemy_emb_inputs, ally_emb_inputs], dim=1)
        # own_hidden, enemy_hidden, ally_hidden = (
        #     th.cat([own_emb_inputs, own_hidden], dim=-1),
        #     th.cat([enemy_emb_inputs, enemy_hidden], dim=-1),
        #     th.cat([ally_emb_inputs, ally_hidden], dim=-1),
        # )

        # own_hidden, enemy_hidden, ally_hidden = (
        #     self.own_merge(own_hidden),
        #     self.enemy_merge(enemy_hidden),
        #     self.ally_merge(ally_hidden),
        # )

        # pre_own_h = self.pre_own(pre_state[:, 0]).unsqueeze(1)
        # pre_enemy_h = self.pre_enemy(pre_state[:, 1:1+n_enemy])
        # pre_ally_h = self.pre_ally(pre_state[:, 2+n_enemy:2+n_enemy+n_ally])

        # total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=1)
        # total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden, pre_own_h, pre_enemy_h, pre_ally_h, history_hidden], dim=1)

        total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden], dim=-2)

        outputs = self.transformer(total_hidden, None)
        # h = outputs[:, -1, :]
        # outputs = self.cross_attn(outputs[:, :-1], emb_inputs, task)
        outputs = th.cat([outputs, emb_inputs], dim=-1)
        outputs = outputs[:, :n_entity]

        # cls_out = outputs[:, -2]
        # cls_out = cls_hidden
        cls_out = self.cls_fc(th.zeros_like(hidden_state).detach())
        # outputs = outputs[:, :-2]
        # skill_hidden = discr_h.reshape(-1, 1, self.entity_embed_dim).repeat(1, outputs.shape[1], 1)
        # outputs = th.cat([outputs, skill_hidden], dim=-1)
        # skill_ = outputs[:, -2, :]
        # skill_ = history_skill
        # skill_emb = self.skill_enc(skill)
        # base_action_inputs = th.cat([outputs[:, 0, :], skill_emb], dim=-1)
        base_action_inputs = outputs[:, 0, :]
        # q_base = self.q_skill(base_action_inputs)

        # if t % 5 == 0 or q_skill is None:
        # agent_skill_inputs = th.cat([base_action_inputs, skill_], dim=-1)
        # q_skill = self.agent_q_skill(agent_skill_inputs)
        # q_skill = self.agent_q_skill(skill_)
        # q_skill = F.softmax(q_skill, dim=-1)
        # skill_id = th.max(q_skill, dim=-1, keepdim=True)[1]
        # agent_skill_inputs = th.cat([own_obs, skill_hidden.squeeze(1), hidden_state.squeeze(1).detach()], dim=-1)
        # agent_skill_inputs = th.cat([base_action_inputs, outputs[:, -1, :].detach()], dim=-1)
        # skill_ = self.agent_skill_hidden(agent_skill_inputs)
        # q_skill = th.rand(h.shape[0], self.skill_dim)

        # q_attack_list = []
        # skill_emb = self.skill_dict(skill_id).squeeze(1)
        # skill_emb = self.skill_encoder(q_skill).unsqueeze(1).repeat(1, enemy_feats.size(0), 1)

        # q_base = self.base_q_skill(th.cat([base_action_inputs, skill_emb[:, 0]], dim=-1))
        q_base = self.base_q_skill(base_action_inputs)
        # base_action_inputs = base_action_inputs.unsqueeze(1).repeat(1, enemy_feats.size(1), 1)
        # skill_emb = skill_emb.unsqueeze(1).repeat(1, enemy_feats.size(1), 1)
        attack_action_inputs = outputs[:, 1 : 1 + n_enemy]
        # attack_action_inputs = th.cat(
        # [base_action_inputs, outputs[:, 1: 1+enemy_feats.size(1), :], skill_emb], dim=-1)
        # [base_action_inputs, outputs[:, 1: 1+enemy_feats.size(1), :]], dim=-1)
        # [outputs[:, 1: 1+enemy_feats.size(1), :], skill_emb[:, 1:]], dim=-1)
        # [outputs[:, 1: 1+enemy_feats.size(1), :]], dim=-1)
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

        return q, hidden_state, cls_out


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
        self.belif_dim = self.entity_embed_dim
        if self.args.belif_type == "rec":
            self.belif_dim = 2 * self.entity_embed_dim

        # Encoder
        self.ally_value = nn.Linear(obs_al_dim, self.entity_embed_dim)
        self.enemy_value = nn.Linear(obs_en_dim, self.entity_embed_dim)
        self.own_value = nn.Linear(wrapped_obs_own_dim, self.entity_embed_dim)
        self.value_vale = nn.Linear(1, self.entity_embed_dim)
        self.transformer = Transformer(
            self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )
        # self.seq_wrapper = SequenceObservationWrapper(
        #     task2input_shape_info, task2decomposer, task2n_agents, decomposer, args)
        # self.obs_decoder = Transformer(self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim)

        # # Variational
        self.ally_var = nn.Linear(self.entity_embed_dim, self.belif_dim)
        self.enemy_var = nn.Linear(self.entity_embed_dim, self.belif_dim)
        self.own_var = nn.Linear(self.entity_embed_dim, self.belif_dim)

        # Decoder
        self.dec_transformer = Transformer(
            2 * self.entity_embed_dim, args.head, args.depth, self.entity_embed_dim
        )
        self.ally_dec = nn.Linear(self.entity_embed_dim, obs_al_dim)
        self.enemy_dec = nn.Linear(self.entity_embed_dim, obs_en_dim)
        self.own_dec = nn.Linear(self.entity_embed_dim, wrapped_obs_own_dim)

        # Processer
        self.own_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.enemy_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
        self.ally_forward = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)

        if self.args.belif_type == "moco":
            self.own_moco = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
            self.enemy_moco = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)
            self.ally_moco = MLPNet(self.entity_embed_dim, self.entity_embed_dim, 128)

        self.base_q_skill = nn.Linear(self.entity_embed_dim * 2, n_actions_no_attack)
        self.ally_q_skill = nn.Linear(self.entity_embed_dim * 2, 1)

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

    def reparameterization(self, inputs, test_mode=False, component="all"):
        if component == "own":
            inputs = self.own_var(inputs)
        elif component == "enemy":
            inputs = self.enemy_var(inputs)
        elif component == "ally":
            inputs = self.ally_var(inputs)

        mu, log_var = inputs.chunk(2, dim=-1)
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return eps * std + mu, mu, log_var

    def feedforward(self, inputs, detach=True):
        if detach:
            own_emb, enemy_emb, ally_emb = (
                inputs[0].detach(),
                inputs[1].detach(),
                inputs[2].detach(),
            )
        else:
            own_emb, enemy_emb, ally_emb = inputs
        # n_enemy, n_ally = enemy_emb.shape[1], ally_emb.shape[1]
        own_out = self.own_forward(own_emb)
        enemy_out = self.enemy_forward(enemy_emb)
        ally_out = self.ally_forward(ally_emb)

        return [own_out, enemy_out, ally_out]

    def moco_forward(self, inputs):
        own_emb, enemy_emb, ally_emb = inputs
        own_out = self.own_moco(own_emb)
        enemy_out = self.enemy_moco(enemy_emb)
        ally_out = self.ally_moco(ally_emb)

        return [own_out, enemy_out, ally_out]

    def contrastive_conpute(self, q, k):
        return

    def forward(
        self,
        inputs,
        hidden_state,
        t,
        task,
        test_mode=False,
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

        # value = th.zeros(own_obs.shape[0], 1)
        # if t == 0:
        #     obs_dict = self.seq_wrapper.obs_reset(own_obs, enemy_feats, ally_feats, value)
        # else:
        #     obs_dict = self.seq_wrapper.obs_step(own_obs, enemy_feats, ally_feats, value)
        #
        # own_stack, enemy_stack, ally_stack, value_stack, mask_stack = obs_dict['own'], \
        # obs_dict['enemy'], obs_dict['ally'], obs_dict['value'], obs_dict['mask']
        own_stack, enemy_stack, ally_stack = (
            own_obs.unsqueeze(1).unsqueeze(1),
            enemy_feats.unsqueeze(1),
            ally_feats.unsqueeze(1),
        )

        # compute key, query and value for attention
        own_hidden = self.own_value(own_stack)
        ally_hidden = self.ally_value(ally_stack)
        enemy_hidden = self.enemy_value(enemy_stack)
        # value_hidden = self.value_value(value_stack)
        history_hidden = hidden_state.unsqueeze(1)

        # value = self.rew_enc(value)
        b = own_hidden.shape[0]
        # mask_stack = mask_stack.reshape(1, -1, 1, 1).repeat(b, 1, n_enemy+n_ally+2, self.entity_embed_dim)
        total_hidden = th.cat(
            [own_hidden, enemy_hidden, ally_hidden, history_hidden], dim=2
        )
        # total_hidden *= mask_stack
        total_hidden = total_hidden.reshape(b, -1, self.entity_embed_dim)

        outputs = self.transformer(total_hidden, None).reshape(
            b, self.args.num_stack_frames, -1, self.entity_embed_dim
        )
        h = outputs[:, -1, -1]
        # h = outputs[:, -1, -1].chunk(2, dim=-1)[0]
        outputs = outputs[:, :, :-1]
        # outputs_dis, outputs = self.reparameterization(outputs)
        # h = hidden_state.reshape(-1, self.entity_embed_dim)

        # cls_out_h = outputs[:, -1]
        # cls_out = self.cls_fc(cls_out_h)

        # own_out = own_out_h.reshape(bs, task_n_agents, -1, self.entity_embed_dim)
        # enemy_out = enemy_out_h.reshape(bs, task_n_agents, -1, self.entity_embed_dim)
        # ally_out = ally_out_h.reshape(bs, task_n_agents, -1, self.entity_embed_dim)

        own_out_h = outputs[:, -1, 0].unsqueeze(1)
        enemy_out_h = outputs[:, -1, 1 : 1 + n_enemy]
        ally_out_h = outputs[:, -1, -n_ally:]

        own_out_h, own_out_mu, own_out_logvar = self.reparameterization(
            own_out_h, test_mode, component="own"
        )
        enemy_out_h, enemy_out_mu, enemy_out_logvar = self.reparameterization(
            enemy_out_h, test_mode, component="enemy"
        )
        ally_out_h, ally_out_mu, ally_out_logvar = self.reparameterization(
            ally_out_h, test_mode, component="ally"
        )

        # own_out = self.own_fc(own_out_h).reshape(bs * task_n_agents, -1)
        # enemy_out = self.enemy_fc(enemy_out_h).reshape(bs * task_n_agents, n_enemy, -1)
        # ally_out = self.ally_fc(ally_out_h).reshape(bs * task_n_agents, n_ally, -1)

        # own_out = self.ln(own_out)
        # enemy_out = self.ln(enemy_out)
        # ally_out = self.ln(ally_out)

        out_loss = th.tensor(0.0).to(inputs.device)
        if loss_out and next_inputs is not None:
            if self.args.belif_type == "rec":
                # if t == 0:
                #     self.last_prior = [
                #         th.rand_like(own_out_h) * 2.0 - 1.0,
                #         th.rand_like(enemy_out_h) * 2.0 - 1.0,
                #         th.rand_like(ally_out_h) * 2.0 - 1.0,
                #     ]
                # kl_loss = kl_divergence(outputs_dis, self.last_prior).mean()
                # self.last_prior = outputs_dis
                kl_loss = th.mean(
                    th.mean(
                        -0.5
                            * th.sum(
                                1 + own_out_logvar - own_out_mu**2 - own_out_logvar.exp(),
                                dim=1,
                            ),
                        dim=0,
                    )
                        + th.mean(
                            -0.5
                                * th.sum(
                                    1
                                        + enemy_out_logvar
                                        - enemy_out_mu**2
                                        - enemy_out_logvar.exp(),
                                    dim=1,
                                ),
                            dim=0,
                        )
                        + th.mean(
                            -0.5
                                * th.sum(
                                    1
                                        + ally_out_logvar
                                        - ally_out_mu**2
                                        - ally_out_logvar.exp(),
                                    dim=1,
                                ),
                            dim=0,
                        )
                )
                # self.last_prior = [own_out_dis, enemy_out_dis, ally_out_dis]
                out_loss += self.beta * kl_loss

                next_obs_inputs, last_action_inputs, agent_id_inputs = (
                    next_inputs[:, :obs_dim],
                    next_inputs[:, obs_dim : obs_dim + last_action_shape],
                    next_inputs[:, obs_dim + last_action_shape :],
                )
                next_own_obs, next_enemy_feats, next_ally_feats = (
                    task_decomposer.decompose_obs(next_obs_inputs)
                )  # own_obs: [bs*self.n_agents, own_obs_dim]
                bs = int(next_own_obs.shape[0] / task_n_agents)

                # embed agent_id inputs and decompose last_action_inputs
                agent_id_inputs = [
                    th.as_tensor(
                        binary_embed(i + 1, self.args.id_length, self.args.max_agent),
                        dtype=next_own_obs.dtype,
                    )
                    for i in range(task_n_agents)
                ]
                agent_id_inputs = (
                    th.stack(agent_id_inputs, dim=0)
                    .repeat(bs, 1)
                    .to(next_own_obs.device)
                )
                _, attack_action_info, compact_action_states = (
                    task_decomposer.decompose_action_info(last_action_inputs)
                )

                # incorporate agent_id embed and compact_action_states
                next_own_obs = th.cat(
                    [next_own_obs, agent_id_inputs, compact_action_states], dim=-1
                )

                # incorporate attack_action_info into enemy_feats
                attack_action_info = attack_action_info.transpose(0, 1).unsqueeze(-1)
                next_enemy_feats = th.cat(
                    [th.stack(next_enemy_feats, dim=0), attack_action_info], dim=-1
                )
                next_ally_feats = th.stack(next_ally_feats, dim=0)

                next_enemy_feats = next_enemy_feats.permute(1, 0, 2)
                next_ally_feats = next_ally_feats.permute(1, 0, 2)

                # compute key, query and value for attention
                own_hidden = self.own_value(own_obs).unsqueeze(1)
                ally_hidden = self.ally_value(ally_feats)
                enemy_hidden = self.enemy_value(enemy_feats)

                own_hidden = th.cat([own_out_h, own_hidden], dim=-1)
                ally_hidden = th.cat([ally_out_h, ally_hidden], dim=-1)
                enemy_hidden = th.cat([enemy_out_h, enemy_hidden], dim=-1)

                total_hidden = th.cat([own_hidden, enemy_hidden, ally_hidden], dim=-2)
                outputs = self.dec_transformer(total_hidden, None)

                own_output = outputs[:, 0].squeeze(1)
                enemy_output = outputs[:, 1 : 1 + n_enemy]
                ally_output = outputs[:, -n_ally:]

                own_pre = self.own_dec(own_output).squeeze(1)
                enemy_pre = self.enemy_dec(enemy_output)
                ally_pre = self.ally_dec(ally_output)

                out_loss += th.mean(
                    F.mse_loss(own_pre, next_own_obs.detach())
                        + F.mse_loss(enemy_pre, next_enemy_feats.detach())
                        + F.mse_loss(ally_pre, next_ally_feats.detach())
                )

            elif self.args.belif_type == "moco":
                out_loss += self.contrastive_compute(own_obs, enemy_feats, ally_feats)

        return [own_out_h, enemy_out_h, ally_out_h], h, out_loss


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
        skill_emb = th.cat(skill_emb, dim=1)
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
            self.ally_dec_fc = MLPNet(
                self.entity_embed_dim,
                state_nf_al + (self.n_actions_no_attack + 1) * 2,
                128,
            )
            self.enemy_dec_fc = MLPNet(self.entity_embed_dim, state_nf_en + 1, 128)
        else:
            self.ally_dec_fc = MLPNet(
                self.entity_embed_dim, state_nf_al + (self.n_actions_no_attack + 1), 128
            )
            self.enemy_dec_fc = MLPNet(self.entity_embed_dim, state_nf_en + 1, 128)

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
        own_emb, enemy_emb, ally_emb = emb_inputs
        ally_states, enemy_states = self.global_process(states, task, actions=actions)
        bs, n_agents, n_enemies = (
            ally_states.shape[0],
            ally_states.shape[1],
            enemy_states.shape[1],
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

        al_dim, en_dim = ally_states.shape[-1], enemy_states.shape[-1]
        ally_out = self.ally_dec_fc(ally_out).reshape(-1, al_dim)
        enemy_out = self.enemy_dec_fc(enemy_out).reshape(-1, en_dim)
        # print(enemy_out.shape, enemy_states.shape, n_enemies)
        # enemy_states = enemy_states.unsqueeze(1).repeat(1, n_agents, 1, 1)

        loss = F.mse_loss(
            ally_out, ally_states.reshape(-1, al_dim).detach()
        ) + F.mse_loss(enemy_out, enemy_states.reshape(-1, en_dim).detach())
        return loss
