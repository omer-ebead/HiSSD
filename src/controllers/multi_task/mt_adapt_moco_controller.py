from re import L
from collections import deque
from modules.agents.multi_task import REGISTRY as agent_REGISTRY
from modules.decomposers import REGISTRY as decomposer_REGISTRY

from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import torch.distributions as D
import numpy as np
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


# This multi-agent controller shares parameters between agents
class ADAPTMOCOSMAC:
    def __init__(self, train_tasks, task2scheme, task2args, main_args):
        # set some task-specific attributes
        self.train_tasks = train_tasks
        self.task2scheme = task2scheme
        self.task2args = task2args
        self.task2n_agents = {
            task: self.task2args[task].n_agents for task in train_tasks
        }
        self.main_args = main_args

        # set some common attributes
        self.agent_output_type = main_args.agent_output_type
        self.action_selector = action_REGISTRY[main_args.action_selector](main_args)

        # get decomposer for each task
        env2decomposer = {
            "sc2": "sc2_decomposer",
        }
        self.task2decomposer, self.task2dynamic_decoder = {}, {}
        self.surrogate_decomposer = None
        for task in train_tasks:
            task_args = self.task2args[task]
            if task_args.env == "sc2":
                task_decomposer = decomposer_REGISTRY[env2decomposer[task_args.env]](
                    task_args
                )
                self.task2decomposer[task] = task_decomposer
                if not self.surrogate_decomposer:
                    self.surrogate_decomposer = task_decomposer
            else:
                raise NotImplementedError(f"Unsupported env decomposer {task_args.env}")
            # set obs_shape
            task_args.obs_shape = task_decomposer.obs_dim

        # build agents
        task2input_shape_info = self._get_input_shape()
        self._build_agents(task2input_shape_info)

        self.skill_dim = main_args.skill_dim
        self.c_step = main_args.c_step
        self.init_params()

    def init_params(self):
        self.hidden_states_value = None
        self.hidden_states_enc = None
        self.hidden_states_dec = None
        self.hidden_states_plan = None
        self.last_prior = None
        self.skill_hidden = None
        self.q_skill = None
        self.skill = None
        self.cls_dim = 3
        self.last_out_h = None
        self.last_obs_loss = None

    def select_actions(
        self, ep_batch, t_ep, t_env, task, bs=slice(None), test_mode=False
    ):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, task, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward_global_hidden(self, ep_batch, t, task, actions=None, test_mode=False):
        agent_inputs = ep_batch["state"][:, t]
        agent_outs, self.hidden_states_enc = self.agent.forward_global_hidden(
            agent_inputs, self.hidden_states_enc, task, t=t, actions=actions
        )
        return agent_outs.reshape(
            ep_batch.batch_size, -1, self.main_args.entity_embed_dim
        )
        # return agent_outs.reshape(ep_batch.batch_size, self.task2n_agents[task], -1, self.main_args.entity_embed_dim)

    def forward_global_action(self, ep_batch, obs_emb, t, task, test_model=False):
        bs = ep_batch.batch_size
        agent_inputs = self._build_inputs(ep_batch, t, task)

        act_out, self.hidden_states_dec = self.agent.forward_action(
            agent_inputs,
            obs_emb,
            self.hidden_states_dec,
            task,
            t=t,
        )
        return act_out.reshape(bs, self.task2n_agents[task], -1)

    def forward_value(
        self, ep_batch, t, task, test_mode=False, actions=None, emb_inputs=None
    ):
        bs = ep_batch.batch_size
        agent_inputs = self._build_inputs(ep_batch, t, task)
        # batch_emb = batch_emb.reshape(bs * self.task2n_agents[task], -1, self.main_args.entity_embed_dim)
        agent_outs, agent_outs_hidden, self.hidden_states_value = (
            self.agent.forward_value(
                agent_inputs,
                self.hidden_states_value,
                task,
                actions=actions,
                emb_inputs=emb_inputs,
            )
        )

        return agent_outs.reshape(
            bs, self.task2n_agents[task], 1
        ), agent_outs_hidden.reshape(bs, self.task2n_agents[task], -1)

    def forward_encoder(self, ep_batch, t, task, test_mode=False, actions=None):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        agent_outs, self.hidden_states_enc = self.agent.forward_encoder(
            agent_inputs, self.hidden_states_enc, task, t=t, actions=actions
        )
        return agent_outs.reshape(
            ep_batch.batch_size,
            # self.task2n_agents[task],
            -1,
            self.main_args.entity_embed_dim,
        )

    def forward_global_decoder(self, emb_inputs, t, task, test_mode=False):
        ally_outs, enemy_outs, self.hidden_states_global_dec = (
            self.agent.forward_global_decoder(
                emb_inputs, self.hidden_states_global_dec, task=task
            )
        )
        return ally_outs, enemy_outs

    def global_state_process(self, ep_batch, t, task, test_mode=False, actions=None):
        agent_inputs = ep_batch["state"][:, t]
        ally_outs, enemy_outs = self.agent.global_state_process(
            agent_inputs, task=task, actions=actions
        )
        return ally_outs, enemy_outs

    # def forward_seq_action(self, ep_batch, skill_action, t, task, test_model=False):
    def forward_seq_action(self, ep_batch, t, task, mask=False, test_model=False):
        agent_seq_inputs = []
        # if t == 0:
        #     self.q_skill = None
        # skill_action = skill_action.reshape(-1, self.skill_dim)
        for i in range(self.c_step):
            agent_inputs = self._build_inputs(ep_batch, t + i, task)
            agent_seq_inputs.append(agent_inputs)
        agent_seq_inputs = th.stack(agent_seq_inputs, dim=1)
        actions = ep_batch["actions"][:, t : t + self.c_step]

        agent_seq_outs, self.hidden_states_dec, self.hidden_states_plan = (
            self.agent.forward_seq_action(
                agent_seq_inputs,
                self.hidden_states_dec,
                self.hidden_states_plan,
                task,
                mask,
                t,
                actions,
            )
        )

        return agent_seq_outs.view(
            ep_batch.batch_size, self.c_step, self.task2n_agents[task], -1
        )
        # obs_seq_outs.view(ep_batch.batch_size, self.c_step, self.task2n_agents[task], -1, self.main_args.entity_embed_dim)

    def forward_latent_planner(
        self,
        emb_inputs,
        t,
        task,
        actions=None,
        end_t=0,
        rewards=None,
        test_mode=True,
    ):
        bs = emb_inputs.shape[0]
        # if t == 0:
        #     self.reset_planner(task, emb_inputs.device, emb_inputs.shape[0])
        out_emb, self.hidden_states_plan = self.agent.forward_latent_planner(
            emb_inputs,
            task,
            actions=actions,
            rewards=rewards,
            hidden_state=self.hidden_states_plan,
            test_mode=test_mode,
        )
        return out_emb.reshape(
            bs,
            self.main_args.context_len,
            self.task2n_agents[task],
            -1,
            self.main_args.entity_embed_dim,
        )

    def forward_reconstruction(self, ep_batch, emb_inputs, t, task, actions=None):
        agent_inputs = ep_batch["state"][:, t + 1]
        emb_out, loss = self.agent.forward_reconstruction(
            emb_inputs, agent_inputs, t=t, task=task, actions=actions
        )
        return (
            emb_out.reshape(ep_batch.batch_size, -1, self.main_args.entity_embed_dim),
            loss,
        )

    def forward_planner(
        self,
        ep_batch,
        t,
        task,
        actions=None,
        training=False,
        loss_out=False,
    ):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        next_inputs = None
        if training:
            # next_inputs = ep_batch["state"][:, t + self.c_step]
            next_inputs = self._build_inputs(ep_batch, t + self.c_step, task)
        out_h, self.hidden_states_plan = self.agent.forward_planner(
            agent_inputs,
            self.hidden_states_plan,
            t,
            task,
            actions=actions,
            next_inputs=next_inputs,
            loss_out=loss_out,
        )
        self.last_out_h = out_h
        # agent_inputs = ep_batch["state"][:, t]
        # out_h, self.hidden_states_plan = self.agent.forward_global_hidden(agent_inputs, task, self.hidden_states_plan, actions=actions)
        # out_loss, out_h, self.hidden_states_plan = self.agent.forward_planner(agent_inputs, self.hidden_states_plan, t, task)

        # out_h[0] = out_h[0].unsqueeze(-2)
        # out_h[0], out_h[1], out_h[2] = out_h[0].reshape(ep_batch.batch_size, self.task2n_agents[task], 1, self.main_args.entity_embed_dim), \
        # out_h[1].reshape(ep_batch.batch_size, self.task2n_agents[task], -1, self.main_args.entity_embed_dim), \
        # out_h[2].reshape(ep_batch.batch_size, self.task2n_agents[task], -1, self.main_args.entity_embed_dim)

        return self.last_out_h
        # return own.reshape(ep_batch.batch_size, self.task2n_agents[task], 1, self.main_args.skill_dim), \
        # enemy.reshape(ep_batch.batch_size, self.task2n_agents[task], -1, self.main_args.skill_dim), \
        # ally.reshape(ep_batch.batch_size, self.task2n_agents[task], -1, self.main_args.skill_dim)

    def forward_skill_feedforward(self, emb_inputs):
        out_h = self.agent.forward_skill_feedforward(emb_inputs)
        return out_h

    def forward_skill_value(self, emb_inputs):
        out_h = self.agent.forward_skill_value(emb_inputs)
        return out_h

    def forward_discriminator(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        ssl_out, ssl_out_h, self.hidden_states_dis = self.agent.forward_discriminator(
            agent_inputs, t, task, self.hidden_states_dis
        )
        return ssl_out.reshape(
            ep_batch.batch_size, self.task2n_agents[task], -1
        ), ssl_out_h.reshape(ep_batch.batch_size, self.task2n_agents[task], -1)

    def forward_contrastive(self, emb_inputs, emb_pos):
        logits = self.agent.forward_contrastive(emb_inputs, emb_pos)
        return logits

    def reset_planner(self, task, device, bs):
        self.agent.reset_planner(task, device, bs)

    def forward(self, ep_batch, t, task, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t, task)
        avail_actions = ep_batch["avail_actions"][:, t]
        actions = ep_batch["actions"][:, t]

        bs = agent_inputs.shape[0] // self.task2n_agents[task]
        # task_repre = self.get_task_repres(task, require_grad=False)
        # task_repre = task_repre.repeat(bs, 1)
        if t == 0:
            self.last_action = th.zeros(actions.shape).long().to(actions.device)

        if t % self.c_step == 0:
            (
                agent_outs,
                self.hidden_states_enc,
                self.hidden_states_dec,
                self.hidden_states_plan,
                self.skill,
            ) = self.agent(
                agent_inputs,
                self.hidden_states_enc,
                self.hidden_states_dec,
                self.hidden_states_plan,
                t,
                task,
                None,
                actions=None,
                local_obs=None,
                test_mode=test_mode,
            )
            # local_obs=local_agent_inputs)
        else:
            (
                agent_outs,
                self.hidden_states_plan,
                self.hidden_states_dec,
                _,
            ) = self.agent(
                agent_inputs,
                self.hidden_states_plan,
                self.hidden_states_dec,
                t,
                task,
                self.skill,
                actions=None,
                local_obs=None,
                test_mode=test_mode,
            )
            # local_obs=local_agent_inputs)

        # agent_outs, self.hidden_states_enc, self.hidden_states_dec, self.q_skill, self.skill_hidden, _ = self.agent(
        #     agent_inputs, self.hidden_states_enc, self.hidden_states_dec, task, self.skill_hidden, self.q_skill, t)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.main_args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.task2n_agents[task], -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if not test_mode and self.adaptation:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.main_args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(
                        dim=1, keepdim=True
                    ).float()

                agent_outs = (
                    1 - self.action_selector.epsilon
                ) * agent_outs + th.ones_like(
                    agent_outs
                ) * self.action_selector.epsilon / epsilon_action_num

                if getattr(self.main_args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        self.last_action = agent_outs.max(dim=-1)[1].reshape(actions.shape).long()

        return agent_outs.view(ep_batch.batch_size, self.task2n_agents[task], -1)

    def init_hidden(self, batch_size, task):
        # we always know we are in which task when do init_hidden
        n_agents = self.task2n_agents[task]
        n_enemies = self.task2decomposer[task].n_enemies
        n_entities = n_agents + n_enemies

        (
            hidden_states_value,
            hidden_states_dec,
            hidden_states_plan,
            hidden_states_enc,
        ) = self.agent.init_hidden()

        self.hidden_states_value = hidden_states_value.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_dec = hidden_states_dec.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_plan = hidden_states_plan.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        self.hidden_states_enc = hidden_states_enc.unsqueeze(0).expand(
            batch_size, n_agents, -1
        )
        # self.skill_hidden = skill_hidden.unsqueeze(0).expand(batch_size, n_agents, -1)

        self.reset_planner(task, hidden_states_dec.device, batch_size)
        self.last_action = None
        # self.policy_inputs = deque([], maxlen=self.main_args.context_len)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        """we don't load the state of task dynamic decoder"""
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        # for task in self.train_tasks:
        #     self.task2dynamic_decoder[task].cuda()

    def save_models(self, path):
        """we don't save the state of task dynamic decoder"""
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        """we don't load the state of task_encoder"""
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path), map_location=lambda storage, loc: storage
            )
        )

    def _build_agents(self, task2input_shape_info):
        self.agent = agent_REGISTRY[self.main_args.agent](
            task2input_shape_info,
            self.task2decomposer,
            self.task2n_agents,
            self.surrogate_decomposer,
            self.main_args,
        )

    def _build_actions(self, actions):
        actions = actions.reshape(-1) - 5
        zeros = th.zeros_like(actions).to(self.main_args.device)
        actions = th.where(actions >= 0, actions, zeros)
        return actions

    def _build_inputs(self, batch, t, task):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        # get args, n_agents for this specific task
        task_args, n_agents = self.task2args[task], self.task2n_agents[task]
        if task_args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if task_args.obs_agent_id:
            inputs.append(
                th.eye(n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )

        inputs = th.cat([x.reshape(bs * n_agents, -1) for x in inputs], dim=1)
        return inputs

    def seq_build_inputs(self, batch, t, end_t, task):
        flag = t
        seq_inputs = []
        while flag <= end_t:
            inputs = self._build_inputs(batch, flag, task)
            seq_inputs.append(inputs)
            flag += 1

        seq_inputs = th.stack(seq_inputs, dim=1)
        return seq_inputs

    def _get_input_shape(self):
        task2input_shape_info = {}
        for task in self.train_tasks:
            task_scheme = self.task2scheme[task]
            input_shape = task_scheme["obs"]["vshape"]
            last_action_shape, agent_id_shape = 0, 0
            if self.task2args[task].obs_last_action:
                input_shape += task_scheme["actions_onehot"]["vshape"][0]
                last_action_shape = task_scheme["actions_onehot"]["vshape"][0]
            if self.task2args[task].obs_agent_id:
                input_shape += self.task2n_agents[task]
                agent_id_shape = self.task2n_agents[task]
            task2input_shape_info[task] = {
                "input_shape": input_shape,
                "last_action_shape": last_action_shape,
                "agent_id_shape": agent_id_shape,
            }
        return task2input_shape_info


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = th.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=th.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = th.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
