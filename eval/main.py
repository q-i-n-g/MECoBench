import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.scene.scene import Scene
from eval.agents.VLM import VlmAgent
from eval.agents.protocols import (
    BaseCommunicationProtocol,
    NoCommProtocol,
    DiscussThenActProtocol,
    LeaderWorkerProtocol,
    SharedMemoryProtocol,
)
from eval.agents.schemas import ActionDecodeError
import json
import random
import virtualhome.simulation.unity_simulator as comm_unity
import numpy as np
from eval.utils.logger import init_logger, log_print
from eval.utils.utils import goal_test, resolve_action_conflicts, get_satisfied_object_ids
from eval.utils.token_usage import reset as reset_token_usage, snapshot as snapshot_token_usage, total_tokens_all_models


def _resolve_agent_models(num_agents, default_model, agent_models=None, leader_model=None, worker_model=None):
    """Resolve per-agent model names.

    Priority:
    1. agent_models list / str / dict (explicit per-agent or role mapping)
    2. leader_model / worker_model (agent 0 vs others)
    3. default_model for all agents
    """
    if agent_models is not None:
        if isinstance(agent_models, str):
            return [agent_models] * num_agents
        if isinstance(agent_models, dict):
            leader = agent_models.get('leader', agent_models.get('0', default_model))
            worker = agent_models.get('worker', default_model)
            return [leader if i == 0 else worker for i in range(num_agents)]
        models = list(agent_models)
        if len(models) < num_agents:
            raise ValueError(
                f"agent_models has {len(models)} entries but {num_agents} agents are required"
            )
        return models[:num_agents]

    leader = leader_model or default_model
    worker = worker_model or default_model
    if leader_model is not None or worker_model is not None:
        return [leader if i == 0 else worker for i in range(num_agents)]
    return [default_model] * num_agents


class Game:
    def __init__(
        self,
        seed,
        port='8080',
        agent_models=None,
        leader_model=None,
        worker_model=None,
        view_mode='multi_view',
        no_communication=False,
        communication_mode='base',
        discussion_rounds=2,
        historical_dialogue_rounds=None,
        leader_comm_mode='text_only',
    ):
        """
        Args:
            seed: 随机种子
            port: Unity 通信端口
            agent_models: 智能体使用的模型；可为 str、list，或 dict（如
                {'leader': 'gpt-5.4', 'worker': 'gemma-4-31b'}）
            leader_model: leader（agent 0）使用的模型；未设置时回退到 VLM_MAIN_MODEL
            worker_model: worker（agent 1+）使用的模型；未设置时回退到 VLM_MAIN_MODEL
            view_mode: 视角模式
                - 'first_person': 只使用第一人称视角（前视图）
                - 'multi_view': 使用多视角（前、后、左、右）
            no_communication: 是否禁用智能体之间的通信
                - False: 多智能体模式，智能体之间可以发送消息
                - True: 多智能体无通信模式，智能体之间不发送消息
            communication_mode: 沟通协议模式
                - 'base': 先生成一轮 message，再生成 action + memory
                - 'no_comm': 不生成 message，直接 action + memory
                - 'discuss_then_act': 最多多轮讨论后，再生成 action + memory
                - 'leader_worker': worker 单轮汇报，leader 单轮分配后，再生成 action + memory
                - 'shared_memory': 隐式沟通，所有 agent 共享同一份 memory，不发送消息
            discussion_rounds: discuss_then_act 模式的最大讨论轮数，必须 >= 1
            historical_dialogue_rounds: 历史对话滑动窗口轮数（默认读环境变量 VLM_HISTORICAL_DIALOGUE_ROUNDS，缺省 3）
            leader_comm_mode: leader_worker 模式下的沟通形态（仅在 communication_mode='leader_worker' 下生效）
                - 'text_only': worker 文字汇报，leader 仅看到自己的视角（默认）
                - 'text_and_image': worker 文字汇报 + leader 收到所有人视角拼接图
                - 'image_only': worker 不汇报，leader 直接看所有人视角拼接图后分配任务
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        # Unity HTTP 读超时：多视角渲染 / 大图 / 慢机时易超过默认；可用 VWAH_UNITY_HTTP_TIMEOUT 覆盖（秒）
        unity_http_timeout = int(os.environ.get('VWAH_UNITY_HTTP_TIMEOUT', '60'))
        self.comm = comm_unity.UnityCommunication(port=port, timeout_wait=unity_http_timeout)
        self.scene = Scene(self.comm, seed=seed, view_mode=view_mode)
        # 默认模型，可通过参数覆盖
        self.default_model = os.getenv("VLM_MAIN_MODEL", "gpt-5-mini-2")
        self.resolve_model = os.getenv("VLM_RESOLVE_MODEL", "gpt-5-mini-2")
        self.agent_models = agent_models  # None 表示使用默认模型或 leader/worker 拆分
        self.leader_model = leader_model
        self.worker_model = worker_model
        self.view_mode = view_mode  # 视角模式: 'first_person' 或 'multi_view'
        self.no_communication = no_communication  # 是否禁用多智能体通信
        self.communication_mode = communication_mode
        self.discussion_rounds = discussion_rounds
        if historical_dialogue_rounds is None:
            historical_dialogue_rounds = int(
                os.environ.get('VLM_HISTORICAL_DIALOGUE_ROUNDS', '8')
            )
        self.historical_dialogue_rounds = max(1, int(historical_dialogue_rounds))
        valid_leader_modes = ('text_only', 'text_and_image', 'image_only')
        if leader_comm_mode not in valid_leader_modes:
            raise ValueError(
                f"leader_comm_mode must be one of {valid_leader_modes}, got: {leader_comm_mode}"
            )
        self.leader_comm_mode = leader_comm_mode
        self.agents = []
        self.constraints_enabled = False
        self.communication_protocol = None

    def reset(self, task, profile, num_agents=1, mode='raw_goal'):
        """
        Args:
            task: 任务配置
            profile: 智能体配置字典，键为 agent_id（字符串 "0", "1" 等）
            num_agents: 智能体数量。若 task 中启用了 room_constraints，
                        则优先使用 task["room_constraints"]["agents"] 的数量；
                        否则 None 表示使用 profile 中的所有智能体。
        """
        self.task = task
        room_constraints = task.get('room_constraints') or {}
        self.constraints_enabled = bool(room_constraints.get('enabled', False))
        if self.constraints_enabled:
            configured_agents = room_constraints.get('agents') or {}
            if configured_agents:
                num_agents = len(configured_agents)
        # 根据 num_agents 筛选 profile
        filtered_profile = {k: v for k, v in profile.items() if int(k) < num_agents}
        self.profile = filtered_profile
        self.num_agents = len(self.profile)
        
        models = _resolve_agent_models(
            self.num_agents,
            self.default_model,
            agent_models=self.agent_models,
            leader_model=self.leader_model,
            worker_model=self.worker_model,
        )

        self.scene.initialize(task, self.profile)
        self.satisfied, self.unsatisfied, _ = self.goal_test()
        self.agents = []
        # 判断是否为单智能体模式
        # single_agent_mode: 只有一个智能体时为 True
        # 或者多智能体但禁用通信时，每个智能体也使用单智能体的 prompt
        self.single_agent_mode = self.num_agents == 1
        use_single_agent_prompt = self.single_agent_mode or self.no_communication
        use_shared_memory = self.communication_mode == 'shared_memory'
        agent_ids = sorted(self.profile.keys(), key=int)
        agent_names = [self.profile[aid].get("name", f"Agent {aid}") for aid in agent_ids]
        for agent_id in agent_ids:
            agent_profile = self.profile[agent_id]
            agent = VlmAgent(
                model_name=models[int(agent_id)],
                profile=agent_profile,
                single_agent=use_single_agent_prompt,
                view_mode=self.view_mode,
                decouple_ids=True,
                resolve_model_name=self.resolve_model,
                agent_id=int(agent_id),
                all_agent_names=agent_names,
                historical_dialogue_round_window=self.historical_dialogue_rounds,
                shared_memory=use_shared_memory,
            )
            agent.init_task(task, self.unsatisfied, mode=mode)
            self.agents.append(agent)
        self.communication_protocol = self._build_communication_protocol()
        self.no_location_mode = mode in ('raw_goal', 'goal_with_desc')
        self.steps = 0

    def _build_communication_protocol(self):
        if self.single_agent_mode or self.no_communication or self.communication_mode == 'no_comm':
            return NoCommProtocol()
        if self.communication_mode == 'base':
            return BaseCommunicationProtocol()
        if self.communication_mode == 'discuss_then_act':
            return DiscussThenActProtocol(rounds=self.discussion_rounds)
        if self.communication_mode == 'leader_worker':
            return LeaderWorkerProtocol(
                leader_agent_id=0,
                leader_comm_mode=self.leader_comm_mode,
            )
        if self.communication_mode == 'shared_memory':
            return SharedMemoryProtocol()
        raise ValueError(f"Unsupported communication_mode: {self.communication_mode}")
        
    def goal_test(self):
        s, g = self.comm.environment_graph()
        satisfied, unsatisfied, result = goal_test(g, self.task)
        return satisfied, unsatisfied, result


    def run(self, ouput_path, steps_threshold = 25):
        log_jsonl_path = ouput_path.replace('.jsonl', '_logs.jsonl') if ouput_path.endswith('.jsonl') else ouput_path + '_logs.jsonl'
        self.scene.record_dir = os.path.join(os.path.dirname(log_jsonl_path), 'img')
        os.makedirs(self.scene.record_dir, exist_ok=True)
        init_logger(log_jsonl_path)
        reset_token_usage()
        with open(ouput_path, 'w') as f:
            pass
        feedback = [None] * len(self.agents)
        last_action = ['No action'] * len(self.agents)
        # 保存上一轮的所有 messages，格式：list[CommEvent]
        last_round_messages = []
        # 停滞检测：连续无进展时向 agent 注入提示
        stagnation_counter = 0
        last_satisfied_count = sum(v for v in self.satisfied.values() if v > 0)
        while True:
            if self.steps > steps_threshold:
                break
            log_print(f"Step {self.steps}", step=self.steps)
            obs = self.scene.get_observation(self.steps)
            satisfied_obj_ids = get_satisfied_object_ids(self.scene.curr_graph, self.task)
            for agent_obs in obs:
                agent_obs['satisfied_obj_ids'] = list(satisfied_obj_ids)
            scripts = []
            proposals = [None] * len(self.agents)
            comm_result = self.communication_protocol.run_phase(
                self.agents,
                obs,
                self.satisfied,
                self.steps,
                feedback,
                last_round_messages,
            )
            for agent_id, agent in enumerate(self.agents):
                log_print(f"Agent {agent_id}", step=self.steps, agent_id=agent_id)
                try:
                    log_print(f"holding objects: {obs[agent_id]['hoding_obj']}", step=self.steps, agent_id=agent_id)
                    action, ids = agent.plan_action_and_memory(
                        obs[agent_id],
                        self.satisfied,
                        step=self.steps,
                        agent_id=agent_id,
                        feedback=feedback[agent_id],
                        extra_dialogue=comm_result.action_dialogue_per_agent.get(agent_id, []),
                    )
                    # Shared-memory mode: propagate this agent's updated memory
                    # to the shared store so the next agent sees it immediately.
                    if hasattr(self.communication_protocol, 'sync_after_action'):
                        self.communication_protocol.sync_after_action(agent)
                except ActionDecodeError as e:
                    log_print(e, step=self.steps, agent_id=agent_id)
                    if hasattr(self.communication_protocol, 'sync_after_action'):
                        agent.memory = e.memory or agent.memory
                        self.communication_protocol.sync_after_action(agent)
                    feedback[agent_id] = f"Cannot perform action: {e}"
                    last_action[agent_id] = 'Waiting'
                    continue
                except Exception as e:
                    log_print(e, step=self.steps, agent_id=agent_id)
                    feedback[agent_id] = f"Cannot perform action: {e}"
                    last_action[agent_id] = 'Waiting'
                    continue
                try:
                    script = self.scene.get_script(action, ids, agent_id)
                except Exception as e:
                    log_print(e, step=self.steps, agent_id=agent_id)
                    feedback[agent_id] = f"Cannot perform action: {e}"
                    last_action[agent_id] = 'Waiting'
                    continue
                log_print(step=self.steps, agent_id=agent_id, structured={"action": script})
                feedback[agent_id] = None
                proposals[agent_id] = {
                    'action': action,
                    'ids': ids,
                    'script': script,
                }
            for agent_id, agent in enumerate(self.agents):
                agent.append_dialogue_history(
                    comm_result.action_dialogue_per_agent.get(agent_id, [])
                )
            allowed, conflict_groups, winners = resolve_action_conflicts(proposals)
            for obj_id, agent_ids in conflict_groups.items():
                if len(agent_ids) <= 1:
                    continue
                winner = winners[obj_id]
                for conflict_agent_id in agent_ids:
                    if conflict_agent_id == winner:
                        continue
                    feedback[conflict_agent_id] = (
                        f"conflict: someone else is also "
                        f"{proposals[conflict_agent_id]['action']} "
                        f"the same object (id={obj_id}), you can't do at the same time."
                    )
                    last_action[conflict_agent_id] = 'Waiting'

            for agent_id, proposal in enumerate(proposals):
                if proposal is None or not allowed[agent_id]:
                    continue
                script = proposal['script']
                log_print(script, step=self.steps, agent_id=agent_id)
                self.agents[agent_id].update(script)
                last_action[agent_id] = script
                scripts.append(script)
            # 保存当前轮 messages 为下一轮协议使用
            last_round_messages = list(comm_result.transcript)
            s, m = self.scene.step(scripts)

            # 将 handover 的模拟器反馈同步给 giver 和 receiver
            if isinstance(m, dict):
                for agent_id, proposal in enumerate(proposals):
                    if proposal is None or not allowed[agent_id]:
                        continue
                    if proposal['action'] != 'handover':
                        continue
                    receiver_agent_id = int(proposal['ids'][1][0])
                    giver_msg = m.get(str(agent_id), {}).get('message', '')
                    obj_id = int(proposal['ids'][0][0])
                    obj_node = self.scene.id2nodes.get(obj_id)
                    obj_name = obj_node['class_name'] if obj_node else 'object'
                    handover_feedback = (
                        f"Handover {obj_name} (id={obj_id}) "
                        f"from Agent {agent_id} to Agent {receiver_agent_id}: {giver_msg}"
                    )
                    feedback[agent_id] = handover_feedback
                    feedback[receiver_agent_id] = handover_feedback

            self.steps += 1
            self.satisfied, self.unsatisfied, result = self.goal_test()
            satisfied_step = self.steps - 1 if self.steps > 0 else self.steps
            log_print(f"Satisfied: {self.satisfied}", step=satisfied_step)
            for agent_id in range(len(self.agents)):
                log_print(
                    step=satisfied_step,
                    agent_id=agent_id,
                    structured={"satisfied_task": self.satisfied},
                )

            if result:
                break

            # 停滞检测：连续 N 步无新的子目标完成时，注入提示
            current_satisfied_count = sum(
                v for v in self.satisfied.values() if v > 0
            )
            if current_satisfied_count > last_satisfied_count:
                stagnation_counter = 0
                last_satisfied_count = current_satisfied_count
            else:
                stagnation_counter += 1

            STAGNATION_THRESHOLD = 6
            if self.no_location_mode and stagnation_counter >= STAGNATION_THRESHOLD and stagnation_counter % STAGNATION_THRESHOLD == 0:
                stagnation_hint = (
                    f"WARNING: No task progress in the last {stagnation_counter} steps. "
                    f"Your current approach is NOT working. "
                    f"Try a COMPLETELY DIFFERENT strategy: "
                    f"go to a DIFFERENT room you haven't fully explored, "
                    f"or search on other open surfaces and other containers "
                )
                log_print(f"Stagnation detected: {stagnation_counter} steps without progress", step=satisfied_step)
                for agent_id in range(len(self.agents)):
                    if feedback[agent_id] is None:
                        feedback[agent_id] = stagnation_hint
                    else:
                        feedback[agent_id] += " " + stagnation_hint


def _run_case_worker(args):
    """子进程内跑单个 case；独占一个 Unity 端口。须为顶层函数以便 multiprocessing pickle。"""
    (
        case_idx,
        task,
        port,
        main_output_dir,
        base_seed,
        view_mode,
        no_communication,
        communication_mode,
        profile_agent0,
        steps_threshold,
        num_agents,
        historical_dialogue_rounds,
        goal_mode,
        leader_comm_mode,
        leader_model,
        worker_model,
    ) = args
    case_seed = base_seed + case_idx
    random.seed(case_seed)
    np.random.seed(case_seed)
    case_output_dir = os.path.join(main_output_dir, f'case_{case_idx}')
    os.makedirs(case_output_dir, exist_ok=True)
    output_path = os.path.join(case_output_dir, 'output')
    result = {
        'case_idx': case_idx,
        'port': port,
        'satisfied_count': 0,
        'total_count': 0,
        'completion_rate': 0.0,
        'satisfied': {},
        'unsatisfied': {},
        'steps': 0,
        'success': False,
        'error': None,
    }
    try:
        game = Game(
            seed=case_seed,
            port=str(port),
            view_mode=view_mode,
            no_communication=no_communication,
            communication_mode=communication_mode,
            historical_dialogue_rounds=historical_dialogue_rounds,
            leader_comm_mode=leader_comm_mode,
            leader_model=leader_model,
            worker_model=worker_model,
        )
        game.reset(task, profile_agent0, mode=goal_mode, num_agents=num_agents)
        game.run(output_path, steps_threshold=steps_threshold)
        satisfied = game.satisfied
        unsatisfied = game.unsatisfied
        satisfied_count = sum(satisfied.values())
        total_count = satisfied_count + sum(unsatisfied.values())
        completion_rate = satisfied_count / total_count if total_count > 0 else 0.0
        usage_snap = snapshot_token_usage()
        result.update(
            {
                'satisfied_count': satisfied_count,
                'total_count': total_count,
                'completion_rate': completion_rate,
                'satisfied': satisfied,
                'unsatisfied': unsatisfied,
                'steps': game.steps,
                'success': sum(unsatisfied.values()) == 0,
                'token_usage': usage_snap,
                'token_usage_estimate_total': total_tokens_all_models(),
            }
        )
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
        snap = snapshot_token_usage()
        if snap:
            result['token_usage'] = snap
            result['token_usage_estimate_total'] = total_tokens_all_models()
    case_result_path = os.path.join(case_output_dir, 'result.json')
    with open(case_result_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from tqdm import tqdm
    from datetime import datetime

    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    profile = json.load(open('data/chars/profile_sample.json', 'r'))
    task_file = os.environ.get('VWAH_TASK_FILE', 'data/task/parallel.json')
    with open(task_file, 'r') as f:
        tasks = json.load(f)

    # Match src/start_unity_multi_8gpu.sh by default; override with env vars.
    unity_base_port = int(os.environ.get('UNITY_BASE_PORT', '8001'))
    unity_num_sims = int(os.environ.get('UNITY_NUM_SIMULATORS', '1'))
    ports = [unity_base_port + i for i in range(unity_num_sims)]
    steps_threshold = int(os.environ.get('VWAH_STEPS_THRESHOLD', '10'))

    view_mode = os.environ.get('VWAH_VIEW_MODE', 'multi_view')
    # 与 bash 串联实验配合：export VWAH_NO_COMMUNICATION=1 VWAH_COMMUNICATION_MODE=base VWAH_NUM_AGENTS=2
    _no_comm = os.environ.get('VWAH_NO_COMMUNICATION', '0').strip().lower()
    no_communication = _no_comm in ('1', 'true', 'yes', 'on')
    communication_mode = os.environ.get('VWAH_COMMUNICATION_MODE', 'base')
    num_agents = int(os.environ.get('VWAH_NUM_AGENTS', '2'))
    historical_dialogue_rounds = int(os.environ.get('VLM_HISTORICAL_DIALOGUE_ROUNDS', '8'))
    goal_mode = os.environ.get('VWAH_GOAL_MODE', 'goal_with_location_and_desc')
    # leader_worker 模式专用：选择 leader 与 worker 之间的沟通形态。
    #   text_only       — worker 文字汇报，leader 只看自己的视角（默认）
    #   text_and_image  — worker 文字汇报 + leader 收到所有人视角拼接图
    #   image_only      — worker 不汇报，leader 直接看所有人视角拼接图后分配
    # 兼容旧脚本：若未设置 VWAH_LEADER_COMM_MODE 但 VWAH_ENABLE_VISUAL_COMM=true，
    # 则等价于 text_and_image。
    _leader_comm_mode_env = os.environ.get('VWAH_LEADER_COMM_MODE', 'text_only').strip().lower()
    if _leader_comm_mode_env:
        leader_comm_mode = _leader_comm_mode_env
    else:
        _legacy_visual = os.environ.get('VWAH_ENABLE_VISUAL_COMM', '0').strip().lower()
        leader_comm_mode = 'text_and_image' if _legacy_visual in ('1', 'true', 'yes', 'on') else 'text_only'

    # leader_worker 模式可为 leader / worker 分别指定模型；未设置时均回退到 VLM_MAIN_MODEL
    leader_model = os.environ.get('VLM_LEADER_MODEL', '').strip() or None
    worker_model = os.environ.get('VLM_WORKER_MODEL', '').strip() or None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.environ.get('VWAH_OUTPUT_ROOT', 'test/mm-communication')
    main_output_dir = os.path.join(output_root, f'batch_{timestamp}')
    os.makedirs(main_output_dir, exist_ok=True)

    task_id_list_str = os.environ.get('VWAH_TASK_ID_LIST', '').strip()
    if task_id_list_str:
        idx = [int(x) for x in task_id_list_str.split(',') if x.strip()]
    else:
        idx = list(range(len(tasks)))
    all_results = []

    # 动态并行：最多同时占用 len(ports) 个 Unity 端口；任意一个 case 完成，
    # 立刻用“同一个端口”补上下一 个 case，避免同端口并发冲突。
    case_iter = iter(idx)
    max_concurrency = min(len(ports), len(idx))
    with ProcessPoolExecutor(max_workers=len(ports)) as ex, tqdm(
        total=len(idx), desc="Running cases"
    ) as pbar:
        future_to_port = {}
        for port in ports[:max_concurrency]:
            case_idx = next(case_iter)
            fut = ex.submit(
                _run_case_worker,
                (
                    case_idx,
                    tasks[case_idx],
                    port,
                    main_output_dir,
                    seed,
                    view_mode,
                    no_communication,
                    communication_mode,
                    profile[0],
                    steps_threshold,
                    num_agents,
                    historical_dialogue_rounds,
                    goal_mode,
                    leader_comm_mode,
                    leader_model,
                    worker_model,
                ),
            )
            future_to_port[fut] = port

        while future_to_port:
            done, _ = wait(
                future_to_port.keys(),
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                # 取出“刚释放的端口”，并立即用它提交下一个 case
                freed_port = future_to_port.pop(fut)
                all_results.append(fut.result())
                pbar.update(1)

                next_case_idx = next(case_iter, None)
                if next_case_idx is None:
                    continue

                next_fut = ex.submit(
                    _run_case_worker,
                    (
                        next_case_idx,
                        tasks[next_case_idx],
                        freed_port,
                        main_output_dir,
                        seed,
                        view_mode,
                        no_communication,
                        communication_mode,
                        profile[0],
                        steps_threshold,
                        num_agents,
                        historical_dialogue_rounds,
                        goal_mode,
                        leader_comm_mode,
                        leader_model,
                        worker_model,
                    ),
                )
                future_to_port[next_fut] = freed_port

    for r in all_results:
        if r.get('error'):
            print(f"Case {r['case_idx']} (port {r['port']}): ERROR {r['error']}")
        else:
            print(
                f"Case {r['case_idx']} (port {r['port']}): "
                f"{r['satisfied_count']}/{r['total_count']} = {r['completion_rate']:.2%}"
            )

    sum_tokens = sum(
        int(r.get('token_usage_estimate_total') or 0) for r in all_results
    )
    summary = {
        'total_cases': len(idx),
        'successful_cases': sum(1 for r in all_results if r['success']),
        'average_completion_rate': sum(r['completion_rate'] for r in all_results) / len(all_results)
        if all_results
        else 0.0,
        'token_usage_estimate_total_batch': sum_tokens,
        'token_usage_estimate_avg_per_case': (sum_tokens / len(all_results)) if all_results else 0.0,
        'unity_ports': f'{ports[0]}-{ports[-1]} ({len(ports)} sims)',
        'experiment': {
            'task_file': task_file,
            'view_mode': view_mode,
            'no_communication': no_communication,
            'communication_mode': communication_mode,
            'num_agents': num_agents,
            'goal_mode': goal_mode,
            'historical_dialogue_rounds': historical_dialogue_rounds,
            'steps_threshold': steps_threshold,
            'output_root': output_root,
            'leader_comm_mode': leader_comm_mode,
            'leader_model': leader_model or os.environ.get('VLM_MAIN_MODEL', 'gpt-5-mini-2'),
            'worker_model': worker_model or os.environ.get('VLM_MAIN_MODEL', 'gpt-5-mini-2'),
        },
        'results': all_results,
    }

    summary_path = os.path.join(main_output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Summary ===")
    print(f"Successful cases: {summary['successful_cases']}/{len(idx)}")
    print(f"Average completion rate: {summary['average_completion_rate']:.2%}")
    print(
        f"Token usage (estimate, sum over cases): {summary['token_usage_estimate_total_batch']} "
        f"(avg/case: {summary['token_usage_estimate_avg_per_case']:.1f})"
    )
    print(f"Results saved to: {main_output_dir}")
