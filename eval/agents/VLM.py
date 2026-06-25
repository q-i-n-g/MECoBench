from openai import OpenAI
import dotenv
import re
import os
import json
import time

from .action import ActionDecodeHelper
from .action_resolver import ActionResolver
from .utils import *
from .prompt_tranfer import PromptTransfer
from .schemas import ActionMemoryDecision, ActionDecodeError

curr_dir = os.path.dirname(os.path.abspath(__file__))
from eval.scene.utils import *
from eval.utils.logger import log_print
from eval.utils.token_usage import record_chat_completion, SCOPE_MAIN

dotenv.load_dotenv(os.path.join(curr_dir, '.env'))

prompt_transfer = PromptTransfer()
prompts = prompt_transfer.get_prompts()

_CONTAINER_HINT = (
    '\nThe possible locations for the target object are open surfaces (such as the kitchentable, kitchencounter, coffeetable, etc.) and containers (possible containers may contain objects: bathroomcabinet, cabinet in bedroom, fridge and microwave).'
    'Search the surfaces and containers through all rooms for the target objects.'
)

cfg = json.load(open(os.path.join(curr_dir, 'models.json'), 'r'))

def _build_client_from_model_name(model_name):
    model_name = (model_name or "").strip()
    if not model_name:
        raise ValueError("model_name is required")

    model_cfg = cfg.get(model_name)
    base_url = model_cfg.get("base_url")
    base_url_env = model_cfg.get("base_url_env")
    if base_url_env:
        base_url = os.getenv(base_url_env, base_url)

    return {
        "client": OpenAI(
            api_key=os.getenv(model_cfg["api_key_env"]),
            base_url=base_url,
        ),
        "model_name": model_cfg["model"],
        "config": model_cfg,
    }

class VlmAgent:
    def __init__(
        self,
        model_name,
        profile,
        single_agent=False,
        view_mode='multi_view',
        decouple_ids=False,
        resolve_model_name=None,
        agent_id=None,
        all_agent_names=None,
        historical_dialogue_round_window=None,
        shared_memory=False,
    ):
        """
        Args:
            model_name: 主模型名称；为空时使用环境变量 VLM_MAIN_MODEL 或默认 gpt-5-mini
            profile: 智能体配置
            single_agent: 是否为单智能体模式
            view_mode: 视角模式
                - 'first_person': 只使用第一人称视角（前视图）
                - 'multi_view': 使用多视角（前、后、左、右）
            resolve_model_name: 用于解析ID的模型名称
        """
        # Main model client
        main_model_name = model_name or os.getenv("VLM_MAIN_MODEL")
        main_client_cfg = _build_client_from_model_name(main_model_name)
        self._main_model_key = (main_model_name or "").strip()
        self.cfg = main_client_cfg["config"]
        self.model_name = main_client_cfg["model_name"]
        self.client = main_client_cfg["client"]
        self.generation_config = self.cfg.get("generation_config", {})

        # Embedding model
        embedding_model_name = os.getenv("VLM_EMBEDDING_MODEL", "text-embedding-3-large")
        embedding_client_cfg = _build_client_from_model_name(embedding_model_name)

        # Resolve model
        resolve_model_name = resolve_model_name or os.getenv("VLM_RESOLVE_MODEL")
        resolve_client_cfg = _build_client_from_model_name(resolve_model_name)
        resolve_client = resolve_client_cfg["client"]
        resolve_model = resolve_client_cfg["model_name"]

        # Agent identity
        self.profile = profile
        self.name = self.profile['name']
        self.agent_id = agent_id
        self.all_agent_names = all_agent_names or []
        self.action_decoder = ActionDecodeHelper(self.all_agent_names)
        self.single_agent = single_agent
        self.view_mode = view_mode
        env_decouple = os.getenv("VLM_DECOUPLE_IDS", "").lower() in ["1", "true", "yes"]
        self.decouple_ids = decouple_ids or env_decouple
        self.use_structured_io = os.getenv("VLM_STRUCTURED_IO", "1").lower() in ["1", "true", "yes"]
        if historical_dialogue_round_window is None:
            historical_dialogue_round_window = int(
                os.getenv("VLM_HISTORICAL_DIALOGUE_ROUNDS", "3")
            )
        self.historical_dialogue_round_window = max(
            1, int(historical_dialogue_round_window)
        )
        self.shared_memory = shared_memory
        self.other_agent_names = [] if self.single_agent else self._compute_other_agent_names()
        self.task_room_constraints = {}
        self.prompt_restriction_context = None
        self.task_valid_actions = []
        self.action_str = ""

        # Action resolver (handles candidate collection, similarity filtering, VLM-based ID resolution)
        self.action_resolver = ActionResolver(
            resolve_client=resolve_client,
            resolve_model_name=resolve_model,
            resolve_model_key=(resolve_model_name or "").strip(),
            embedding_client=embedding_client_cfg["client"],
            embedding_model=embedding_client_cfg["model_name"],
            embedding_model_key=(embedding_model_name or "").strip(),
            sim_threshold=float(os.getenv("VLM_EMBED_SIM_THRESHOLD", "0.4")),
            min_keep=int(os.getenv("VLM_EMBED_MIN_KEEP", "5")),
            embedding_enabled=os.getenv("VLM_USE_EMBEDDING_FILTER", "1").lower() in ["1", "true", "yes"],
            vis_max_distance=float(os.getenv("VLM_VIS_MAX_DISTANCE", "5")),
            vis_min_area=int(os.getenv("VLM_VIS_MIN_AREA", "400")),
            vis_min_area_for_far=int(os.getenv("VLM_VIS_MIN_AREA_FAR", "60000")),
            view_mode=self.view_mode,
            action_decoder=self.action_decoder,
            resolve_prompt_template=prompts['ACTION_DESC_RESOLVE_PROMPT'],
        )

    # ---- Agent Identity ----

    def _compute_other_agent_names(self):
        if not self.all_agent_names:
            return []
        if self.name in self.all_agent_names:
            return [name for name in self.all_agent_names if name != self.name]
        if self.agent_id is not None and 0 <= self.agent_id < len(self.all_agent_names):
            return [name for idx, name in enumerate(self.all_agent_names) if idx != self.agent_id]
        return [name for name in self.all_agent_names]

    def _format_other_agents_str(self):
        if not self.other_agent_names:
            return "none"
        return ", ".join(self.other_agent_names)

    def _resolve_agent_name_to_id(self, agent_name):
        if not agent_name:
            return None
        lowered = str(agent_name).strip().lower()
        if lowered in self.action_decoder.agent_name_to_id:
            return self.action_decoder.agent_name_to_id[lowered]
        for idx, name in enumerate(self.all_agent_names):
            if str(name).strip().lower() == lowered:
                return idx
        return None

    def _empty_room_memory(self):
        return {
            "exploration": "unexplored",
            "objects": [],
            "containers": [],
            "notes": [],
        }

    def _initial_memory_json(self):
        memory = {
            "livingroom": self._empty_room_memory(),
            "kitchen": self._empty_room_memory(),
            "bedroom": self._empty_room_memory(),
            "bathroom": self._empty_room_memory(),
            "global": {
                "notes": [],
            },
        }
        if self.shared_memory:
            # Shared memory mode: all agents read/write the same memory.
            # Include coordination fields for implicit communication.
            memory["global"]["agent_locations"] = {
                name: "unknown" for name in self.all_agent_names
            }
            memory["global"]["agent_plans"] = {
                name: "" for name in self.all_agent_names
            }
            memory["global"]["agent_status"] = {
                name: "" for name in self.all_agent_names
            }
            memory["global"]["coordination_board"] = []
        elif not self.single_agent:
            memory["global"]["agent_locations"] = {}
            memory["global"]["communication_summary"] = ""
            for agent_name in self.other_agent_names:
                memory["global"]["agent_locations"][agent_name] = "unknown"
            if self.prompt_restriction_context and self.prompt_restriction_context.get("enabled"):
                memory["global"]["agent_allowed_rooms"] = {
                    agent_name: "unknown" for agent_name in self.all_agent_names
                }
        return json.dumps(memory, ensure_ascii=True, indent=2)

    def init_task(self, task, goal, mode='raw_goal'):
        self.task = task
        self.goal = goal
        self.task_room_constraints = task.get('room_constraints') or {}
        self.task_valid_actions = task.get("valid_actions") or []
        self.prompt_restriction_context = prompt_transfer.build_restriction_context(
            self.task_room_constraints,
            self.agent_id,
            self.all_agent_names,
            task.get("department_topology"),
        )
        self.task_goal_str = task.get(mode) if task.get(mode) is not None else simple_goal_for_vlm(self.goal)
        self.no_location_mode = mode in ('raw_goal', 'goal_with_desc')
        if self.no_location_mode:
            self.task_goal_str += _CONTAINER_HINT
            container_hint = _CONTAINER_HINT
        else:
            container_hint = ''
        log_print(self.task_goal_str)
        goal_str = task.get(mode) if task.get(mode) is not None else goal_for_vlm(goal)
        goal_str += container_hint

        if self.single_agent:
            self.system_prompt = prompt_transfer.build_system_prompt(
                name=self.profile['name'],
                goal=goal_str,
                single_agent=True,
            )
        else:
            self.system_prompt = prompt_transfer.build_system_prompt(
                name=self.profile['name'],
                goal=goal_str,
                single_agent=False,
                other_agents=self._format_other_agents_str()
            )

        self.memory = self._initial_memory_json()
        self.dialogue_history = []
        self.dialogue_round_history = []
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.hold_objs = []
        self.action_history = []
        self.action_str = ""

        self.action_resolver.set_task(self.task, self.prompt_restriction_context)

    def decode_action(self, action, obs, step=0, agent_id=0):
        return self.action_resolver.decode_action(action, obs, step=step, agent_id=agent_id)

    def append_dialogue_history(self, dialogue_entries):
        if self.single_agent or not dialogue_entries:
            return
        round_entries = []
        for item in dialogue_entries:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role is None or content is None:
                continue
            round_entries.append({"role": role, "content": content})
        if not round_entries:
            return
        self.dialogue_round_history.append(round_entries)
        self.dialogue_round_history = self.dialogue_round_history[
            -self.historical_dialogue_round_window :
        ]
        self.dialogue_history = [
            item
            for round_items in self.dialogue_round_history
            for item in round_items
        ]

    def _format_dialogue_entries(self, dialogue_entries):
        lines = []
        for item in dialogue_entries:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role is None or content is None:
                continue
            safe_content = strip_id_tokens(content)
            lines.append(f"{role}: {safe_content}")
        return "\n".join(lines).strip() if lines else "None."

    def _build_turn_prompt_context(self, obs, complete_subgoals=None, feedback=None, extra_dialogue=None, local_context=None):
        complete_subgoals = complete_subgoals or {}
        extra_dialogue = extra_dialogue or []

        self.hoding_obj_str = ', '.join([item['class_name'] for item in obs['hoding_obj']]) if len(obs['hoding_obj']) > 0 else 'nothing'
        self.complete_str = subgoals_to_string(complete_subgoals)

        action_history_str = ', '.join([strip_id_tokens(item) for item in self.action_history]) if len(self.action_history) > 0 else 'No action has been taken yet.'
        last_executed_action = strip_id_tokens(self.action_history[-1]) if len(self.action_history) > 0 else 'No action has been taken yet.'
        last_intended_action = strip_id_tokens(self.action_str) if getattr(self, "action_str", None) else 'No intended action has been produced yet.'
        last_action = f"Executed last action: {last_executed_action} Intended last action: {last_intended_action}"

        if feedback is not None:
            action_feedback = f"Last action:{last_intended_action}. Feedback:{feedback}"
            if self.use_structured_io:
                action_feedback = strip_id_tokens(action_feedback)
        else:
            if len(self.action_history) > 0:
                action_feedback = "success"
            else:
                action_feedback = "No action has taken yet."

        historical_dialogue_str = "None."
        current_round_dialogue_str = "None."
        if not self.single_agent:
            historical_entries = [
                item
                for round_items in self.dialogue_round_history[
                    -self.historical_dialogue_round_window :
                ]
                for item in round_items
            ]
            historical_dialogue_str = self._format_dialogue_entries(historical_entries)
            current_round_dialogue_str = self._format_dialogue_entries(extra_dialogue)

        return {
            "memory": self.memory,
            "historical_dialogue": historical_dialogue_str,
            "current_round_dialogue": current_round_dialogue_str,
            "local_context": local_context or "None.",
            "action_history": action_history_str,
            "feedback": action_feedback,
            "last_action": last_action,
            "current_room": obs["current_room"],
            "holding_objects": self.hoding_obj_str,
            "task_goal": self.task_goal_str,
            "task_progress": self.complete_str,
            "other_agents": self._format_other_agents_str(),
        }

    def _run_prompt(self, prompt_text, obs=None, use_images=False, composite_image=None):
        """
        composite_image: optional pre-built BGR np.ndarray. When provided, a
            single image_url content item is appended (with an explanatory text
            header) and `use_images` is ignored — the caller is responsible for
            making sure the composite already contains every view the model
            needs. Used in leader_worker visual comm mode where the leader
            receives exactly one stitched image containing its own view and
            every worker's view, each row labeled with the agent name.
        """
        content = [{"type": "text", "text": prompt_text}]
        if composite_image is not None:
            content.append({
                "type": "text",
                "text": (
                    "## Combined view from all agents\n"
                    "The single image below is a vertical stack of the current views of "
                    "the leader and every worker. Each row is labeled above with the "
                    "corresponding agent's name. In multi-view mode, each row itself is "
                    "a horizontal stack of that agent's front/back/left/right views."
                ),
            })
            content.append(build_image_content_item(composite_image))
        elif use_images:
            if obs is None:
                raise ValueError("obs is required for multimodal prompt execution")
            content += build_raw_view_content(obs, self.view_mode)
        curr_vision = {"role": "user", "content": content}
        gen_cfg = self.generation_config
        api_params = {
            "model": self.model_name,
            "messages": self.messages + [curr_vision],
        }
        if "temperature" in gen_cfg:
            api_params["temperature"] = gen_cfg["temperature"]
        if "top_p" in gen_cfg:
            api_params["top_p"] = gen_cfg["top_p"]
        if "presence_penalty" in gen_cfg:
            api_params["presence_penalty"] = gen_cfg["presence_penalty"]
        if "max_output_tokens" in gen_cfg:
            api_params["max_tokens"] = gen_cfg["max_output_tokens"]
        extra_body = dict(gen_cfg.get("extra_body") or {})
        if "top_k" in gen_cfg:
            extra_body["top_k"] = gen_cfg["top_k"]
        if "repetition_penalty" in gen_cfg:
            extra_body["repetition_penalty"] = gen_cfg["repetition_penalty"]
        if extra_body:
            api_params["extra_body"] = extra_body

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(**api_params)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)
        record_chat_completion(SCOPE_MAIN, self._main_model_key, response)
        answer = response.choices[0].message.content
        if "</think>" in answer:
            answer = answer.split("</think>")[1].strip()
        return answer

    def _normalize_message_text(self, message_text):
        if message_text is None:
            return None
        cleaned = message_text.strip()
        if not cleaned or cleaned.startswith("<NO_MESSAGE>"):
            return None
        return cleaned

    def plan_message(self, obs, complete_subgoals=None, step=0, agent_id=0, feedback=None, extra_dialogue=None):
        if self.single_agent:
            return None

        prompt_context = self._build_turn_prompt_context(
            obs,
            complete_subgoals=complete_subgoals,
            feedback=feedback,
            extra_dialogue=extra_dialogue,
        )
        comm_prompt = prompt_transfer.build_communication_prompt(
            context=prompt_context,
            restriction_context=self.prompt_restriction_context,
            valid_actions=self.task_valid_actions,
        )
        answer = self._run_prompt(comm_prompt, obs=obs, use_images=True)
        log_print(answer)
        parsed_output = parse_output_blocks(answer)
        message = self._normalize_message_text(parsed_output.get("message"))

        log_print(
            step=step,
            agent_id=agent_id,
            structured={
                "message_thinking": parsed_output.get("message_thinking") or "",
                "message": message or "",
            },
        )
        return message

    def plan_discussion_first_round(self, obs, complete_subgoals=None, step=0, agent_id=0, feedback=None, extra_dialogue=None):
        prompt_context = self._build_turn_prompt_context(
            obs,
            complete_subgoals=complete_subgoals,
            feedback=feedback,
            extra_dialogue=extra_dialogue,
        )
        prompt_text = prompt_transfer.build_discussion_first_round_prompt(
            context=prompt_context,
            restriction_context=self.prompt_restriction_context,
            valid_actions=self.task_valid_actions,
        )

        answer = self._run_prompt(prompt_text, obs=obs, use_images=True)
        log_print(answer)
        parsed = parse_discussion_first_round_output(answer)
        message = self._normalize_message_text(parsed.get("message"))
        local_context = (parsed.get("local_context") or "").strip() or "None."
        log_print(
            step=step,
            agent_id=agent_id,
            structured={
                "discussion_thinking": parsed.get("thinking") or "",
                "discussion_ready": parsed["ready"],
                "discussion_message": message or "",
                "discussion_local_context": local_context,
            },
        )
        return {
            "ready": parsed["ready"],
            "message": message,
            "local_context": local_context,
            "raw_output": answer,
            "thinking": parsed.get("thinking") or "",
        }

    def plan_discussion_text_round(self, obs, complete_subgoals=None, step=0, agent_id=0, feedback=None, extra_dialogue=None, local_context=None):
        prompt_context = self._build_turn_prompt_context(
            obs,
            complete_subgoals=complete_subgoals,
            feedback=feedback,
            extra_dialogue=extra_dialogue,
            local_context=local_context,
        )
        prompt_text = prompt_transfer.build_discussion_text_round_prompt(
            context=prompt_context,
            restriction_context=self.prompt_restriction_context,
            valid_actions=self.task_valid_actions,
        )
        answer = self._run_prompt(prompt_text, use_images=False)
        log_print(answer)
        parsed = parse_discussion_text_round_output(answer)
        message = self._normalize_message_text(parsed.get("message"))
        log_print(
            step=step,
            agent_id=agent_id,
            structured={
                "discussion_thinking": parsed.get("thinking") or "",
                "discussion_ready": parsed["ready"],
                "discussion_message": message or "",
            },
        )
        return {
            "ready": parsed["ready"],
            "message": message,
            "raw_output": answer,
            "thinking": parsed.get("thinking") or "",
        }

    def plan_worker_report(self, obs, leader_name, complete_subgoals=None, step=0, agent_id=0, feedback=None, extra_dialogue=None):
        prompt_context = self._build_turn_prompt_context(
            obs,
            complete_subgoals=complete_subgoals,
            feedback=feedback,
            extra_dialogue=extra_dialogue,
        )
        prompt_text = prompt_transfer.build_worker_report_prompt(
            leader_name=leader_name,
            context=prompt_context,
            restriction_context=self.prompt_restriction_context,
            step=step,
        )
        answer = self._run_prompt(prompt_text, obs=obs, use_images=True)
        log_print(answer)
        parsed = parse_worker_report_output(answer)
        message = self._normalize_message_text(parsed.get("message"))
        log_print(
            step=step,
            agent_id=agent_id,
            structured={
                "report_thinking": parsed.get("thinking") or "",
                "report_message": message or "",
            },
        )
        return {
            "message": message,
            "raw_output": answer,
            "thinking": parsed.get("thinking") or "",
        }

    def plan_leader_assignments(self, obs, assignee_agent_ids, complete_subgoals=None, step=0, agent_id=0, feedback=None, extra_dialogue=None, composite_image=None, leader_comm_mode='text_only'):
        assignee_names = [self.all_agent_names[target_id] for target_id in assignee_agent_ids]
        prompt_context = self._build_turn_prompt_context(
            obs,
            complete_subgoals=complete_subgoals,
            feedback=feedback,
            extra_dialogue=extra_dialogue,
        )
        prompt_text = prompt_transfer.build_leader_assignment_prompt(
            assignee_names=assignee_names,
            context=prompt_context,
            restriction_context=self.prompt_restriction_context,
            valid_actions=self.task_valid_actions,
            leader_comm_mode=leader_comm_mode,
        )
        max_json_attempts = 3
        parsed = None
        answer = ""
        use_own_view = composite_image is None
        for attempt in range(max_json_attempts):
            answer = self._run_prompt(
                prompt_text,
                obs=obs,
                use_images=use_own_view,
                composite_image=composite_image,
            )
            log_print(answer)
            try:
                parsed = parse_leader_assignments_output(answer)
                break
            except ValueError as exc:
                log_print(exc, step=step, agent_id=agent_id)
                if attempt == max_json_attempts - 1:
                    parsed = {"thinking": "", "messages": []}

        expected_agent_ids = set(assignee_agent_ids)
        seen_agent_ids = set()
        assignment_events = []
        for item in parsed["messages"]:
            if not isinstance(item, dict):
                raise ValueError("Each leader assignment must be a JSON object")
            receiver_name = str(item.get("receiver") or "").strip()
            content = str(item.get("content") or "").strip()
            event_type = str(item.get("type") or "").strip() or "assignment"
            receiver_id = self._resolve_agent_name_to_id(receiver_name)
            if receiver_id is None:
                raise ValueError(f"Unknown assignment receiver: {receiver_name}")
            if receiver_id not in expected_agent_ids:
                raise ValueError(f"Leader assignment receiver is not in assignee list: {receiver_name}")
            if receiver_id in seen_agent_ids:
                raise ValueError(f"Duplicate assignment for agent: {receiver_name}")
            if event_type != "assignment":
                raise ValueError(f"Leader message type must be assignment, got: {event_type}")
            seen_agent_ids.add(receiver_id)
            assignment_events.append(
                {
                    "receiver_id": receiver_id,
                    "receiver_name": receiver_name,
                    "content": content,
                    "event_type": event_type,
                }
            )
        for target_id in sorted(expected_agent_ids - seen_agent_ids):
            assignment_events.append(
                {
                    "receiver_id": target_id,
                    "receiver_name": self.all_agent_names[target_id],
                    "content": "",
                    "event_type": "assignment",
                }
            )
        log_print(
            step=step,
            agent_id=agent_id,
            structured={
                "leader_thinking": parsed.get("thinking") or "",
                "leader_assignments": assignment_events,
            },
        )
        return {
            "assignments": assignment_events,
            "raw_output": answer,
            "thinking": parsed.get("thinking") or "",
        }

    def plan_action_and_memory(self, obs, complete_subgoals=None, step=0, agent_id=0, feedback=None, extra_dialogue=None):
        prompt_context = self._build_turn_prompt_context(
            obs,
            complete_subgoals=complete_subgoals,
            feedback=feedback,
            extra_dialogue=extra_dialogue,
        )
        action_memory_prompt = prompt_transfer.build_action_memory_prompt(
            single_agent=self.single_agent,
            context=prompt_context,
            restriction_context=self.prompt_restriction_context,
            no_location_mode=getattr(self, 'no_location_mode', False),
            shared_memory=self.shared_memory,
        )
        answer = self._run_prompt(action_memory_prompt, obs=obs, use_images=True)
        log_print(answer)
        parsed_output = parse_output_blocks(answer)

        action_text = parsed_output.get("action")
        self.action_str = action_text or ""
        if action_text is None:
            raise ValueError(f"Failed to parse action from model output: {answer}")

        memory_text = parsed_output.get("memory")
        if memory_text is None:
            raise ValueError(f"Failed to parse memory from model output: {answer}")
        if memory_text.startswith("The updated memory:"):
            memory_text = memory_text[len("The updated memory:"):].strip()

        decision = ActionMemoryDecision(
            action=action_text,
            memory=memory_text,
            raw_output=answer,
            action_thinking=parsed_output.get("action_thinking") or "",
            memory_thinking=parsed_output.get("memory_thinking") or "",
        )
        self.memory = decision.memory
        log_print(
            step=step,
            agent_id=agent_id,
            structured={
                "raw_output": decision.raw_output,
                "action_thinking": decision.action_thinking,
                "memory_thinking": decision.memory_thinking,
                "memory": self.memory or "",
            },
        )
        try:
            action, ids, _ = self.decode_action(
                decision.action,
                obs,
                step=step,
                agent_id=agent_id,
            )
        except Exception as exc:
            raise ActionDecodeError(str(exc), memory=self.memory) from exc
        return action, ids

    def update(self, new_script):
        new_clean = re.sub(r"^<char\d+>\s*", "", new_script)
        self.action_history.append(new_clean)
