import os

import yaml


curr_dir = os.path.dirname(os.path.abspath(__file__))


class PromptTransfer:
    def __init__(self):
        self.prompt_templates = self._load_prompt_templates()

    def _load_yaml(self, filename):
        with open(os.path.join(curr_dir, filename), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_prompt_templates(self):
        templates = self._load_yaml("prompt_new.yaml")
        return templates

    def _get_template(self, key):
        if key not in self.prompt_templates:
            raise KeyError(f"Missing prompt template: {key}")
        return self.prompt_templates[key]

    def _render(self, key, replacements=None):
        text = self._get_template(key)
        for placeholder, value in (replacements or {}).items():
            text = text.replace(f"${placeholder}$", "" if value is None else str(value))
        return text

    def _format_valid_actions(self, valid_actions):
        normalized = []
        for action in valid_actions or []:
            normalized.append(str(action).strip())
        return ", ".join(normalized) 

    def get_prompts(self):
        return {
            "ACTION_DESC_RESOLVE_PROMPT": self._get_template("ACTION_DESC_RESOLVE_PROMPT"),
        }

    def build_system_prompt(self, *, name, goal, single_agent, other_agents="none"):
        if single_agent:
            return self._render(
                "SYSTEM_PROMPT_SINGLE",
                {"NAME": name, "GOAL": goal},
            )
        return self._render(
            "SYSTEM_PROMPT_MULTI",
            {"NAME": name, "OTHER_NAMES": other_agents, "GOAL": goal},
        )

    def build_restriction_context(
        self,
        task_room_constraints,
        agent_id,
        all_agent_names,
        department_topology=None,
    ):
        task_room_constraints = task_room_constraints or {}
        if not task_room_constraints.get("enabled", False):
            return None

        agents_cfg = task_room_constraints.get("agents") or {}
        agent_cfg = agents_cfg.get(str(agent_id), {})
        allowed_room_names = []
        for room_name in agent_cfg.get("allowed_room_names") or []:
            room_name = str(room_name).strip().lower()
            if room_name and room_name not in allowed_room_names:
                allowed_room_names.append(room_name)

        outgoing_specs = []
        for spec in task_room_constraints.get("handover_chain") or []:
            try:
                from_agent = int(spec.get("from_agent"))
            except (TypeError, ValueError):
                continue
            if agent_id is not None and from_agent == int(agent_id):
                outgoing_specs.append(
                    {
                        "receiver": self._format_agent_ref(spec.get("to_agent"), all_agent_names),
                        "door_id": spec.get("door_id"),
                        "room_pair_ids": spec.get("room_pair_ids"),
                    }
                )

        return {
            "enabled": True,
            "allowed_room_names": allowed_room_names,
            "outgoing_specs": outgoing_specs,
            "department_topology": department_topology,
        }

    def build_communication_prompt(self, *, context, restriction_context=None, valid_actions=None):
        sections = [
            self._render("COMM_BLOCK_INTRO"),
            self._render(
                "PLAN_BLOCK_COMMUNICATION",
                {
                    "OTHER_AGENTS": context.get("other_agents", "none"),
                    "VALID_ACTIONS": self._format_valid_actions(valid_actions),
                },
            ),
            self._build_restricted_handover_coordination_block(restriction_context),
            self._render("COMM_BLOCK_OUTPUT"),
            self._build_inputs_block(
                include_historical_dialogue=True,
                include_current_round_dialogue=True,
                context=context,
                restriction_context=restriction_context,
            ),
        ]
        return self._join_sections(sections)

    def build_discussion_first_round_prompt(
        self,
        *,
        context,
        restriction_context=None,
        valid_actions=None,
    ):
        sections = [
            self._render("DISCUSS_FIRST_ROUND_BLOCK_INTRO"),
            self._render(
                "DISCUSS_BLOCK_RULES",
                {"VALID_ACTIONS": self._format_valid_actions(valid_actions)},
            ),
            self._build_restricted_handover_coordination_block(restriction_context),
            self._render("DISCUSS_FIRST_ROUND_BLOCK_OUTPUT"),
            self._build_inputs_block(
                include_historical_dialogue=True,
                include_current_round_dialogue=True,
                context=context,
                restriction_context=restriction_context,
            ),
        ]
        return self._join_sections(sections)

    def build_discussion_text_round_prompt(
        self,
        *,
        context,
        restriction_context=None,
        valid_actions=None,
    ):
        sections = [
            self._render("DISCUSS_TEXT_ROUND_BLOCK_INTRO"),
            self._render(
                "DISCUSS_BLOCK_RULES",
                {"VALID_ACTIONS": self._format_valid_actions(valid_actions)},
            ),
            self._build_restricted_handover_coordination_block(restriction_context),
            self._render("DISCUSS_TEXT_ROUND_BLOCK_OUTPUT"),
            self._build_inputs_block(
                include_historical_dialogue=True,
                include_current_round_dialogue=True,
                include_local_context=True,
                context=context,
                restriction_context=restriction_context,
            ),
        ]
        return self._join_sections(sections)

    def build_worker_report_prompt(self, *, leader_name, context, restriction_context=None, step=0):
        sections = [
            self._render("WORKER_REPORT_BLOCK_INTRO", {"LEADER_NAME": leader_name}),
            self._render("WORKER_REPORT_BLOCK_RULES"),
            self._build_restricted_worker_report_handover_hint(restriction_context),
            self._build_restricted_rooms_report_hint(restriction_context, step=step),
            self._build_restricted_handover_coordination_block(restriction_context),
            self._render("WORKER_REPORT_BLOCK_OUTPUT"),
            self._build_inputs_block(
                include_historical_dialogue=True,
                include_current_round_dialogue=True,
                context=context,
                restriction_context=restriction_context,
            ),
        ]
        return self._join_sections(sections)

    def build_leader_assignment_prompt(
        self,
        *,
        assignee_names=None,
        worker_names=None,
        context,
        restriction_context=None,
        valid_actions=None,
        leader_comm_mode='text_only',
    ):
        agent_names = assignee_names if assignee_names is not None else worker_names
        intro_key = {
            'text_only': "LEADER_ASSIGNMENT_BLOCK_INTRO",
            'text_and_image': "LEADER_ASSIGNMENT_BLOCK_INTRO_VISUAL",
            'image_only': "LEADER_ASSIGNMENT_BLOCK_INTRO_VISUAL_ONLY",
        }[leader_comm_mode]
        sections = [
            self._render(intro_key),
            self._render(
                "LEADER_ASSIGNMENT_BLOCK_RULES",
                {
                    "AGENT_NAMES": ", ".join(agent_names) if agent_names else "none",
                    "VALID_ACTIONS": self._format_valid_actions(valid_actions),
                },
            ),
            self._build_restricted_rooms_assignment_hint(restriction_context),
            self._build_restricted_handover_coordination_block(restriction_context),
            self._render("LEADER_ASSIGNMENT_BLOCK_OUTPUT"),
            self._build_inputs_block(
                include_historical_dialogue=True,
                include_current_round_dialogue=True,
                context=context,
                restriction_context=restriction_context,
            ),
        ]
        return self._join_sections(sections)

    def build_action_memory_prompt(self, *, single_agent, context, restriction_context=None, no_location_mode=False, shared_memory=False):
        intro_template_key = (
            "ACTION_MEMORY_BLOCK_INTRO_SHARED"
            if shared_memory
            else "ACTION_MEMORY_BLOCK_INTRO"
        )
        sections = [
            self._render(intro_template_key),
            self._build_physical_actions_block(
                restriction_context,
                other_agents=context.get("other_agents", "none"),
            ),
        ]
        if no_location_mode:
            sections.append(self._render("PLAN_BLOCK_ANTI_STAGNATION"))
        if shared_memory:
            memory_template_key = "PLAN_BLOCK_MEMORY_SHARED"
        elif single_agent:
            memory_template_key = "PLAN_BLOCK_MEMORY_SINGLE"
        elif restriction_context and restriction_context.get("enabled"):
            memory_template_key = "PLAN_BLOCK_MEMORY_MULTI_RESTRICTED"
        else:
            memory_template_key = "PLAN_BLOCK_MEMORY_MULTI"
        sections += [
            self._render(memory_template_key),
            self._render("ACTION_MEMORY_BLOCK_OUTPUT"),
            self._build_inputs_block(
                include_historical_dialogue=False,
                include_current_round_dialogue=not single_agent and not shared_memory,
                context=context,
                restriction_context=restriction_context,
            ),
        ]
        return self._join_sections(sections)

    def _build_inputs_block(
        self,
        *,
        include_historical_dialogue,
        include_current_round_dialogue,
        context,
        restriction_context,
        include_local_context=False,
    ):
        dialogue_section = ""
        if include_historical_dialogue:
            dialogue_section += self._render(
                "PLAN_INPUT_HISTORICAL_DIALOGUE_SECTION",
                {"HISTORICAL_DIALOGUE": context.get("historical_dialogue", "None.")},
            )
        if include_current_round_dialogue:
            dialogue_section += self._render(
                "PLAN_INPUT_CURRENT_ROUND_DIALOGUE_SECTION",
                {"CURRENT_ROUND_DIALOGUE": context.get("current_round_dialogue", "None.")},
            )

        local_context_section = ""
        if include_local_context:
            local_context_section = self._render(
                "PLAN_INPUT_LOCAL_CONTEXT_SECTION",
                {"LOCAL_CONTEXT": context.get("local_context", "None.")},
            )

        restriction_section = ""
        if restriction_context and restriction_context.get("enabled"):
            restriction_section = self._render(
                "PLAN_INPUT_RESTRICTION_SECTION",
                {
                    "ALLOWED_ROOMS": self._format_allowed_rooms(restriction_context),
                    "DEPARTMENT_TOPOLOGY": self._format_department_topology(
                        restriction_context
                    ),
                },
            )

        return self._render(
            "PLAN_BLOCK_INPUTS_BASE",
            {
                "MEMORY": context["memory"],
                "DIALOGUE_SECTION": dialogue_section + local_context_section,
                "ACTION_HISTORY": context["action_history"],
                "FEEDBACK": context["feedback"],
                "LAST_ACTION": context["last_action"],
                "RESTRICTION_SECTION": restriction_section,
                "CURRENT_ROOM": context["current_room"],
                "HOLDING_OBJECTS": context["holding_objects"],
                "TASK_GOAL": context["task_goal"],
                "TASK_PROGRESS": context["task_progress"],
            },
        )

    def _build_physical_actions_block(self, restriction_context, other_agents="none"):
        if restriction_context and restriction_context.get("enabled"):
            return self._render(
                "PLAN_BLOCK_PHYSICAL_ACTIONS_RESTRICTED",
                {
                    "ALLOWED_ROOMS": self._format_allowed_rooms_inline(
                        restriction_context.get("allowed_room_names") or []
                    ),
                    "OTHER_NAMES": other_agents or "none",
                },
            )
        return self._render(
            "PLAN_BLOCK_PHYSICAL_ACTIONS_UNRESTRICTED",
            {"ALLOWED_ROOMS": "kitchen, bedroom, livingroom, bathroom"},
        )

    def _build_restricted_handover_coordination_block(self, restriction_context):
        if restriction_context and restriction_context.get("enabled"):
            return self._render("COMM_BLOCK_RESTRICTED_HANDOVER")
        return ""

    def _build_restricted_worker_report_handover_hint(self, restriction_context):
        if restriction_context and restriction_context.get("enabled"):
            return self._render("WORKER_REPORT_RESTRICTED_HANDOVER_HINT")
        return ""

    def _build_restricted_rooms_report_hint(self, restriction_context, step=0):
        if restriction_context and restriction_context.get("enabled") and step == 0:
            return self._render("WORKER_REPORT_RESTRICTED_ROOMS_HINT")
        return ""

    def _build_restricted_rooms_assignment_hint(self, restriction_context):
        if restriction_context and restriction_context.get("enabled"):
            return self._render("LEADER_ASSIGNMENT_RESTRICTED_ROOMS_HINT")
        return ""

    def _format_allowed_rooms(self, restriction_context):
        room_names = restriction_context.get("allowed_room_names") or []
        if not room_names:
            return "- unknown"
        return "\n".join(f"- {room_name}" for room_name in room_names)

    def _format_allowed_rooms_inline(self, room_names):
        if not room_names:
            return "none"
        return ", ".join(str(room_name) for room_name in room_names)

    def _format_department_topology(self, restriction_context):
        topology = restriction_context.get("department_topology")
        if not topology:
            return "unknown"
        return str(topology)

    def _format_agent_ref(self, agent_id, all_agent_names):
        try:
            agent_idx = int(agent_id)
        except (TypeError, ValueError):
            return f"agent {agent_id}"
        if 0 <= agent_idx < len(all_agent_names or []):
            return f"{all_agent_names[agent_idx]} (agent {agent_idx})"
        return f"agent {agent_idx}"

    def _join_lines(self, lines):
        return "\n".join(lines) if lines else ""

    def _join_sections(self, sections):
        ordered_sections = [section for section in sections if section]
        if not ordered_sections:
            return ""
        prompt = ordered_sections[0]
        for section in ordered_sections[1:]:
            prompt += f"\n\n{'-' * 50}\n{section}"
        return prompt
