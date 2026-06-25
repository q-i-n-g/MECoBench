import os

import cv2

from .schemas import CommEvent, CommPhaseResult
from .utils import build_labeled_view_composite
from eval.utils.logger import log_print


class BaseCommunicationProtocol:
    """Generate one round of sequential broadcast messages before action planning."""

    name = "base"

    def _format_receiver_label(self, event, agents):
        if event.receiver == "all":
            return "all agents"
        if isinstance(event.receiver, int):
            return agents[event.receiver].name
        if isinstance(event.receiver, (list, tuple, set)):
            labels = []
            for receiver_id in event.receiver:
                labels.append(agents[receiver_id].name)
            return ", ".join(labels)
        return str(event.receiver)

    def _format_dialogue_content(self, event, agents, viewer_agent_id=None):
        receiver_label = self._format_receiver_label(event, agents)
        if event.event_type in {"message", "discussion"}:
            return f"to {receiver_label}: {event.content}"
        content = f"to {receiver_label} ({event.event_type}): {event.content}"
        if (
            event.event_type == "assignment"
            and isinstance(event.receiver, int)
            and viewer_agent_id == event.receiver
            and event.sender != event.receiver
        ):
            content += " You must follow the leader's (Agent 0) instruction."
        return content

    def _build_visible_dialogue(self, agent_id, agents, events):
        visible_dialogue = []
        for event in events:
            if not event.is_visible_to(agent_id):
                continue
            visible_dialogue.append(
                event.as_dialogue(
                    agents[event.sender].name,
                    self._format_dialogue_content(event, agents, viewer_agent_id=agent_id),
                )
            )
        return visible_dialogue

    def _build_action_dialogue_per_agent(self, agents, transcript):
        return {
            agent_id: self._build_visible_dialogue(agent_id, agents, transcript)
            for agent_id in range(len(agents))
        }

    def _finalize_result(self, agents, transcript, round_transcripts, assignments=None, stop_reason=None):
        action_dialogue_per_agent = self._build_action_dialogue_per_agent(agents, transcript)
        inbox_per_agent = {
            agent_id: list(dialogue)
            for agent_id, dialogue in action_dialogue_per_agent.items()
        }
        if assignments is None:
            assignments = {agent_id: None for agent_id in range(len(agents))}
        return CommPhaseResult(
            transcript=transcript,
            round_transcripts=round_transcripts,
            inbox_per_agent=inbox_per_agent,
            action_dialogue_per_agent=action_dialogue_per_agent,
            assignments=assignments,
            stop_reason=stop_reason,
        )

    def run_phase(
        self,
        agents,
        obs,
        complete_subgoals,
        step,
        feedback,
        last_round_messages,
    ):
        transcript = []
        current_round = []

        for agent_id, agent in enumerate(agents):
            visible_messages = self._build_visible_dialogue(
                agent_id,
                agents,
                transcript,
            )
            try:
                message = agent.plan_message(
                    obs[agent_id],
                    complete_subgoals,
                    step=step,
                    agent_id=agent_id,
                    feedback=feedback[agent_id],
                    extra_dialogue=visible_messages,
                )
            except Exception as exc:
                log_print(exc, step=step, agent_id=agent_id)
                continue
            if message is None:
                continue
            event = CommEvent(
                sender=agent_id,
                receiver="all",
                content=message,
                event_type="message",
            )
            transcript.append(event)
            current_round.append(event)

        return self._finalize_result(
            agents,
            transcript=transcript,
            round_transcripts=[current_round],
            stop_reason="completed",
        )


class NoCommProtocol(BaseCommunicationProtocol):
    name = "no_comm"

    def run_phase(
        self,
        agents,
        obs,
        complete_subgoals,
        step,
        feedback,
        last_round_messages,
    ):
        empty_dialogue = {agent_id: [] for agent_id in range(len(agents))}
        return CommPhaseResult(
            transcript=[],
            round_transcripts=[],
            inbox_per_agent=empty_dialogue,
            action_dialogue_per_agent=empty_dialogue,
            assignments={agent_id: None for agent_id in range(len(agents))},
            stop_reason="no_comm",
        )


class DiscussThenActProtocol(BaseCommunicationProtocol):
    name = "discuss_then_act"

    def __init__(self, rounds):
        if rounds < 1:
            raise ValueError("discussion_rounds must be >= 1")
        self.rounds = rounds

    def run_phase(
        self,
        agents,
        obs,
        complete_subgoals,
        step,
        feedback,
        last_round_messages,
    ):
        transcript = []
        round_transcripts = []
        local_context_per_agent = {}
        stop_reason = "max_rounds"

        for round_idx in range(self.rounds):
            current_round = []
            all_ready = True
            for agent_id, agent in enumerate(agents):
                visible_messages = self._build_visible_dialogue(
                    agent_id,
                    agents,
                    transcript + current_round,
                )
                try:
                    if round_idx == 0:
                        result = agent.plan_discussion_first_round(
                            obs[agent_id],
                            complete_subgoals,
                            step=step,
                            agent_id=agent_id,
                            feedback=feedback[agent_id],
                            extra_dialogue=visible_messages,
                        )
                        local_context_per_agent[agent_id] = result.get("local_context") or "None."
                    else:
                        result = agent.plan_discussion_text_round(
                            obs[agent_id],
                            complete_subgoals,
                            step=step,
                            agent_id=agent_id,
                            feedback=feedback[agent_id],
                            extra_dialogue=visible_messages,
                            local_context=local_context_per_agent.get(agent_id, "None."),
                        )
                except Exception as exc:
                    log_print(exc, step=step, agent_id=agent_id)
                    result = {"ready": False, "message": None}

                if not result.get("ready", False):
                    all_ready = False
                if result.get("message") is None:
                    continue

                event = CommEvent(
                    sender=agent_id,
                    receiver="all",
                    content=result["message"],
                    event_type="discussion",
                )
                current_round.append(event)

            transcript.extend(current_round)
            round_transcripts.append(current_round)
            if all_ready:
                stop_reason = "all_ready"
                break

        return self._finalize_result(
            agents,
            transcript=transcript,
            round_transcripts=round_transcripts,
            stop_reason=stop_reason,
        )


class SharedMemoryProtocol(BaseCommunicationProtocol):
    """Implicit communication via shared memory — no explicit messages exchanged.

    All agents read/write the same memory JSON. After each agent plans its
    action and updates the memory, the next agent immediately sees the update.
    Coordination happens through dedicated fields in the shared memory
    (agent_plans, agent_status, coordination_board).
    """

    name = "shared_memory"

    def __init__(self):
        self._shared_memory = None 
        self._agents = None  

    def run_phase(
        self,
        agents,
        obs,
        complete_subgoals,
        step,
        feedback,
        last_round_messages,
    ):
        self._agents = agents
        if self._shared_memory is None:
            self._shared_memory = agents[0].memory
        for agent in agents:
            agent.memory = self._shared_memory

        empty_dialogue = {agent_id: [] for agent_id in range(len(agents))}
        return CommPhaseResult(
            transcript=[],
            round_transcripts=[],
            inbox_per_agent=empty_dialogue,
            action_dialogue_per_agent=empty_dialogue,
            assignments={agent_id: None for agent_id in range(len(agents))},
            stop_reason="shared_memory",
        )

    def sync_after_action(self, agent):
        """Called after an agent plans its action and updates memory.
        Propagates the updated memory string to the shared store AND to every
        peer agent, so the next agent to plan within this step reads the
        latest state (agent.memory is a string, so rebinding one agent's
        attribute does not update the others)."""
        self._shared_memory = agent.memory
        if self._agents is not None:
            for peer in self._agents:
                peer.memory = self._shared_memory


class LeaderWorkerProtocol(BaseCommunicationProtocol):
    name = "leader_worker"

    VALID_COMM_MODES = ("text_only", "text_and_image", "image_only")

    def __init__(self, leader_agent_id=0, leader_comm_mode="text_only"):
        self.leader_agent_id = leader_agent_id
        if leader_comm_mode not in self.VALID_COMM_MODES:
            raise ValueError(
                f"leader_comm_mode must be one of {self.VALID_COMM_MODES}, got: {leader_comm_mode}"
            )
        self.leader_comm_mode = leader_comm_mode

    def run_phase(
        self,
        agents,
        obs,
        complete_subgoals,
        step,
        feedback,
        last_round_messages,
    ):
        if self.leader_agent_id < 0 or self.leader_agent_id >= len(agents):
            raise ValueError(f"Invalid leader_agent_id: {self.leader_agent_id}")

        transcript = []
        round_transcripts = []
        assignments = {agent_id: None for agent_id in range(len(agents))}
        leader = agents[self.leader_agent_id]
        worker_ids = [agent_id for agent_id in range(len(agents)) if agent_id != self.leader_agent_id]
        assignee_ids = list(range(len(agents)))

        report_events = []
        if self.leader_comm_mode != "image_only":
            for worker_id in worker_ids:
                try:
                    result = agents[worker_id].plan_worker_report(
                        obs[worker_id],
                        leader_name=leader.name,
                        complete_subgoals=complete_subgoals,
                        step=step,
                        agent_id=worker_id,
                        feedback=feedback[worker_id],
                        extra_dialogue=[],
                    )
                except Exception as exc:
                    log_print(exc, step=step, agent_id=worker_id)
                    continue
                if result.get("message") is None:
                    continue
                report_events.append(
                    CommEvent(
                        sender=worker_id,
                        receiver=self.leader_agent_id,
                        content=result["message"],
                        event_type="report",
                    )
                )
        transcript.extend(report_events)
        round_transcripts.append(report_events)

        assignment_events = []
        leader_visible = self._build_visible_dialogue(
            self.leader_agent_id,
            agents,
            transcript,
        )
        composite_image = None
        if self.leader_comm_mode in ("text_and_image", "image_only"):
            try:
                leader_obs = obs[self.leader_agent_id]
                composite_groups = [
                    {
                        "label": f"Leader: {leader.name} (agent {self.leader_agent_id}) self view",
                        "obs": leader_obs,
                    }
                ]
                for worker_id in worker_ids:
                    composite_groups.append(
                        {
                            "label": f"{agents[worker_id].name} (agent {worker_id}) current view",
                            "obs": obs[worker_id],
                        }
                    )
                composite_image = build_labeled_view_composite(
                    composite_groups,
                    leader.view_mode,
                )
                record_dir = (leader_obs or {}).get("record_dir")
                if composite_image is not None and record_dir:
                    os.makedirs(record_dir, exist_ok=True)
                    composite_path = os.path.join(
                        record_dir,
                        f"leader_comm_view_{step}.png",
                    )
                    cv2.imwrite(composite_path, composite_image)
                    log_print(
                        step=step,
                        agent_id=self.leader_agent_id,
                        structured={
                            "leader_comm_composite_path": composite_path,
                            "leader_comm_composite_agents": [
                                self.leader_agent_id,
                                *worker_ids,
                            ],
                        },
                    )
            except Exception as exc:
                log_print(exc, step=step, agent_id=self.leader_agent_id)
                composite_image = None
        try:
            result = leader.plan_leader_assignments(
                obs[self.leader_agent_id],
                assignee_agent_ids=assignee_ids,
                complete_subgoals=complete_subgoals,
                step=step,
                agent_id=self.leader_agent_id,
                feedback=feedback[self.leader_agent_id],
                extra_dialogue=leader_visible,
                composite_image=composite_image,
                leader_comm_mode=self.leader_comm_mode,
            )
        except Exception as exc:
            log_print(exc, step=step, agent_id=self.leader_agent_id)
            result = {
                "assignments": [
                    {"receiver_id": i, "content": ""}
                    for i in range(len(agents))
                ]
            }

        for item in result.get("assignments", []):
            receiver_id = item["receiver_id"]
            assignments[receiver_id] = item["content"]
            assignment_events.append(
                CommEvent(
                    sender=self.leader_agent_id,
                    receiver=receiver_id,
                    content=item["content"],
                    event_type="assignment",
                )
            )
        for agent_id in range(len(agents)):
            if assignments[agent_id] is None:
                assignments[agent_id] = ""
                assignment_events.append(
                    CommEvent(
                        sender=self.leader_agent_id,
                        receiver=agent_id,
                        content="",
                        event_type="assignment",
                    )
                )
        transcript.extend(assignment_events)
        round_transcripts.append(assignment_events)

        return self._finalize_result(
            agents,
            transcript=transcript,
            round_transcripts=round_transcripts,
            assignments=assignments,
            stop_reason="completed",
        )
