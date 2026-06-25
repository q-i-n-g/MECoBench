import re
import json
from dataclasses import dataclass

from .utils import strip_code_fence


def check_legality(action, objs, obs):
    valid_rooms = {'bedroom', 'livingroom', 'kitchen', 'bathroom'}
    if len(obs['hoding_obj']) == 2:
        if action in ['open', 'close', 'switch_on', 'switch_off', 'grab']:
            return False, "Your hands are busy, you should find a place to put down one of the objects you are holding to free your hands first."
    if action == 'grab' and len(obs['others_holding_obj']) >= 1:
        if objs[0] in [item['id'] for item in obs['others_holding_obj']]:
            return False, "The object is already being held by another agent, you can't grab it."
    if action == 'grab' and len(obs['hoding_obj']) >= 1:
        if objs[0] in [item['id'] for item in obs['hoding_obj']]:
            return False, "The object is already being held by you, you can't grab it again."
    if action in ['standup', 'turn_left', 'turn_right', 'walk_forward', 'squat_down']:
        return True, None
    if action in ['wait']:
        return len(objs) == 0, 'The action should involve no objects.'
    elif action in ['handover']:
        return len(objs) == 2, 'The action should involve exactly one object and one receiver agent.'
    elif action in ['walk_to_room']:
        return len(objs) == 1 and objs[0] in valid_rooms, 'The action should involve exactly one room from: bedroom, livingroom, kitchen, bathroom.'
    elif action in ['walk', 'open', 'close', 'switch_on', 'switch_off', 'sit', 'grab']:
        return len(objs) == 1, 'The action should involve exactly one object.'
    elif action in ['put_in', 'put_on']:
        return len(objs) == 2, 'The action should involve exactly two objects.'
    else:
        return False, 'The action is not in the valid actions list.'


def normalize_action_name(action_name):
    """Normalize action name aliases to canonical form."""
    if not action_name:
        return action_name
    name = action_name.strip()
    mapping = {
        "walk_to_object": "walk",
        "walk_to": "walk",
        "walkto": "walk",
        "put_inside": "put_in",
        "put_into": "put_in",
        "putin": "put_in",
        "puton": "put_on",
        "hand_over": "handover",
        "handout": "handover",
        "transfer": "handover",
        "waiting": "wait",
        "recieve": "wait",
        "receive": "wait",
    }
    return mapping.get(name, name)


def normalize_param(param):
    """Normalize action parameter to standard dict format."""
    if param is None:
        return None
    if isinstance(param, str):
        return {"name": param, "description": ""}
    if isinstance(param, dict):
        name = param.get("name") or param.get("class_name") or param.get("object") or ""
        desc = param.get("description") or param.get("desc") or ""
        return {"name": name, "description": desc}
    return None


def normalize_optional_int(value, field_name):
    """Normalize optional integer payload fields."""
    if value in (None, ""):
        return None
    return int(value)


def parse_action_json(action_text):
    """Parse action JSON from text, handling code fences."""
    cleaned = strip_code_fence(action_text)
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        raise ValueError(f"Failed to parse ACTION JSON: {exc}")
    if not isinstance(parsed, dict):
        raise ValueError(f"ACTION must be a JSON object, got: {type(parsed)}")
    return parsed


@dataclass
class ParsedAction:
    action_type: str
    param_1: dict | None
    param_2: dict | None
    room: str | None
    door_id: int | None


class ActionDecodeHelper:
    NO_PARAM_ACTIONS = {"turn_left", "turn_right", "walk_forward", "standup", "squat_down", "wait"}
    VALID_ROOMS = {"bedroom", "livingroom", "kitchen", "bathroom"}
    SINGLE_TARGET_ACTIONS = {"walk", "open", "close", "switch_on", "switch_off", "sit", "grab"}
    DOUBLE_TARGET_ACTIONS = {"put_in", "put_on"}

    def __init__(self, all_agent_names=None):
        self.agent_name_to_id = {
            str(name).lower(): idx
            for idx, name in enumerate(all_agent_names or [])
        }

    def parse_action(self, action_text):
        action_payload = parse_action_json(action_text)
        action_type = normalize_action_name(action_payload.get("action", ""))
        if not action_type:
            raise ValueError("Missing action in ACTION JSON")
        param_1 = normalize_param(action_payload.get("parameter_1") or action_payload.get("object"))
        door_id = normalize_optional_int(action_payload.get("door_id"), "door_id")
        if action_type == "walk" and door_id is not None:
            door_desc = ""
            if isinstance(param_1, dict):
                door_desc = param_1.get("description") or ""
            param_1 = {"name": "door", "description": door_desc}
        param_2 = normalize_param(action_payload.get("parameter_2") or action_payload.get("target"))
        room_raw = action_payload.get("room")
        if (
            action_type == "walk"
            and door_id is None
            and room_raw not in (None, "")
            and param_1 is None
            and param_2 is None
        ):
            action_type = "walk_to_room"
        return ParsedAction(
            action_type=action_type,
            param_1=param_1,
            param_2=param_2,
            room=room_raw,
            door_id=door_id,
        )

    def resolve_special_action(self, parsed_action, obs):
        action_type = parsed_action.action_type
        if action_type in self.NO_PARAM_ACTIONS:
            return action_type, []
        if action_type == "walk" and parsed_action.door_id is not None:
            return action_type, [self._resolve_walk_door_id(parsed_action, obs)]
        if action_type == "walk_to_room":
            return action_type, [self._resolve_room_name(parsed_action)]
        if action_type == "handover":
            return action_type, self._resolve_handover_ids(parsed_action.param_1, parsed_action.param_2, obs)
        return None

    def validate_resolved_ids(self, action_type, final_ids, candidate_ids, holding_ids):
        if action_type in self.DOUBLE_TARGET_ACTIONS:
            if len(final_ids) != 2:
                raise ValueError("put_in/put_on must resolve two ids")
            if final_ids[0] not in holding_ids:
                raise ValueError("First parameter must be a held object id")
            if final_ids[1] not in candidate_ids:
                raise ValueError("Second parameter must be a candidate object id")
            return
        if action_type in self.SINGLE_TARGET_ACTIONS:
            if len(final_ids) != 1:
                raise ValueError("Action must resolve exactly one id")
            if final_ids[0] not in candidate_ids:
                raise ValueError("Resolved id is not in candidate ids")

    def _resolve_room_name(self, parsed_action):
        room_name = parsed_action.room
        if not room_name and isinstance(parsed_action.param_1, dict):
            room_name = parsed_action.param_1.get("name")
        if not room_name:
            raise ValueError("walk_to_room requires a room name")
        room_name = str(room_name).strip().lower()
        if room_name not in self.VALID_ROOMS:
            raise ValueError(f"Invalid room name: {room_name}")
        return room_name

    def _resolve_walk_door_id(self, parsed_action, obs):
        door_id = parsed_action.door_id

        curr_graph = obs.get("curr_graph") or {}
        nodes = {
            int(node["id"]): node
            for node in curr_graph.get("nodes")
        }
        door_node = nodes.get(int(door_id))
        if door_node is None:
            raise ValueError(f"door_id {door_id} not found in current graph")
        if str(door_node.get("class_name", "")).strip().lower() not in ["door", "doorjamb"]:
            raise ValueError(f"door_id {door_id} does not refer to a door")
        return int(door_id)

    def _resolve_handover_ids(self, param_1, param_2, obs):
        holding_objects = obs.get('hoding_obj', []) or []
        selected_obj_id = self._resolve_handover_object_id(param_1, holding_objects)
        if selected_obj_id is None:
            raise ValueError("handover requires a valid object in parameter_1")

        receiver_agent_id = self._resolve_receiver_agent_id(param_2)
        if receiver_agent_id is None:
            raise ValueError("handover requires a valid receiver agent in parameter_2")
        return [selected_obj_id, int(receiver_agent_id)]

    def _resolve_handover_object_id(self, param_1, holding_objects):
        selected_obj_id = None
        if isinstance(param_1, dict):
            obj_id = param_1.get("id")
            try:
                if obj_id is not None:
                    selected_obj_id = int(obj_id)
            except (TypeError, ValueError):
                selected_obj_id = None
            target_name = (param_1.get("name") or "").strip().lower()
            if target_name:
                for item in holding_objects:
                    if item.get('class_name', '').lower() == target_name:
                        return int(item['id'])
        if selected_obj_id is not None:
            return selected_obj_id
        match = re.search(r'\((\d+)\)', str(param_1))
        if match:
            return int(match.group(1))
        return None

    def _resolve_receiver_agent_id(self, receiver_param):
        if receiver_param is None:
            return None
        if isinstance(receiver_param, dict):
            raw_name = receiver_param.get("name") or receiver_param.get("description") or ""
        else:
            raw_name = str(receiver_param)
        raw_name = str(raw_name).strip()
        if not raw_name:
            return None
        if raw_name.isdigit():
            return int(raw_name)
        lowered = raw_name.lower()
        if lowered in self.agent_name_to_id:
            return self.agent_name_to_id[lowered]
        for name, agent_id in self.agent_name_to_id.items():
            if name in lowered:
                return agent_id
        match = re.search(r'char\s*(\d+)', lowered)
        if match:
            return int(match.group(1))
        match = re.search(r'agent\s*(\d+)', lowered)
        if match:
            return int(match.group(1))
        return None
