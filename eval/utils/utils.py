
def _infer_goal_target_options_from_task(task, subgoal):
    task_name = str(task.get('task_name', ''))
    if 'put_fridge' not in task_name:
        return None

    elements = subgoal.split('_')
    if len(elements) < 3:
        return None

    try:
        target_id = int(elements[2])
    except (TypeError, ValueError):
        return None

    init_graph = task.get('init_graph') or {}
    nodes = init_graph.get('nodes') or []
    id2nodes = {int(node['id']): node for node in nodes if 'id' in node}
    target_node = id2nodes.get(target_id)
    if target_node is None:
        return None

    target_class_name = str(target_node.get('class_name', ''))
    if 'fridge' not in target_class_name:
        return None

    candidate_ids = sorted([
        int(node['id']) for node in nodes
        if str(node.get('class_name', '')) == target_class_name
    ])
    return candidate_ids if len(candidate_ids) > 1 else None


def get_allowed_target_ids(task, subgoal):
    goal_target_options = task.get('goal_target_options') or {}
    if subgoal in goal_target_options:
        return [int(target_id) for target_id in goal_target_options[subgoal]]

    inferred_target_ids = _infer_goal_target_options_from_task(task, subgoal)
    if inferred_target_ids:
        return inferred_target_ids

    elements = subgoal.split('_')
    return [int(elements[2])]


def goal_test(graph, task):
    id2nodes = {node['id']: node for node in graph['nodes']}
    satisfied = {}
    unsatisfied = {}
    result = True
    for subgoal, num in task['task_goal']['0'].items():
        if num == 0:
            continue
        cnt = 0
        elements = subgoal.split('_')
        to_obj = id2nodes[int(elements[2])]['class_name']
        new_subgoal = f"{elements[0]}_{elements[1]}_{to_obj}_{elements[2]}"
        allowed_target_ids = set(get_allowed_target_ids(task, subgoal))
        for egde in graph['edges']:
            if egde['to_id'] not in allowed_target_ids:
                continue
            if egde['relation_type'].lower() == elements[0] and id2nodes[egde['from_id']]['class_name'] == elements[1]:
                cnt += 1
            if cnt == num:
                satisfied[new_subgoal] = cnt
                unsatisfied[new_subgoal] = 0
                break
        if cnt > 0:
            satisfied[new_subgoal] = cnt
        unsatisfied[new_subgoal] = num - cnt
        if num - cnt > 0:
            result = False
    return satisfied, unsatisfied, result


def get_satisfied_object_ids(graph, task):
    id2nodes = {node['id']: node for node in graph['nodes']}
    satisfied_ids = set()
    for subgoal, num in task['task_goal']['0'].items():
        if num == 0:
            continue
        elements = subgoal.split('_')
        relation = elements[0]
        obj_class = elements[1]
        allowed_target_ids = set(get_allowed_target_ids(task, subgoal))
        for edge in graph['edges']:
            if edge['to_id'] not in allowed_target_ids:
                continue
            if edge['relation_type'].lower() == relation and id2nodes[edge['from_id']]['class_name'] == obj_class:
                satisfied_ids.add(edge['from_id'])
    return satisfied_ids


def resolve_action_conflicts(proposals, rng=None):
    """
    Resolve conflicts for single-object actions across agents.

    Args:
        proposals: list where each item is a dict with keys like 'action' and 'ids', or None.
        conflict_actions: optional set of action names that should be conflict-checked.
        rng: optional random-like object with a choice method (defaults to random module).

    Returns:
        allowed: list of bools indicating which proposals can execute.
        conflict_groups: dict mapping obj_id -> list of agent_ids that target it.
        winners: dict mapping obj_id -> chosen winner agent_id (only for conflicts).
    """
    import random as _random

    conflict_actions = {'open', 'close', 'grab', 'sit', 'switch_on', 'switch_off'}
    if rng is None:
        rng = _random

    allowed = [proposal is not None for proposal in proposals]
    conflict_groups = {}

    for agent_id, proposal in enumerate(proposals):
        if proposal is None:
            continue
        action = proposal.get('action')
        if action not in conflict_actions:
            continue
        ids = proposal.get('ids')
        try:
            obj_id = int(ids[0][0])
        except (TypeError, ValueError):
            continue
        conflict_groups.setdefault(obj_id, []).append(agent_id)

    winners = {}
    for obj_id, agent_ids in conflict_groups.items():
        if len(agent_ids) <= 1:
            continue
        winner = rng.choice(agent_ids)
        winners[obj_id] = winner
        for conflict_agent_id in agent_ids:
            if conflict_agent_id == winner:
                continue
            allowed[conflict_agent_id] = False

    return allowed, conflict_groups, winners
