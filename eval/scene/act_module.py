class ActModule:
    def __init__(self, comm, seed=123):
        self.comm = comm
        self.seed = seed

    def _render(self, script):
        s, m = self.comm.render_script(
            [script], recording=False, skip_animation=True, random_seed=self.seed
        )
        if not s:
            raise Exception(m)
        return s, m

    def DoWalk(self, obj_node, char_id):
        for node in obj_node:
            script = f'<char{char_id}> [walk] <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoWalkForward(self, char_id):
        script = f'<char{char_id}> [walkforward]'
        return self._render(script)

    def DoStandUp(self, char_id):
        script = f'<char{char_id}> [standup]'
        return self._render(script)

    def DoTurnLeft(self, char_id):
        script = f'<char{char_id}> [turnleft]:60:'
        return self._render(script)

    def DoTurnRight(self, char_id):
        script = f'<char{char_id}> [turnright]:60:'
        return self._render(script)

    def DoSit(self, obj_node, char_id):
        for node in obj_node:
            if 'SITTABLE' in node['properties']:
                script = f'<char{char_id}> [sit] <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoSwitchOn(self, obj_node, char_id):
        for node in obj_node:
            if 'HAS_SWITCH' in node['properties'] and 'OFF' in node['states']:
                script = f'<char{char_id}> [switchon] <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoSwitchOff(self, obj_node, char_id):
        for node in obj_node:
            if 'HAS_SWITCH' in node['properties'] and 'ON' in node['states']:
                script = f'<char{char_id}> [switchoff] <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoOpen(self, obj_node, char_id):
        for node in obj_node:
            if 'CAN_OPEN' in node['properties'] and 'CLOSED' in node['states']:
                script = f'<char{char_id}> [open] <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoClose(self, obj_node, char_id):
        for node in obj_node:
            if 'CAN_OPEN' in node['properties'] and 'OPEN' in node['states']:
                script = f'<char{char_id}> [close] <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoGrab(self, obj_node, char_id):
        for node in obj_node:
            if 'GRABBABLE' in node['properties'] or 'DRINKABLE' in node['properties']:
                script = f'<char{char_id}> [grab] <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoPutIn(self, obj_node, container_node, char_id):
        """Put an object into a container."""
        for node in container_node:
            if 'CONTAINERS' in node['properties']:
                script = f'<char{char_id}> [putin] <{obj_node[0]["class_name"]}> ({obj_node[0]["id"]}) <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoPutOn(self, obj_node, surface_node, char_id):
        """Put an object on a surface (uses putback in unity script)."""
        for node in surface_node:
            if 'SURFACES' in node['properties']:
                script = f'<char{char_id}> [putback] <{obj_node[0]["class_name"]}> ({obj_node[0]["id"]}) <{node["class_name"]}> ({node["id"]})'
        return self._render(script)

    def DoSquatDown(self, char_id):
        return True, {'message': 'Squat down successfully'}

    def DoWait(self, char_id):
        return True, {'message': 'Wait'}
    
    def DoWalkToRoom(self, room_node, char_id):
        script = f'<char{char_id}> [walk] <{room_node["class_name"]}> ({room_node["id"]})'
        return self._render(script)

    def DoAction(self, action, candidate_nodes, char_id):
        """Dispatch action string to the corresponding primitive executor."""
        action = action.lower()
        if action == 'walk':
            return self.DoWalk(candidate_nodes[0], char_id)
        if action == 'walk_to_room':
            return self.DoWalkToRoom(candidate_nodes[0], char_id)
        if action == 'walkforward':
            return self.DoWalkForward(char_id)
        if action == 'standup':
            return self.DoStandUp(char_id)
        if action == 'turnleft':
            return self.DoTurnLeft(char_id)
        if action == 'turnright':
            return self.DoTurnRight(char_id)
        if action == 'sit':
            return self.DoSit(candidate_nodes[0], char_id)
        if action == 'switchon':
            return self.DoSwitchOn(candidate_nodes[0], char_id)
        if action == 'switchoff':
            return self.DoSwitchOff(candidate_nodes[0], char_id)
        if action == 'open':
            return self.DoOpen(candidate_nodes[0], char_id)
        if action == 'close':
            return self.DoClose(candidate_nodes[0], char_id)
        if action == 'grab':
            return self.DoGrab(candidate_nodes[0], char_id)
        if action == 'putin':
            return self.DoPutIn(candidate_nodes[0], candidate_nodes[1], char_id)
        if action in ['putback', 'puton']:
            return self.DoPutOn(candidate_nodes[0], candidate_nodes[1], char_id)
        if action == 'squatdown':
            return self.DoSquatDown(char_id)
        if action == 'wait':
            return self.DoWait(char_id)
        raise ValueError(f"Unsupported action: {action}")
