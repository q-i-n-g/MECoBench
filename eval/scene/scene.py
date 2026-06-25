import os
from virtualhome.simulation.unity_simulator import UnityCommunication
import pickle
import random
import json
import numpy as np
from eval.scene.utils import *
import cv2
from eval.scene.act_module import ActModule
from eval.utils.logger import log_print

ACTION_MAP ={"walk": "walk", "open": "open", "close": "close", "switch_on": "switchon", "switch_off": "switchoff", "sit": "sit", "standup": "standup", "turn_left": "turnleft", "turn_right": "turnright", "walk_forward": "walkforward", "talk": "talk", "walk_to_room": "walk_to_room",'put_in': 'putin', 'put_on': 'putback', 'grab': 'grab', 'squat_down': 'squatdown', 'wait': 'wait', 'handover': 'handover'}

class Scene:
    def __init__(self, comm:UnityCommunication, seed=123, view_mode='multi_view'):
        """
        Args:
            comm: Unity 通信对象
            seed: 随机种子
            view_mode: 视角模式
                - 'first_person': 只使用第一人称视角（前视图）
                - 'multi_view': 使用多视角（前、后、左、右）
        """
        self.comm = comm
        self.seed = seed
        self.view_mode = view_mode
        self.image_width = 256
        self.image_height = 256
        self.horizontal_field_view = 90
        self.vertical_field_view = 90
        self.base_camera_per_agent = 8
        self.extra_camera_order = ["front", "down", "back", "left", "right"]
        self.camera_types = ["adult"]
        self.camera_offsets = {
            "adult": {
                name: self.base_camera_per_agent + idx
                for idx, name in enumerate(self.extra_camera_order)
            }
        }
        self.num_camera_per_agent = self.base_camera_per_agent + len(self.extra_camera_order)
        self.view_names = ["front", "back", "left", "right"]
        self.CAMERA_NUM = self.camera_offsets["adult"]["front"]
        self.record_dir = 'outputs/test_outputs'
        self.act_module = ActModule(comm, seed)
        self.adult_camera_height = 1.8
        self.child_camera_height = 0.9
        self.agent_camera_types = {}
        self.room_constraints = {}
        self.constraints_enabled = False
        self.allowed_room_ids_by_agent = {}

    def _normalize_agent_type(self, agent_type):
        if agent_type is None:
            return "adult"
        agent_type = str(agent_type).strip().lower()
        return agent_type if agent_type in {"adult", "child"} else "adult"

    def _add_character_camera_set(self, height, suffix):
        name_suffix = f"_{suffix}" if suffix else ""
        self.comm.add_character_camera(position=[0, height, 0.15], rotation=[30, 0, 0], field_view=90, name=f"up_camera{name_suffix}")
        self.comm.add_character_camera(position=[0, 0.5, 0.15], rotation=[0, 0, 0], field_view=90, name=f"down_camera{name_suffix}")
        self.comm.add_character_camera(position=[0, height, -0.15], rotation=[30, 180, 0], field_view=90, name=f"back_camera{name_suffix}")
        self.comm.add_character_camera(position=[-0.15, height, 0], rotation=[30, 270, 0], field_view=90, name=f"left_camera{name_suffix}")
        self.comm.add_character_camera(position=[0.15, height, 0], rotation=[30, 90, 0], field_view=90, name=f"right_camera{name_suffix}")

    def _setup_character_cameras(self, profile):
        self.agent_camera_types = {}
        types_present = set()
        for agent_id, agent in profile.items():
            agent_type = self._normalize_agent_type(agent.get("type"))
            self.agent_camera_types[int(agent_id)] = agent_type
            types_present.add(agent_type)

        self.camera_types = [t for t in ["adult", "child"] if t in types_present]
        if not self.camera_types:
            self.camera_types = ["adult"]

        self.camera_offsets = {}
        offset_base = self.base_camera_per_agent
        for cam_type in self.camera_types:
            self.camera_offsets[cam_type] = {
                name: offset_base + idx
                for idx, name in enumerate(self.extra_camera_order)
            }
            offset_base += len(self.extra_camera_order)

        self.num_camera_per_agent = offset_base
        default_type = "adult" if "adult" in self.camera_offsets else self.camera_types[0]
        self.CAMERA_NUM = self.camera_offsets[default_type]["front"]

        for cam_type in self.camera_types:
            height = self.child_camera_height if cam_type == "child" else self.adult_camera_height
            self._add_character_camera_set(height, cam_type)

    def _extract_allowed_room_ids(self):
        self.allowed_room_ids_by_agent = {}
        agents_cfg = self.room_constraints.get('agents') or {}
        for agent_id_str, cfg in agents_cfg.items():
            try:
                agent_id = int(agent_id_str)
            except (TypeError, ValueError):
                continue
            room_ids = set()
            for room_id in cfg.get('allowed_room_ids', []):
                try:
                    room_ids.add(int(room_id))
                except (TypeError, ValueError):
                    continue
            self.allowed_room_ids_by_agent[agent_id] = room_ids

    def _add_character(self, character_resource, initial_room, allowed_room_ids=None):
        kwargs = {"initial_room": initial_room}
        if allowed_room_ids:
            kwargs["allowed_rooms"] = [str(room_id) for room_id in allowed_room_ids]
        return self.comm.add_character(character_resource, **kwargs)

    def initialize(self, task, profile):
        self.task = task

        self.init_rooms = task['init_rooms']
        self.task_goal = task['task_goal']
        self.goal_class = task['goal_class']
        self.task_name = task['task_name']
        self.room_constraints = task.get('room_constraints') or {}
        self.constraints_enabled = bool(self.room_constraints.get('enabled', False))
        self.obj_list = get_obj_list(task)
        self.comm.reset(self.task['env_id'])
        self.comm.expand_scene(self.task['init_graph'], random_seed=self.seed)
        self._extract_allowed_room_ids()

        self.num_static_cameras = self.comm.camera_count()[1]
        # s, g = self.comm.add_character_camera(position=[0, 1.8, 0.15], rotation=[30,0,90], field_view=90, name="up_camera", follow_head=True)
        self._setup_character_cameras(profile)
        self.agents_ids = []
        for agent_id, agent in profile.items():
            agent_idx = int(agent_id)
            if agent_idx >= len(self.init_rooms):
                raise ValueError(
                    f"init_rooms does not provide an initial room for agent {agent_idx}."
                )
            initial_room = self.init_rooms[agent_idx]
            allowed_room_ids = sorted(self.allowed_room_ids_by_agent.get(agent_idx, set()))
            self._add_character(agent['3d_model'], initial_room=initial_room, allowed_room_ids=allowed_room_ids)
            self.agents_ids.append(int(agent_id))

        self.curr_graph = self.comm.environment_graph()[1]
        self.id2nodes = {node['id']: node for node in self.curr_graph['nodes']}
        self.cache_id_map = dict()
        
        # 获取 character 节点的真实 ID 映射 (agent_id -> char_node_id)
        char_nodes = [node for node in self.curr_graph['nodes'] if node['class_name'] == 'character']
        char_nodes.sort(key=lambda x: x['id'])  # 按 ID 排序
        self.agent_to_char_id = {agent_id: char_nodes[i]['id'] for i, agent_id in enumerate(self.agents_ids)}
    

    def get_camera_id(self, agent_id, view_name):
        if view_name not in self.extra_camera_order:
            raise ValueError(f"Unknown view_name: {view_name}")
        agent_type = self.agent_camera_types.get(agent_id, "adult")
        if agent_type not in self.camera_offsets:
            agent_type = "adult" if "adult" in self.camera_offsets else self.camera_types[0]
        return self.num_static_cameras + agent_id * self.num_camera_per_agent + self.camera_offsets[agent_type][view_name]

    def get_view(self, agent_id, view_name="front", step=0, save=False):
        camera_ids = [self.get_camera_id(agent_id, view_name)]
        s, bgr_images = self.comm.camera_image(camera_ids, mode='normal', image_width=self.image_width, image_height=self.image_height, horizontal_field_view=self.horizontal_field_view, vertical_field_view=self.vertical_field_view)
        if save:
            cv2.imwrite(os.path.join(self.record_dir, f'{view_name}_view_{agent_id}_{step}.png'), bgr_images[0])
        return bgr_images[0]

    def get_front_view(self, agent_id, step=0):
        return self.get_view(agent_id, "front", step)
    
    def get_img_bbox(self, agent_id, img, id_img, visible_objects, location, view_name="front", step=0, save=False):
        label_img = label_img_with_bbox(img, id_img, visible_objects, location, obj_list=self.obj_list)
        if save:
            cv2.imwrite(os.path.join(self.record_dir, f'label_img_{view_name}_{agent_id}_{step}.png'), label_img)
        return label_img

    def get_overall_view(self, step=0):
        camera_ids = [self.num_static_cameras - 1]
        s, bgr_images = self.comm.camera_image(camera_ids, mode='normal', image_width=self.image_width, image_height=self.image_height)
        cv2.imwrite(os.path.join(self.record_dir, f'overall_view_{step}.png'), bgr_images[0])
        return bgr_images[0]
        
    def get_seg_objects(self, agent_id, view_name="front", step=0):
        camera_ids = [self.get_camera_id(agent_id, view_name)]
        s, seg_img = self.comm.camera_image(camera_ids, mode='seg_inst', image_width=self.image_width, image_height=self.image_height, horizontal_field_view=self.horizontal_field_view, vertical_field_view=self.vertical_field_view)

        s, id_rgb = self.comm.instance_colors()
        background_ids = get_id(self.curr_graph, ['wall','floor','ceiling','bathroom','kitchen','livingroom','bedroom'])
        def find_id_by_colors(rgb):
            if rgb in self.cache_id_map.keys():
                return self.cache_id_map[rgb]
            ans = []
            for k, v in id_rgb.items():
                if abs(rgb[0] - v[2] * 255) + abs(rgb[1] - v[1] * 255) + abs(rgb[2] - v[0] * 255) < 10 and int(k) not in background_ids:
                    self.cache_id_map[rgb] = int(k)
                    return int(k)
            self.cache_id_map[rgb] = -1
            return -1

        id_img = np.zeros((self.image_height, self.image_width))
        for i in range(self.image_height):
            for j in range(self.image_width):
                id = find_id_by_colors(tuple(seg_img[0][i, j]))
                id_img[i][j] = id
        colorids = np.stack(((id_img % 10) * 10, ((id_img // 10) % 10) * 10, ((id_img // 100) % 10) * 10), axis=2)
        
        objs = np.unique(id_img)
        objs = [self.id2nodes[obj] for obj in objs if obj != -1]
        
        return id_img, objs
    
    def get_current_state(self, agent_id, step=0):
        char_node_id = self.agent_to_char_id[agent_id]
        hoding_obj = []
        for edge in self.curr_graph['edges']:
            if edge['from_id'] == char_node_id and edge['relation_type'] in ["HOLDS_RH", "HOLDS_LH"]:
                hoding_obj.append(self.id2nodes[edge['to_id']])
        location = self.id2nodes[char_node_id]['obj_transform']['position']
        current_room = which_room(self.curr_graph, char_node_id)
        return location, current_room, hoding_obj
    
    def get_observation(self, step=0, messages=[]):
        self.curr_graph = self.comm.environment_graph()[1]
        self.id2nodes = {node['id']: node for node in self.curr_graph['nodes']}
        observations = []
        for agent_id in self.agents_ids:
            location, current_room, hoding_obj = self.get_current_state(agent_id, step)
            
            # 计算除了当前智能体以外其他智能体正在holding的物品
            others_holding_obj = []
            for other_agent_id in self.agents_ids:
                if other_agent_id != agent_id:
                    other_char_node_id = self.agent_to_char_id[other_agent_id]
                    for edge in self.curr_graph['edges']:
                        if edge['from_id'] == other_char_node_id and edge['relation_type'] in ["HOLDS_RH", "HOLDS_LH"]:
                            others_holding_obj.append(self.id2nodes[edge['to_id']])
            
            active_views = ['front'] if self.view_mode == 'first_person' else list(self.view_names)
            views = {}
            for view_name in active_views:
                view_img = self.get_view(agent_id, view_name, step, save=False)
                id_img, objs = self.get_seg_objects(agent_id, view_name, step)
                label_img = self.get_img_bbox(agent_id, view_img, id_img, objs, location, view_name, step, save=False)
                views[view_name] = {
                    'rgb': view_img,
                    'id_img': id_img,
                    'visible_objects': objs,
                    'label_img': label_img,
                }
            label_views = [views[name]['label_img'] for name in active_views]
            raw_views = [views[name]['rgb'] for name in active_views]
            view_order = active_views

            if self.view_mode == 'first_person':
                cv2.imwrite(os.path.join(self.record_dir, f'label_img_front_{agent_id}_{step}.png'), label_views[0])
                cv2.imwrite(os.path.join(self.record_dir, f'raw_img_front_{agent_id}_{step}.png'), raw_views[0])
            else:
                cv2.imwrite(os.path.join(self.record_dir, f'raw_img_combined_{agent_id}_{step}.png'), np.hstack(raw_views))
            
            observations.append({
                'location': location,
                'current_room': current_room,
                'hoding_obj': hoding_obj,
                'messages': messages,
                'views': views,
                'view_order': view_order,
                'label_views': label_views,
                'raw_views': raw_views,
                'others_holding_obj': others_holding_obj,
                'record_dir': self.record_dir,
                'curr_graph': self.curr_graph,
                'constraints_enabled': self.constraints_enabled,
                'allowed_room_ids': sorted(list(self.allowed_room_ids_by_agent.get(agent_id, set()))),
            })

        _ = self.get_overall_view(step)
        return observations
    
    def get_script(self, action, ids, agent_id, message=None):
        if action == 'wait':
            return f"<char{agent_id}> [wait]"
        if action == 'walk_to_room':
            target_room_name = ids[0][0]
            room_candidates = [item['id'] for item in self.curr_graph['nodes'] if item['class_name'] == target_room_name and item.get('category') == 'Rooms']
            if not room_candidates:
                raise ValueError(f'No room candidate found for room {target_room_name}.')
            room_id = room_candidates[0]
            return f"<char{agent_id}> [walk] <{ids[0][0]}> ({room_id})"
        if action == 'handover':
            obj_id = int(ids[0][0])
            receiver_agent_id = int(ids[1][0])
            obj_node = self.id2nodes.get(obj_id)
            obj_name = obj_node['class_name'] if obj_node else 'object'
            return f"<char{agent_id}> [HandOver] <{obj_name}> ({obj_id}) <char{receiver_agent_id}>"
        action = ACTION_MAP[action]
        script = f'<char{agent_id}>'
        cadidate_nodes = []
        for id in ids:
            cadidate_node = [self.id2nodes[int(candidate_id)] for candidate_id in id]
            cadidate_nodes.append(cadidate_node)
        if action == 'talk':
            return f'<char{agent_id}> [talk] <{message}>'
        script += get_unity_script(action, cadidate_nodes)
        return script
    
    def act_step(self, action, ids, agent_id):
        if action == 'handover':
            script = self.get_script(action, ids, agent_id)
            s, m = self.comm.render_script([script], recording=False, skip_animation=True, random_seed=self.seed)
            return s, m
        action = ACTION_MAP[action]
        cadidate_nodes = []
        for id in ids:
            cadidate_node = [self.id2nodes[candidate_id] for candidate_id in id]
            cadidate_nodes.append(cadidate_node)
        s, m = self.act_module.DoAction(action, cadidate_nodes, agent_id)
        return s, m
        
    def step(self, script:list):
        filtered_scripts = []
        for item in script:
            if not item:
                continue
            if '[wait]' in item:
                continue
            filtered_scripts.append(item)
        if not filtered_scripts:
            log_print("No-op step (all agents waiting).")
            return True, {"message": "wait"}
        # script_str = '|'.join(filtered_scripts)
        s, m = self.comm.render_script(filtered_scripts, recording=False, skip_animation=True, random_seed=self.seed)        
        log_print(m)
        return s, m
