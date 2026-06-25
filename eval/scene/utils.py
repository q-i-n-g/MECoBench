

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


def goal_for_vlm(goal_class:dict):
    goal = []
    for predicate, count in goal_class.items():
        if count > 0:
            predicate_name = predicate.split('_')
            subgoal = f"Find and put {count} {predicate_name[1]}{'s' if count > 1 else ''} {predicate_name[0]} {predicate_name[2]}"
            goal.append(subgoal)
    goal_str = ''
    for i, subgoal in enumerate(goal):
        goal_str += f"{i+1}. {subgoal}\n"
    return goal_str

def simple_goal_for_vlm(goal_class:dict):
    goal = []
    for predicate, count in goal_class.items():
        if count > 0:
            predicate_name = predicate.split('_')
            subgoal = f"Find and put {count} {predicate_name[1]}{'s' if count > 1 else ''} {predicate_name[0]} {predicate_name[2]}"
            goal.append(subgoal)
    return ';'.join(goal)

def objs_in_room(graph, room):
    id2nodes = {node['id']: node for node in graph['nodes']}
    objs = []
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if id2nodes[edge['to_id']]['class_name'] == room:
                objs.append(id2nodes[edge['from_id']]['id'])
    return objs

def which_room(graph, obj_id):
    id2nodes = {node['id']: node for node in graph['nodes']}
    for edge in graph['edges']:
        if edge['relation_type'] == 'INSIDE':
            if id2nodes[edge['from_id']]['id'] == obj_id and id2nodes[edge['to_id']]['category'] == 'Rooms':
                return id2nodes[edge['to_id']]['class_name']
    return None

def get_id(graph, obj_name:list):
    ids = []
    for node in graph['nodes']:
        if node['class_name'] in obj_name:
            ids.append(node['id'])
    return ids

def get_unity_script(action, candidate_nodes):
    if action == 'open':
        for node in candidate_nodes[0]:
            if 'CAN_OPEN' in node['properties'] and 'CLOSED' in node['states']:
                return f'[open] <{node["class_name"]}> ({node["id"]})'
        for node in candidate_nodes[0]:
            if 'CAN_OPEN' in node['properties'] and 'OPEN' in node['states']:
                raise Exception(f"Object {node['class_name']} is already open")
        raise Exception(f"The object is not a container so cannot be opened")
    if action == 'close':
        for node in candidate_nodes[0]:
            if 'CAN_OPEN' in node['properties'] and 'OPEN' in node['states']:
                return f'[close] <{node["class_name"]}> ({node["id"]})'
        for node in candidate_nodes[0]:
            if 'CAN_CLOSE' in node['properties'] and 'CLOSED' in node['states']:
                raise Exception(f"Object {node['class_name']} is already closed")
        raise Exception(f"The object is not a container so cannot be closed")
    if action == 'grab':
        for node in candidate_nodes[0]:
            if 'GRABBABLE' in node['properties'] or 'DRINKABLE' in node['properties']:
                return f'[grab] <{node["class_name"]}> ({node["id"]})'
        raise Exception(f"No grabbable object found, check the target object if is not visible and grabbable, or description is correct.")
    if action == 'putback':
        for node in candidate_nodes[1]:
            if 'SURFACES' in node['properties']: 
                return f'[putback]<{candidate_nodes[0][0]["class_name"]}> ({candidate_nodes[0][0]["id"]}) <{node["class_name"]}> ({node["id"]})'
    if action == 'putin':
        for node in candidate_nodes[1]:
            if 'CONTAINERS' in node['properties']: 
                return f'[putin]<{candidate_nodes[0][0]["class_name"]}> ({candidate_nodes[0][0]["id"]}) <{node["class_name"]}> ({node["id"]})'
    if action == 'walk':
        for node in candidate_nodes[0]:
            return f'[walk] <{node["class_name"]}> ({node["id"]})'
    if action in ['walkforward', 'standup']:
        return f'[{action}]'
    if action in ['turnleft', 'turnright']:
        return f'[{action}]:90:'
    if action == 'sit':
        for node in candidate_nodes[0]:
            if 'SITTABLE' in node['properties']:
                return f'[sit] <{node["class_name"]}> ({node["id"]})'
    if action == 'switchon':
        for node in candidate_nodes[0]:
            if 'HAS_SWITCH' in node['properties'] and 'OFF' in node['states']:
                return f'[switchon] <{node["class_name"]}> ({node["id"]})'
    if action == 'switchoff':
        for node in candidate_nodes[0]:
            if 'HAS_SWITCH' in node['properties'] and 'ON' in node['states']:
                return f'[switchoff] <{node["class_name"]}> ({node["id"]})'

    raise Exception(f"The action is not legal.")


def label_img_with_bbox(
    img,
    id_map,
    visible_objs: list,
    location,
    max_distance=5,
    obj_list=[],
    allowed_ids=None,
    skip_distance_area=False,
    skip_type_filter=False,
):
    """
    在RGB图像上根据ID map标注每个物品的bbox和id
    
    Args:
        img: RGB图像 (numpy array, shape: H x W x 3)
        id_map: 逐像素的ID map (numpy array, shape: H x W)，每个像素值表示对应物品的id
        visible_objs: 视野内能看到的物品的信息，格式为[{'id':...,'property':[...], 'class_name':'...'}]
        location: 人物当前位置 [x, y, z]
        max_distance: 最大水平距离阈值，超过此距离的物体不绘制bbox
        obj_list: 必须绘制的物体列表
        allowed_ids: 仅绘制这些id（为None时按默认规则筛选）
        skip_distance_area: 跳过距离/面积过滤
        skip_type_filter: 跳过类型过滤
    
    Returns:
        标注后的图像 (numpy array)
    """
    import numpy as np
    import cv2
    
    # 复制图像避免修改原图
    labeled_img = img.copy()
    
    # 创建visible_objs的id到对象信息的映射
    visible_objs_dict = {obj['id']: obj for obj in visible_objs}
    
    # 定义需要绘制的class_name列表
    target_class_names = {'bathroomcabinet', 'kitchencabinet', 'cabinet', 'fridge', 'stove', 'microwave'}
    
    # 计算物体与人物的水平距离（忽略y轴高度）
    def calc_horizontal_distance(obj):
        """计算物体中心与人物位置的水平距离"""
        if 'bounding_box' not in obj or 'center' not in obj['bounding_box']:
            return float('inf')
        obj_center = obj['bounding_box']['center']
        # 水平距离：使用x和z坐标（y是高度）
        dx = obj_center[0] - location[0]
        dz = obj_center[2] - location[2]
        return np.sqrt(dx * dx + dz * dz)
    
    # 判断物品类型是否需要绘制bbox（不考虑距离）
    def should_draw_bbox_by_type(obj_id):
        if obj_id not in visible_objs_dict:
            return False
        obj = visible_objs_dict[obj_id]
        properties = obj.get('properties', [])
        class_name = obj.get('class_name', '')
        
        if class_name in ['wallpictureframe', 'bench', 'rug']:
            return False
        # 检查property中是否包含GRABBABLE或SURFACES
        if 'GRABBABLE' in properties or 'SURFACES' in properties or 'DRINKABLE' in properties:
            return True
        # 检查class_name是否在目标列表中
        if class_name in target_class_names:
            return True
        
        return False
    
    # 判断是否通过距离/面积检查
    def passes_distance_area_check(obj_id, bbox_area, min_area=150, min_area_for_far=6000):
        """
        检查物体是否应该显示：
        - bbox面积太小的物体：不显示（无论距离远近）
        - 距离近且面积足够的物体：显示
        - 距离远但bbox面积很大的物体：也显示
        
        Args:
            obj_id: 物品id
            bbox_area: bbox面积（像素²）
            min_area: 最小面积阈值，低于此值不显示（默认300像素²）
            min_area_for_far: 远处物体的最小面积阈值（默认5000像素²）
        """
        # 面积太小，直接不显示
        if bbox_area < min_area:
            return False
        
        if obj_id not in visible_objs_dict:
            return False
        obj = visible_objs_dict[obj_id]
        distance = calc_horizontal_distance(obj)
        
        # 距离近，通过
        if distance <= max_distance:
            return True
        
        # 距离远，但面积足够大也通过
        if bbox_area >= min_area_for_far:
            return True
        
        return False
    
    allowed_ids_set = set(allowed_ids) if allowed_ids is not None else None
    
    # 获取所有唯一的物品id（排除背景id，通常为0或-1）
    unique_ids = np.unique(id_map)
    unique_ids = unique_ids[unique_ids > 0]  # 假设0或负数为背景
    if allowed_ids_set is not None:
        unique_ids = [obj_id for obj_id in unique_ids if int(obj_id) in allowed_ids_set]
    
    # 为每个物品id分配颜色（按id确定，避免多视角间颜色错位）
    def color_for_id(obj_id):
        rng = np.random.RandomState(int(obj_id) & 0xFFFFFFFF)
        return tuple(int(x) for x in rng.randint(0, 255, 3))

    colors = {}
    for obj_id in unique_ids:
        colors[int(obj_id)] = color_for_id(obj_id)
    
    # 收集所有需要绘制的bbox信息
    bbox_list = []
    img_height, img_width = img.shape[:2]
    
    # 将obj_list转换为set以便快速查找
    obj_list_set = set(obj_list)
    
    apply_type_filter = allowed_ids_set is None and not skip_type_filter
    apply_distance_area = allowed_ids_set is None and not skip_distance_area
    
    for obj_id in unique_ids:
        # 如果obj_id在obj_list中，跳过类型检查；否则检查物品类型是否需要绘制
        in_obj_list = obj_id in obj_list_set
        if apply_type_filter and not in_obj_list and not should_draw_bbox_by_type(obj_id):
            continue
        
        # 找到该物品id对应的所有像素位置
        mask = (id_map == obj_id)
        ys, xs = np.where(mask)
        
        if len(xs) == 0 or len(ys) == 0:
            continue
        
        # 计算bbox (x_min, y_min, x_max, y_max)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        
        # 计算面积用于排序
        area = (x_max - x_min) * (y_max - y_min)
        
        # 如果obj_id在obj_list中，跳过距离/面积检查；否则检查是否通过距离/面积检查
        if apply_distance_area and not in_obj_list and not passes_distance_area_check(obj_id, area):
            continue
        
        bbox_list.append({
            'obj_id': obj_id,
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'area': area,
            'color': colors[int(obj_id)]
        })
    
    # 按面积从大到小排序，这样小物体的标签会绘制在最上面
    bbox_list.sort(key=lambda x: x['area'], reverse=True)
    
    # 第一遍：绘制所有bbox矩形
    for bbox_info in bbox_list:
        x_min, y_min = bbox_info['x_min'], bbox_info['y_min']
        x_max, y_max = bbox_info['x_max'], bbox_info['y_max']
        color = bbox_info['color']
        # 使用线宽1绘制两次，稍微错开位置，模拟1.5的粗细效果
        cv2.rectangle(labeled_img, (x_min, y_min), (x_max, y_max), color, 1)
        # 第二次绘制稍微向内偏移，让线条看起来更粗但不扩大矩形
        if x_max > x_min + 1 and y_max > y_min + 1:
            cv2.rectangle(labeled_img, (x_min + 1, y_min + 1), (x_max - 1, y_max - 1), color, 1)
    
    # 辅助函数：检测两个矩形是否重叠
    def rects_overlap(r1, r2):
        """检查两个矩形是否重叠，r1和r2格式为(x1, y1, x2, y2)"""
        return not (r1[2] <= r2[0] or r2[2] <= r1[0] or r1[3] <= r2[1] or r2[3] <= r1[1])
    
    # 辅助函数：计算重叠面积
    def overlap_area(r1, r2):
        """计算两个矩形的重叠面积"""
        x_overlap = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
        y_overlap = max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
        return x_overlap * y_overlap
    
    # 辅助函数：计算候选位置与已有标签的总重叠面积
    def calc_total_overlap(candidate_rect, placed_labels):
        total = 0
        for placed in placed_labels:
            total += overlap_area(candidate_rect, placed)
        return total
    
    # 记录已放置的标签位置
    placed_labels = []
    
    # 第二遍：绘制所有标签（按面积从大到小，小物体标签在上面）
    for bbox_info in bbox_list:
        obj_id = bbox_info['obj_id']
        x_min, y_min = bbox_info['x_min'], bbox_info['y_min']
        x_max, y_max = bbox_info['x_max'], bbox_info['y_max']
        color = bbox_info['color']
        
        label = str(int(obj_id))
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.3, 1)
        
        label_height = text_height + 2
        label_width = text_width + 2
        
        # 生成多个候选位置
        candidates = []
        
        # 位置1：bbox上方左侧
        if y_min - label_height >= 0:
            candidates.append({
                'x1': x_min, 'y1': y_min - label_height,
                'x2': x_min + label_width, 'y2': y_min,
                'text_y': y_min - 2
            })
        
        # 位置2：bbox上方右侧
        if y_min - label_height >= 0:
            candidates.append({
                'x1': max(0, x_max - label_width), 'y1': y_min - label_height,
                'x2': x_max, 'y2': y_min,
                'text_y': y_min - 2
            })
        
        # 位置3：bbox下方左侧
        if y_max + label_height <= img_height:
            candidates.append({
                'x1': x_min, 'y1': y_max,
                'x2': x_min + label_width, 'y2': y_max + label_height,
                'text_y': y_max + text_height + 2
            })
        
        # 位置4：bbox下方右侧
        if y_max + label_height <= img_height:
            candidates.append({
                'x1': max(0, x_max - label_width), 'y1': y_max,
                'x2': x_max, 'y2': y_max + label_height,
                'text_y': y_max + text_height + 2
            })
        
        # 位置5：bbox左侧
        if x_min - label_width >= 0:
            candidates.append({
                'x1': x_min - label_width, 'y1': y_min,
                'x2': x_min, 'y2': y_min + label_height,
                'text_y': y_min + text_height + 2
            })
        
        # 位置6：bbox右侧
        if x_max + label_width <= img_width:
            candidates.append({
                'x1': x_max, 'y1': y_min,
                'x2': x_max + label_width, 'y2': y_min + label_height,
                'text_y': y_min + text_height + 2
            })
        
        # 位置7-10：带偏移的位置（当标签密集时提供更多选择）
        offset = label_height + 3  # 额外偏移距离
        
        # 上方偏移
        if y_min - label_height - offset >= 0:
            candidates.append({
                'x1': x_min, 'y1': y_min - label_height - offset,
                'x2': x_min + label_width, 'y2': y_min - offset,
                'text_y': y_min - offset - 2
            })
        
        # 下方偏移
        if y_max + label_height + offset <= img_height:
            candidates.append({
                'x1': x_min, 'y1': y_max + offset,
                'x2': x_min + label_width, 'y2': y_max + offset + label_height,
                'text_y': y_max + offset + text_height + 2
            })
        
        # 左侧偏移
        if x_min - label_width - offset >= 0:
            candidates.append({
                'x1': x_min - label_width - offset, 'y1': y_min,
                'x2': x_min - offset, 'y2': y_min + label_height,
                'text_y': y_min + text_height + 2
            })
        
        # 右侧偏移
        if x_max + label_width + offset <= img_width:
            candidates.append({
                'x1': x_max + offset, 'y1': y_min,
                'x2': x_max + offset + label_width, 'y2': y_min + label_height,
                'text_y': y_min + text_height + 2
            })
        
        # 位置fallback：bbox内部左上角
        candidates.append({
            'x1': x_min, 'y1': y_min,
            'x2': min(x_min + label_width, img_width), 'y2': min(y_min + label_height, img_height),
            'text_y': y_min + text_height + 2
        })
        
        # 选择重叠最少的候选位置
        best_candidate = None
        min_overlap = float('inf')
        
        for cand in candidates:
            cand_rect = (cand['x1'], cand['y1'], cand['x2'], cand['y2'])
            total_overlap = calc_total_overlap(cand_rect, placed_labels)
            if total_overlap < min_overlap:
                min_overlap = total_overlap
                best_candidate = cand
            if total_overlap == 0:
                break  # 找到无重叠位置，立即使用
        
        # 计算标签面积
        label_area = label_width * label_height
        
        # 如果最佳位置的重叠面积超过标签面积的50%，跳过这个标签
        if min_overlap > label_area * 0.5:
            continue
        
        # 使用最佳位置
        label_x1, label_y1 = best_candidate['x1'], best_candidate['y1']
        label_x2, label_y2 = best_candidate['x2'], best_candidate['y2']
        text_y = best_candidate['text_y']
        
        # 记录该标签位置
        placed_labels.append((label_x1, label_y1, label_x2, label_y2))
        
        # 绘制标签背景（带黑色边框使标签更清晰）
        cv2.rectangle(labeled_img, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), 2)
        cv2.rectangle(labeled_img, (label_x1, label_y1), (label_x2, label_y2), color, -1)
        
        # 绘制id文字（白色文字带黑色描边，更清晰）
        text_x = label_x1 + 1
        # 黑色描边（使用LINE_AA抗锯齿）
        cv2.putText(labeled_img, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
        # 白色文字
        cv2.putText(labeled_img, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    
    return labeled_img


def get_img_bbox(id_map, visible_objs:list):
    """
    根据ID map返回每个物品的bbox
    
    Args:
        id_map: 逐像素的ID map (numpy array, shape: H x W)，每个像素值表示对应物品的id
        visible_objs: 视野内能看到的物品的信息，格式为[{'id':...,'property':[...], 'class_name':'...'}]
    
    Returns:
        字典，key为物品id，value为bbox (x_min, y_min, x_max, y_max)
        只返回满足条件的物品：GRABBABLE或SURFACES属性，或者class_name在指定列表中
    """
    import numpy as np
    
    # 创建visible_objs的id到对象信息的映射
    visible_objs_dict = {obj['id']: obj for obj in visible_objs}
    
    # 定义需要返回的class_name列表
    target_class_names = {'bathroomcabinet', 'kitchencabinet', 'cabinet', 'fridge', 'stove', 'microwave'}
    
    # 判断物品是否需要返回bbox
    def should_return_bbox(obj_id):
        if obj_id not in visible_objs_dict:
            return False
        obj = visible_objs_dict[obj_id]
        properties = obj.get('properties', [])
        class_name = obj.get('class_name', '')
        # 检查property中是否包含GRABBABLE或SURFACES
        if 'GRABBABLE' in properties or 'SURFACES' in properties or 'DRINKABLE' in properties:
            return True
        
        # 检查class_name是否在目标列表中
        if class_name in target_class_names:
            return True
        
        return False
    
    # 获取所有唯一的物品id（排除背景id，通常为0或-1）
    unique_ids = np.unique(id_map)
    unique_ids = unique_ids[unique_ids > 0]  # 假设0或负数为背景
    
    # 存储每个id对应的bbox
    bbox_dict = {}
    
    # 遍历每个物品id，找到bbox
    for obj_id in unique_ids:
        # 检查是否需要返回该物品的bbox
        if not should_return_bbox(obj_id):
            continue
        
        # 找到该物品id对应的所有像素位置
        mask = (id_map == obj_id)
        ys, xs = np.where(mask)
        
        if len(xs) == 0 or len(ys) == 0:
            continue
        
        # 计算bbox (x_min, y_min, x_max, y_max)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        
        bbox_dict[int(obj_id)] = {'bbox': (x_min, y_min, x_max, y_max), 'class_name': visible_objs_dict[obj_id]['class_name']}
    
    return bbox_dict

def get_obj_list(task):
    task_goal = task['task_goal']['0']
    name_list = []
    target_ids = []
    for item, num in task_goal.items():
        if num > 0:
            elements = item.split('_')
            name_list.append(elements[1])
            target_ids.extend(get_allowed_target_ids(task, item))
    target_ids = list(set(target_ids))
    obj_list = [item['id'] for item in task['init_graph']['nodes'] if item['class_name'] in name_list]
    obj_list += target_ids
    return obj_list
