import numpy as np
import re
import os
import json
import cv2
import sys
import time

from .action import check_legality, normalize_action_name
from .utils import (
    strip_code_fence, normalize_embed_text, cosine_similarity,
    build_query_text, candidate_similarity_text,
    get_label_views, build_bbox_view_content,
)

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{curr_dir}/../../')
from eval.scene.utils import label_img_with_bbox, get_obj_list
from eval.utils.logger import log_print
from eval.utils.token_usage import (
    record_chat_completion,
    record_embedding_usage,
    SCOPE_RESOLVE,
    SCOPE_EMBEDDING,
)


class ActionResolver:
    """Resolves action descriptions to concrete object IDs using visual candidates and VLM."""

    def __init__(self, *, resolve_client, resolve_model_name,
                 resolve_model_key=None,
                 embedding_client, embedding_model,
                 embedding_model_key=None,
                 sim_threshold, min_keep, embedding_enabled,
                 vis_max_distance, vis_min_area, vis_min_area_for_far,
                 view_mode, action_decoder, resolve_prompt_template):
        self.resolve_client = resolve_client
        self.resolve_model_name = resolve_model_name
        self.resolve_model_key = (resolve_model_key or "").strip() or None
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.embedding_model_key = (embedding_model_key or "").strip() or None
        self.sim_threshold = sim_threshold
        self.min_keep = min_keep
        self.embedding_enabled = embedding_enabled
        self.vis_max_distance = vis_max_distance
        self.vis_min_area = vis_min_area
        self.vis_min_area_for_far = vis_min_area_for_far
        self.view_mode = view_mode
        self.action_decoder = action_decoder
        self.resolve_prompt_template = resolve_prompt_template
        self._embedding_cache = {}
        self._task_obj_ids = None
        self._task = None
        self._restriction_context = None

    def set_task(self, task, restriction_context):
        self._task = task
        self._restriction_context = restriction_context
        self._task_obj_ids = None

    def decode_action(self, action_text, obs, step=0, agent_id=0):
        """Full pipeline: parse -> resolve special or VLM resolve -> check legality."""
        parsed = self.action_decoder.parse_action(action_text)
        resolved = self.action_decoder.resolve_special_action(parsed, obs)
        if resolved is None:
            action_type, objs = self._resolve_candidate_action_ids(
                parsed, obs, step=step, agent_id=agent_id,
            )
        else:
            action_type, objs = resolved
        s, m = check_legality(action_type, objs, obs)
        if not s:
            raise ValueError(f"Invalid action: {action_type} {m}")
        objs = [[item] for item in objs]
        return action_type, objs, None

    # ---- Topology / Door Candidates ----

    def _get_department_topology(self):
        task = self._task
        if isinstance(task, dict):
            topology = task.get("department_topology")
            if topology:
                return topology
        ctx = self._restriction_context
        if isinstance(ctx, dict):
            topology = ctx.get("department_topology")
            if topology:
                return topology
        return None

    def _parse_topology_connected_doors(self, current_room, topology):
        if not current_room or not topology:
            return []
        current_room = str(current_room).strip().lower()
        door_specs = []
        seen_ids = set()
        for raw_item in str(topology).split(","):
            item = raw_item.strip()
            if not item:
                continue
            match = re.match(r"^([A-Za-z0-9_]+)--door\((\d+)\)--([A-Za-z0-9_]+)$", item)
            if not match:
                continue
            room_a, door_id_text, room_b = match.groups()
            room_a = room_a.strip().lower()
            room_b = room_b.strip().lower()
            if current_room not in {room_a, room_b}:
                continue
            door_id = int(door_id_text)
            if door_id in seen_ids:
                continue
            seen_ids.add(door_id)
            other_room = room_b if room_a == current_room else room_a
            door_specs.append({
                "door_id": door_id,
                "connected_rooms": [room_a, room_b],
                "other_room": other_room,
                "candidate_source": "department_topology",
            })
        return door_specs

    def _build_topology_door_candidate(self, door_node, spec, location):
        distance = None
        bbox = door_node.get("bounding_box") or {}
        center = bbox.get("center")
        if location is not None and isinstance(center, (list, tuple)) and len(center) >= 3:
            dx = float(center[0]) - float(location[0])
            dz = float(center[2]) - float(location[2])
            distance = float(np.sqrt(dx * dx + dz * dz))

        connected_rooms = [
            str(room).strip().lower()
            for room in (spec.get("connected_rooms") or [])
            if str(room).strip()
        ]
        other_room = str(spec.get("other_room") or "").strip().lower() or None
        current_room = str(spec.get("current_room") or "").strip().lower() or None
        if current_room and other_room:
            description = f"door connecting {current_room} and {other_room}"
        elif connected_rooms:
            description = f"door connected to {' and '.join(connected_rooms)}"
        else:
            description = "door"

        return {
            "id": int(door_node["id"]),
            "class_name": door_node.get("class_name", "door"),
            "view": None,
            "bbox": None,
            "center": None,
            "area": 0.0,
            "area_px": 0,
            "distance": distance,
            "description": description,
            "connected_rooms": connected_rooms,
            "other_room": other_room,
            "candidate_source": spec.get("candidate_source", "department_topology"),
        }

    def _collect_topology_door_candidates(self, obs):
        current_room = str(obs.get("current_room") or "").strip().lower()
        curr_graph = obs.get("curr_graph") or {}
        nodes = {
            int(node["id"]): node
            for node in (curr_graph.get("nodes") or [])
            if node.get("id") is not None
        }
        if not current_room or not nodes:
            return []

        door_specs = self._parse_topology_connected_doors(
            current_room,
            self._get_department_topology(),
        )

        candidates = []
        for spec in door_specs:
            door_node = nodes.get(int(spec["door_id"]))
            if door_node is None or door_node.get("class_name") not in {"door", "doorjamb"}:
                continue
            spec = dict(spec)
            spec["current_room"] = current_room
            candidates.append(
                self._build_topology_door_candidate(
                    door_node,
                    spec,
                    obs.get("location"),
                )
            )
        return candidates

    def _is_topology_candidate(self, candidate):
        return "topology" in str(candidate.get("candidate_source", ""))

    # ---- Task Object IDs ----

    def _get_task_obj_ids(self):
        if self._task_obj_ids is None:
            try:
                if self._task:
                    self._task_obj_ids = set(get_obj_list(self._task))
                else:
                    self._task_obj_ids = set()
            except Exception:
                self._task_obj_ids = set()
        return self._task_obj_ids

    # ---- Embedding / Similarity Filtering ----

    def _get_text_embeddings(self, texts):
        if not texts or not self.embedding_enabled:
            return [None] * len(texts)
        normalized = [normalize_embed_text(text) for text in texts]
        results = [None] * len(texts)
        to_fetch = []
        for idx, norm in enumerate(normalized):
            if not norm:
                continue
            cached = self._embedding_cache.get(norm)
            if cached is not None:
                results[idx] = cached
            else:
                to_fetch.append(norm)
        if to_fetch:
            try:
                max_retries = 3
                for attempt in range(max_retries + 1):
                    try:
                        response = self.embedding_client.embeddings.create(
                            model=self.embedding_model,
                            input=to_fetch,
                        )
                        record_embedding_usage(SCOPE_EMBEDDING, self.embedding_model_key, response)
                        break
                    except Exception as e:
                        if attempt == max_retries:
                            raise
                        time.sleep(2 ** attempt)
                for norm, item in zip(to_fetch, response.data):
                    self._embedding_cache[norm] = np.array(item.embedding, dtype=np.float32)
            except Exception as exc:
                log_print(f"embedding request failed, fallback to no filtering: {exc}")
                return [None] * len(texts)
        for idx, norm in enumerate(normalized):
            if not norm:
                continue
            results[idx] = self._embedding_cache.get(norm)
        return results

    def _apply_similarity_filter(self, candidates, similarities, keep_ids=None):
        """Filter candidates by precomputed similarity scores, ensuring minimum kept count."""
        keep_ids_set = set(keep_ids or [])
        filtered = []
        for cand, sim in zip(candidates, similarities):
            cand["similarity"] = sim
            if cand["id"] in keep_ids_set or sim >= self.sim_threshold:
                filtered.append(cand)
        if len(filtered) < self.min_keep and candidates:
            ranked = sorted(
                zip(candidates, similarities),
                key=lambda item: item[1],
                reverse=True,
            )
            kept_ids = {item["id"] for item in filtered}
            for cand, _sim in ranked:
                if cand["id"] in kept_ids:
                    continue
                filtered.append(cand)
                kept_ids.add(cand["id"])
                if len(filtered) >= self.min_keep:
                    break
        return filtered

    def _filter_candidates_by_similarity(self, candidates, query_text, keep_ids=None):
        if not candidates or not query_text or not self.embedding_enabled:
            return candidates
        candidate_texts = [candidate_similarity_text(item) for item in candidates]
        embeddings = self._get_text_embeddings([query_text] + candidate_texts)
        if not embeddings or embeddings[0] is None:
            return candidates
        query_emb = embeddings[0]
        similarities = [cosine_similarity(query_emb, emb) for emb in embeddings[1:]]
        return self._apply_similarity_filter(candidates, similarities, keep_ids)

    def _filter_candidates_by_similarity_multi(self, candidates, query_texts, keep_ids=None):
        if not candidates or not query_texts or not self.embedding_enabled:
            return candidates
        query_texts = [normalize_embed_text(text) for text in query_texts if text]
        if not query_texts:
            return candidates
        candidate_texts = [candidate_similarity_text(item) for item in candidates]
        embeddings = self._get_text_embeddings(query_texts + candidate_texts)
        if not embeddings or any(emb is None for emb in embeddings[:len(query_texts)]):
            return candidates
        query_embs = embeddings[:len(query_texts)]
        cand_embs = embeddings[len(query_texts):]
        similarities = [
            max(cosine_similarity(qe, emb) for qe in query_embs) if emb is not None else 0.0
            for emb in cand_embs
        ]
        return self._apply_similarity_filter(candidates, similarities, keep_ids)

    # ---- Candidate Collection ----

    def _collect_action_candidates(self, obs):
        candidates_by_id = {}
        exclude_ids = set(obs.get('satisfied_obj_ids') or [])
        views = obs.get('views') or {}
        location = obs.get("location")
        for view_name, view in views.items():
            id_img = view.get('id_img')
            visible_objs = view.get('visible_objects') or []
            if id_img is None or not visible_objs:
                continue
            id_img = np.asarray(id_img)
            h, w = id_img.shape[:2]
            for obj in visible_objs:
                obj_id = int(obj['id'])
                if obj_id in exclude_ids:
                    continue
                mask = id_img == obj_id
                if not np.any(mask):
                    continue
                ys, xs = np.where(mask)
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                if x2 <= x1 or y2 <= y1:
                    continue
                area_px = int((x2 - x1) * (y2 - y1))
                area = float(area_px) / float(h * w)
                distance = None
                if location is not None:
                    bbox = obj.get("bounding_box") or {}
                    center = bbox.get("center")
                    if isinstance(center, (list, tuple)) and len(center) >= 3:
                        dx = float(center[0]) - float(location[0])
                        dz = float(center[2]) - float(location[2])
                        distance = float(np.sqrt(dx * dx + dz * dz))
                cand = {
                    "id": obj_id,
                    "class_name": obj.get('class_name', ''),
                    "view": view_name,
                    "bbox": [x1 / w, y1 / h, x2 / w, y2 / h],
                    "center": [((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h],
                    "area": area,
                    "area_px": area_px,
                    "distance": distance,
                    "candidate_source": "visible",
                }
                prev = candidates_by_id.get(obj_id)
                if prev is None or cand["area"] > prev["area"]:
                    candidates_by_id[obj_id] = cand
        for cand in self._collect_topology_door_candidates(obs):
            prev = candidates_by_id.get(cand["id"])
            if prev is None:
                candidates_by_id[cand["id"]] = cand
                continue
            if cand.get("description"):
                prev["description"] = cand["description"]
            if cand.get("connected_rooms"):
                prev["connected_rooms"] = cand["connected_rooms"]
            if cand.get("other_room"):
                prev["other_room"] = cand["other_room"]
            if self._is_topology_candidate(cand):
                prev["candidate_source"] = "visible+topology"
        return list(candidates_by_id.values())

    @staticmethod
    def _slim_candidates(candidates):
        return [
            {"id": int(item["id"]), "class_name": item.get("class_name", "")}
            for item in candidates
        ]

    # ---- View Building for Candidates ----

    def _build_label_views_for_candidates(self, obs, candidate_ids):
        view_order = obs.get('view_order') or ['front']
        views = obs.get('views') or {}
        if self.view_mode == 'first_person':
            if 'front' in view_order:
                view_order = ['front']
            else:
                view_order = view_order[:1]
        label_views = []
        raw_views = []
        for view_name in view_order:
            view = views.get(view_name)
            if not view:
                continue
            raw_view = view.get('rgb')
            raw_views.append(raw_view)
            label_img = label_img_with_bbox(
                raw_view,
                np.asarray(view.get('id_img')),
                view.get('visible_objects') or [],
                obs.get('location'),
                obj_list=[],
                allowed_ids=candidate_ids,
                skip_distance_area=True,
                skip_type_filter=True,
            )
            label_views.append(label_img)
        return view_order, label_views, raw_views

    @staticmethod
    def _save_filtered_label_views(label_views, view_order, step, agent_id, obs):
        record_dir = obs.get("record_dir")
        if not record_dir or not label_views:
            return
        try:
            os.makedirs(record_dir, exist_ok=True)
            is_multi_view = view_order and len(view_order) > 1
            if is_multi_view:
                combined_label_img = np.hstack(label_views)
                cv2.imwrite(
                    os.path.join(record_dir, f'label_img_combined_{agent_id}_{step}.png'),
                    combined_label_img,
                )
            else:
                view_name = view_order[0] if view_order else "front"
                cv2.imwrite(
                    os.path.join(record_dir, f'label_img_{view_name}_{agent_id}_{step}.png'),
                    label_views[0],
                )
        except Exception as exc:
            log_print(f"save filtered label views failed: {exc}")

    # ---- Main Resolution ----

    def _resolve_candidate_action_ids(self, parsed_action, obs, step=0, agent_id=0):
        action_type = parsed_action.action_type
        param_1 = parsed_action.param_1
        param_2 = parsed_action.param_2
        candidates = self._collect_action_candidates(obs)
        others_holding_ids = set(
            item['id'] for item in obs.get('others_holding_obj', [])
        )
        if others_holding_ids:
            candidates = [c for c in candidates if c['id'] not in others_holding_ids]

        if action_type in {"put_in", "put_on"}:
            query_texts = [
                build_query_text(param_1),
                build_query_text(param_2),
            ]
        else:
            query_texts = [build_query_text(param_1)]
        task_keep_ids = self._get_task_obj_ids()
        visibility_keep_ids = {
            item["id"]
            for item in candidates
            if (
                item.get("area_px", 0) >= self.vis_min_area
                and (
                    (item.get("distance") is not None and item["distance"] <= self.vis_max_distance)
                    or item.get("area_px", 0) >= self.vis_min_area_for_far
                )
            )
        }
        topology_keep_ids = {
            item["id"]
            for item in candidates
            if self._is_topology_candidate(item)
        }
        keep_ids = set(task_keep_ids).union(visibility_keep_ids).union(topology_keep_ids)
        filtered_candidates = self._filter_candidates_by_similarity_multi(
            candidates,
            query_texts,
            keep_ids=keep_ids,
        )
        if not filtered_candidates:
            filtered_candidates = candidates

        holding_objects = [
            {"id": int(item["id"]), "class_name": item.get("class_name", "")}
            for item in obs.get('hoding_obj', [])
        ]

        if action_type in {"walk", "open", "close", "switch_on", "switch_off", "sit", "grab", "put_in", "put_on"} and not candidates:
            raise ValueError("No visible candidates to resolve action")

        candidate_ids_list = [item["id"] for item in filtered_candidates]
        view_order, label_views, raw_views = self._build_label_views_for_candidates(obs, candidate_ids_list)
        if not label_views:
            view_order, label_views, raw_views = get_label_views(obs, self.view_mode)
        self._save_filtered_label_views(label_views, view_order, step, agent_id, obs)

        resolve_input = {
            "action": action_type,
            "parameter_1": param_1,
            "parameter_2": param_2,
            "holding_objects": holding_objects,
            "candidates": self._slim_candidates(filtered_candidates),
        }
        resolve_prompt = self.resolve_prompt_template.replace(
            '$RESOLVE_INPUT$',
            json.dumps(resolve_input, ensure_ascii=True),
        )
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.resolve_client.chat.completions.create(
                    model=self.resolve_model_name,
                    messages=[{
                        "role": "user",
                        "content": [{"type": "text", "text": resolve_prompt}] + build_bbox_view_content(
                            obs,
                            self.view_mode,
                            label_views_override=label_views,
                            raw_views_override=raw_views,
                            view_order_override=view_order,
                        ),
                    }],
                    temperature=0,
                )
                record_chat_completion(SCOPE_RESOLVE, self.resolve_model_key, response)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)
        try:
            resolved_text = strip_code_fence(response.choices[0].message.content)
            resolved = json.loads(resolved_text)
        except Exception as exc:
            raise ValueError(f"action id resolve parse error: {exc}")
        resolved_action = normalize_action_name(resolved.get('action', action_type))
        parameter_ids = resolved.get('parameter_ids', [])
        if not isinstance(parameter_ids, list):
            raise ValueError("Resolved parameter_ids must be a list")
        final_ids = []
        for item in parameter_ids:
            if isinstance(item, str) and item.isdigit():
                final_ids.append(int(item))
            else:
                final_ids.append(item)
        candidate_ids = {c["id"] for c in filtered_candidates}
        holding_ids = {item["id"] for item in obs.get('hoding_obj', [])}
        self.action_decoder.validate_resolved_ids(
            resolved_action,
            final_ids,
            candidate_ids=candidate_ids,
            holding_ids=holding_ids,
        )
        return resolved_action, final_ids
