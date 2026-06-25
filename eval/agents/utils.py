import requests
import cv2
import base64
import numpy as np
from collections import Counter
from io import BytesIO
import re
import ast
import json


_BOX_CHUNK_RE = re.compile(r"<\|begin_of_box\|>|<\|end_of_box\|>", re.I)


# ============ String/Text Utilities ============

def strip_code_fence(text):
    """Remove markdown code fence and GLM box markers from text."""
    cleaned = _BOX_CHUNK_RE.sub("", text).strip() if text else text
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def strip_id_tokens(text):
    """Remove or mask ID tokens from text for display purposes."""
    if not text:
        return text
    cleaned = re.sub(r"\b[idID]\s*=\s*\d+\b", "id=?", text)
    cleaned = re.sub(r"\([^\)]*\d+[^\)]*\)", "", cleaned)
    cleaned = re.sub(r"\[[^\]]*\d+[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_embed_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def build_query_text(param):
    if not isinstance(param, dict):
        return ""
    return (param.get("name") or "").strip()


def candidate_similarity_text(candidate):
    parts = []
    class_name = str(candidate.get("class_name", "")).strip()
    if class_name:
        parts.append(class_name)
    description = str(candidate.get("description", "")).strip()
    if description:
        parts.append(description)
    other_room = str(candidate.get("other_room", "")).strip()
    if other_room:
        parts.append(other_room)
    connected_rooms = [
        str(room).strip()
        for room in (candidate.get("connected_rooms") or [])
        if str(room).strip()
    ]
    if connected_rooms:
        parts.append(" ".join(connected_rooms))
    return " ".join(dict.fromkeys(parts))


def subgoals_to_string(subgoals: dict):
    if len(subgoals) == 0:
        return "No subtask completed."
    subgoals_str = 'There are already'
    for subgoal, num in subgoals.items():
        elements = subgoal.split('_')
        subgoals_str += f"{num} {elements[1]} {elements[0]} {elements[2]};"
    return subgoals_str[:-1] + '.'


# ============ Output Parsers ============

_BOX_ACTION_RE = re.compile(r'^\s*\{.*"action"\s*:', re.S)
_BOX_MEMORY_RE = re.compile(
    r'^\s*\{.*"livingroom"\s*:\s*\{\s*"exploration"\s*:',
    re.S,
)


def _split_box_chunks(answer):
    parts = _BOX_CHUNK_RE.split(answer)
    return [part.strip() for part in parts if part and part.strip()]


def _is_box_action_chunk(text):
    return bool(_BOX_ACTION_RE.search(text))


def _is_box_memory_chunk(text):
    return bool(_BOX_MEMORY_RE.search(text))


def _strip_box_delimiters(text):
    if not text:
        return text
    return _BOX_CHUNK_RE.sub("", text).strip()


def _extract_tagged_json_block(text, tag_name):
    """Pull JSON payload from [TAG] sections embedded inside a box chunk."""
    pattern = rf"\[\s*{tag_name}\s*\]\s*(\{{.*)", re.I
    match = re.search(pattern, text, re.S | re.I)
    if not match:
        return _strip_box_delimiters(text)
    return _strip_box_delimiters(match.group(1))


def _clean_json_block(text, *, tag_name=None):
    cleaned = _strip_box_delimiters(text)
    if tag_name:
        cleaned = _extract_tagged_json_block(cleaned, tag_name)
    return cleaned.strip()


def _parse_output_blocks_from_boxes(answer, parsed):
    """Fallback when GLM emits <|begin_of_box|> instead of [ACTION]/[MEMORY] tags."""
    if "<|begin_of_box|>" not in answer.lower():
        return parsed

    for chunk in _split_box_chunks(answer):
        if _is_box_action_chunk(chunk):
            if parsed["action"] is None:
                parsed["action"] = _clean_json_block(chunk)
            continue
        if _is_box_memory_chunk(chunk):
            if parsed["memory"] is None:
                parsed["memory"] = _clean_json_block(chunk)
            continue
        if "[ACTION]" in chunk.upper():
            if parsed["action"] is None:
                parsed["action"] = _clean_json_block(chunk, tag_name="ACTION")
            continue
        if "[MEMORY]" in chunk.upper():
            if parsed["memory"] is None:
                parsed["memory"] = _clean_json_block(chunk, tag_name="MEMORY")
            continue
        if parsed["action"] is None:
            if parsed["action_thinking"] is None:
                parsed["action_thinking"] = chunk
            else:
                parsed["action_thinking"] = f"{parsed['action_thinking']}\n{chunk}"
        elif parsed["memory"] is None:
            if parsed["action_thinking"] is None:
                parsed["action_thinking"] = chunk
            elif parsed["memory_thinking"] is None:
                parsed["memory_thinking"] = chunk
            else:
                parsed["memory_thinking"] = f"{parsed['memory_thinking']}\n{chunk}"
        elif parsed["memory_thinking"] is None:
            parsed["memory_thinking"] = chunk
        else:
            parsed["memory_thinking"] = f"{parsed['memory_thinking']}\n{chunk}"
    return parsed


def parse_output_blocks(answer):
    """Parse model output into structured blocks (THINKING, MESSAGE, ACTION, MEMORY).

    Block headers may be bracketed ([ACTION]) or markdown-bold (**ACTION**).
    Tag matching is case-insensitive.

    If the model emits multiple ACTION blocks, only the first is kept for downstream
    decoding; later ACTION blocks are ignored.
    """
    parsed = {
        "raw_output": answer,
        "action_thinking": None,
        "action": None,
        "message_thinking": None,
        "message": None,
        "memory_thinking": None,
        "memory": None,
    }
    tags = "THINKING|MESSAGE|ACTION|MEMORY"
    # Strict headers:
    # - [TAG]
    # - **TAG**  (but not /**TAG**/ style wrappers)
    tag_header = rf"(?:\[\s*({tags})\s*\]|(?<!/)\*\*\s*({tags})\s*\*\*(?!/))"
    # Lookahead must not use capturing groups (they would change findall() shape).
    # Also stop at GLM box end markers so [ACTION] JSON is not polluted.
    tag_header_nc = (
        rf"(?:\[\s*(?:{tags})\s*\]|(?<!/)\*\*\s*(?:{tags})\s*\*\*(?!/)|<\|end_of_box\|>)"
    )
    pattern = rf"{tag_header}(.*?)(?=\n?{tag_header_nc}|\Z)"
    matches = re.findall(pattern, answer, re.S | re.I)
    blocks = [(t1 or t2, content) for t1, t2, content in matches]
    pending_thinking = None
    for tag, content in blocks:
        tag = tag.upper()
        content = content.strip()
        if tag == "THINKING":
            pending_thinking = content
        elif tag == "MESSAGE":
            parsed["message_thinking"] = pending_thinking
            parsed["message"] = content
            pending_thinking = None
        elif tag == "ACTION":
            if parsed["action"] is None:
                parsed["action_thinking"] = pending_thinking
                parsed["action"] = content
            pending_thinking = None
        elif tag == "MEMORY":
            parsed["memory_thinking"] = pending_thinking
            parsed["memory"] = content
            pending_thinking = None
    if parsed["action"] is None or parsed["memory"] is None:
        _parse_output_blocks_from_boxes(answer, parsed)
    if parsed["action"] is not None:
        parsed["action"] = _clean_json_block(parsed["action"])
    if parsed["memory"] is not None:
        parsed["memory"] = _clean_json_block(parsed["memory"])
    return parsed


def _extract_block(answer, tag, next_tags):
    escaped_tag = re.escape(tag)
    if next_tags:
        lookahead = "|".join(re.escape(item) for item in next_tags)
        pattern = rf"\[{escaped_tag}\](.*?)(?=\n?\[(?:{lookahead})\]|\Z)"
    else:
        pattern = rf"\[{escaped_tag}\](.*)$"
    match = re.search(pattern, answer, re.S)
    if not match:
        return None
    return match.group(1).strip()


def _parse_ready_token(text):
    if text is None:
        raise ValueError("Missing READY block")
    normalized = text.strip().upper()
    if normalized == "<READY>":
        return True
    if normalized == "<CONTINUE>":
        return False
    raise ValueError(f"Invalid READY token: {text}")


def parse_discussion_first_round_output(answer):
    return {
        "thinking": _extract_block(answer, "THINKING", ["READY", "MESSAGE", "LOCAL_CONTEXT"]) or "",
        "ready": _parse_ready_token(_extract_block(answer, "READY", ["MESSAGE", "LOCAL_CONTEXT"])),
        "message": _extract_block(answer, "MESSAGE", ["LOCAL_CONTEXT"]),
        "local_context": _extract_block(answer, "LOCAL_CONTEXT", []),
    }


def parse_discussion_text_round_output(answer):
    return {
        "thinking": _extract_block(answer, "THINKING", ["READY", "MESSAGE"]) or "",
        "ready": _parse_ready_token(_extract_block(answer, "READY", ["MESSAGE"])),
        "message": _extract_block(answer, "MESSAGE", []),
    }


def parse_worker_report_output(answer):
    return {
        "thinking": _extract_block(answer, "THINKING", ["MESSAGE"]) or "",
        "message": _extract_block(answer, "MESSAGE", []),
    }


def parse_leader_assignments_output(answer):
    messages_block = _extract_block(answer, "MESSAGES", [])
    if messages_block is None:
        raise ValueError("Missing MESSAGES block")
    cleaned = strip_code_fence(messages_block)
    try:
        parsed = json.loads(cleaned)
    except Exception as exc:
        raise ValueError(f"Failed to parse MESSAGES JSON: {exc}")
    if not isinstance(parsed, list):
        raise ValueError(f"MESSAGES must be a JSON array, got: {type(parsed)}")
    return {
        "thinking": _extract_block(answer, "THINKING", ["MESSAGES"]) or "",
        "messages": parsed,
    }


# ============ Image Utilities ============

def encode_image_to_base64(img):
    """Encode a cv2 image (numpy array) to base64 string."""
    success, encoded = cv2.imencode('.png', img)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(encoded).decode('utf-8')


def get_bbox(image, objs, score_thr=0.05):
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    response = requests.post('http://localhost:8080/detect', json={
        'image_base64': image_base64,
        'texts': objs,
        'max_dets': 100,
        'score_thr': score_thr,
    })
    response = response.json()
    return response['bboxes']


def get_mask(image, objs, score_thr=0.1, save_path=None):
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode image")

    image_bytes = BytesIO(buffer.tobytes())
    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
    data = {
        "prompt": objs,
        "confidence_threshold": score_thr,
    }

    response = requests.post('http://localhost:8082/detect', files=files, data=data)

    if response.status_code != 200:
        raise RuntimeError(f"SAM3 API request failed: {response.status_code}, {response.text}")

    result = response.json()

    if save_path is not None and result.get('output_image_base64'):
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img_data = base64.b64decode(result['output_image_base64'])
        with open(save_path, 'wb') as f:
            f.write(img_data)

    masks = []
    if result.get('count', 0) > 0:
        for det in result['detections']:
            if det.get('mask'):
                mask_array = np.array(det['mask'], dtype=np.uint8)
                masks.append(mask_array)

    return masks


def mask_to_id(masks, id_map, ignore_id=-1):
    ids = []
    h, w = id_map.shape[:2]

    for mask in masks:
        if mask.shape[:2] != (h, w):
            mask_h, mask_w = mask.shape[:2]
            if mask_h != h or mask_w != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_bool = (mask > 0.5).astype(bool)

        if not np.any(mask_bool):
            ids.append(None)
            continue

        mask_region = id_map[mask_bool]

        if mask_region.size == 0:
            ids.append(None)
            continue

        flat = mask_region.flatten()
        counter = Counter(flat)

        if ignore_id in counter:
            del counter[ignore_id]

        if counter:
            most_common_id, _ = counter.most_common(1)[0]
            ids.append(int(most_common_id))
        else:
            ids.append(None)

    return ids


def bbox_to_id(bboxes, id_map, ignore_id=-1):
    ids = []
    h, w = id_map.shape[:2]

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        region = id_map[int(y1):int(y2), int(x1):int(x2)]

        if region.size == 0:
            ids.extend([])
            continue

        flat = region.flatten()
        counter = Counter(flat)

        if ignore_id in counter:
            del counter[ignore_id]

        if counter:
            top_ids = [int(id_val) for id_val, _ in counter.most_common(3)]
            ids.extend(top_ids)
        else:
            ids.extend([])

    return ids


def parse_bbox(text):
    match = re.search(r"\[([^\]]+)\]", text)
    if not match:
        raise ValueError(f"No bounding box found in output: {text}")

    bbox_str = "[" + match.group(1) + "]"

    try:
        bbox = ast.literal_eval(bbox_str)
    except:
        raise ValueError(f"Failed to parse bbox: {bbox_str}")

    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError(f"Invalid bbox format: {bbox}")

    bbox = [float(x) for x in bbox]

    return bbox


# ============ View Content Helpers ============

def get_label_views(obs, view_mode):
    """Extract view_order, label_views, raw_views from observation, respecting view_mode."""
    view_order = obs.get('view_order')
    label_views = obs.get('label_views')
    raw_views = obs.get('raw_views')

    if view_order is None:
        view_order = ['front']

    if view_mode == 'first_person':
        if 'front' in view_order:
            front_idx = view_order.index('front')
            view_order = ['front']
            label_views = [label_views[front_idx]] if label_views else None
            raw_views = [raw_views[front_idx]] if raw_views else None

    return view_order, label_views, raw_views


def build_image_content_item(img):
    """Build a single image content item for OpenAI-compatible API request."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(img)}"},
    }


def build_bbox_view_content(obs, view_mode, label_views_override=None, raw_views_override=None, view_order_override=None):
    """Build API content with bbox-annotated views (and optionally raw views)."""
    view_order, label_views, raw_views = get_label_views(obs, view_mode)
    if label_views_override is not None:
        label_views = label_views_override
    if raw_views_override is not None:
        raw_views = raw_views_override
    if view_order_override is not None:
        view_order = view_order_override
    if not label_views:
        raise ValueError("No bbox views found in observation")
    content = []
    is_multi_view = view_order and len(view_order) > 1

    if is_multi_view:
        content.append({
            "type": "text",
            "text": f"Image order (left to right): {', '.join(view_order)}. First image is raw view, second image includes bbox labels.",
        })
        if raw_views:
            content.append(build_image_content_item(np.hstack(raw_views)))
        content.append(build_image_content_item(np.hstack(label_views)))
    else:
        if raw_views:
            content.append(build_image_content_item(raw_views[0]))
        content.append(build_image_content_item(label_views[0]))
    return content


def build_raw_view_content(obs, view_mode):
    """Build API content with raw (un-annotated) views only."""
    view_order, _, raw_views = get_label_views(obs, view_mode)
    if not raw_views:
        raise ValueError("No raw views found in observation")
    content = []
    is_multi_view = view_order and len(view_order) > 1

    if is_multi_view:
        content.append({
            "type": "text",
            "text": f"Image order (left to right): {', '.join(view_order)}. Raw image only (no bbox, no IDs).",
        })
        content.append(build_image_content_item(np.hstack(raw_views)))
    else:
        content.append(build_image_content_item(raw_views[0]))
    return content


def build_labeled_view_composite(groups, view_mode):
    """Build a single vertically-stacked composite image with a text label bar
    above each group's raw multi-view row.

    Args:
        groups: list of {"label": str, "obs": dict}. Each obs must carry
            `raw_views` and `view_order` (as produced by Scene.get_observation).
        view_mode: 'multi_view' or 'first_person' — matches VlmAgent.view_mode.

    Returns:
        A BGR np.ndarray suitable for cv2.imwrite, or None if no valid groups.
    """
    strips = []
    for group in groups or []:
        obs = group.get("obs") or {}
        label = group.get("label") or ""
        view_order, _, raw_views = get_label_views(obs, view_mode)
        if not raw_views:
            continue
        if view_order and len(view_order) > 1:
            row = np.hstack(raw_views)
        else:
            row = raw_views[0]
        bar_h = 18
        bar = np.full((bar_h, row.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(
            bar,
            str(label),
            (5, 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        strips.append(np.vstack([bar, row]))

    if not strips:
        return None

    max_w = max(strip.shape[1] for strip in strips)
    padded = []
    for strip in strips:
        if strip.shape[1] < max_w:
            pad = np.full(
                (strip.shape[0], max_w - strip.shape[1], 3),
                255,
                dtype=np.uint8,
            )
            strip = np.hstack([strip, pad])
        padded.append(strip)
    return np.vstack(padded)
