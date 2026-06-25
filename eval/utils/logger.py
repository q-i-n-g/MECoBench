
import os
import sys
sys.path.append(os.getcwd())
import json
import sys
from datetime import datetime
import traceback

STRUCTURED_FIELDS = (
    "raw_output",
    "action_thinking",
    "action",
    "message_thinking",
    "message",
    "memory_thinking",
    "memory",
    "satisfied_task",
)

class JsonlLogger:
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        parent_dir = os.path.dirname(os.path.abspath(jsonl_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.file = open(jsonl_path, 'w', encoding='utf-8')
        self._structured_path = self._derive_structured_path(jsonl_path)
        self._structured_data = {}

    def _derive_structured_path(self, jsonl_path):
        if jsonl_path.endswith('_logs.jsonl'):
            return jsonl_path[:-10] + '_agent_steps.json'
        if jsonl_path.endswith('.jsonl'):
            return jsonl_path[:-6] + '_agent_steps.json'
        return jsonl_path + '_agent_steps.json'
        
    def log(self, message, source_file=None, line_num=None, step=None, agent_id=None):
        """记录日志到 jsonl 文件，同时输出到控制台"""
        # 获取调用栈信息
        if source_file is None or line_num is None:
            try:
                frame = sys._getframe(2)  # 跳过 log_print 和实际调用函数
                source_file = frame.f_code.co_filename
                line_num = frame.f_lineno
            except:
                source_file = "unknown"
                line_num = 0
        
        # 只保留文件名，不包含完整路径
        source_file = source_file.split('/')[-1] if '/' in source_file else source_file.split('\\')[-1]
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "source_file": source_file,
            "line_num": line_num,
            "message": str(message)
        }
        
        if step is not None:
            log_entry["step"] = step
        if agent_id is not None:
            log_entry["agent_id"] = agent_id
            
        # 写入 jsonl 文件
        self.file.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        self.file.flush()
        
        # 同时输出到控制台
        print(message)
        
    def log_print(self, *args, **kwargs):
        """替代 print 的函数，记录到 jsonl 并输出到控制台"""
        # 获取额外的上下文信息
        structured = kwargs.pop('structured', None)
        step = kwargs.pop('step', None)
        agent_id = kwargs.pop('agent_id', None)
        source_file = kwargs.pop('source_file', None)
        line_num = kwargs.pop('line_num', None)
        
        if structured is not None:
            self.log_structured(structured, step=step, agent_id=agent_id)

        # 格式化消息
        message = ' '.join(str(arg) for arg in args)
        
        self.log(message, source_file=source_file, line_num=line_num, step=step, agent_id=agent_id)

    def log_structured(self, structured, step=None, agent_id=None):
        if step is None or agent_id is None:
            return
        step_key = f"step_{step}"
        agent_key = f"agent_{agent_id}"
        step_entry = self._structured_data.setdefault(step_key, {})
        agent_entry = step_entry.get(agent_key)
        if agent_entry is None:
            agent_entry = {field: "" for field in STRUCTURED_FIELDS}
            step_entry[agent_key] = agent_entry
        for key, value in structured.items():
            if value is None:
                continue
            agent_entry[key] = value
        self._flush_structured()

    def _flush_structured(self):
        if not self._structured_path:
            return
        with open(self._structured_path, 'w', encoding='utf-8') as f:
            json.dump(self._structured_data, f, indent=2, ensure_ascii=False)
        
    def close(self):
        if self.file:
            self.file.close()
            
    def __del__(self):
        self.close()

# 全局日志记录器
_global_logger = None

def init_logger(jsonl_path):
    """初始化全局日志记录器"""
    global _global_logger
    _global_logger = JsonlLogger(jsonl_path)
    return _global_logger

def get_logger():
    """获取全局日志记录器"""
    return _global_logger

def log_print(*args, **kwargs):
    """全局日志打印函数，自动获取调用位置"""
    structured = kwargs.pop('structured', None)
    if _global_logger:
        # 如果没有提供 source_file 和 line_num，自动获取
        if 'source_file' not in kwargs or 'line_num' not in kwargs:
            try:
                frame = sys._getframe(1)  # 获取调用 log_print 的帧
                if 'source_file' not in kwargs:
                    source_file = frame.f_code.co_filename
                    source_file = source_file.split('/')[-1] if '/' in source_file else source_file.split('\\')[-1]
                    kwargs['source_file'] = source_file
                if 'line_num' not in kwargs:
                    kwargs['line_num'] = frame.f_lineno
            except:
                if 'source_file' not in kwargs:
                    kwargs['source_file'] = 'unknown'
                if 'line_num' not in kwargs:
                    kwargs['line_num'] = 0
        if structured is not None:
            step = kwargs.get('step')
            agent_id = kwargs.get('agent_id')
            _global_logger.log_structured(structured, step=step, agent_id=agent_id)
        if args:
            _global_logger.log_print(*args, **kwargs)
    else:
        # 如果没有初始化日志记录器，只输出到控制台
        kwargs.pop('step', None)
        kwargs.pop('agent_id', None)
        kwargs.pop('source_file', None)
        kwargs.pop('line_num', None)
        if args:
            print(*args, **kwargs)
