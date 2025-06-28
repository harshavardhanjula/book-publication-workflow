"""Reinforcement Learning agent for content refinement suggestions."""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

class RLContentAgent:
    def __init__(self, model_path: Optional[Path] = None):
        
        self.model_path = model_path or Path("models/rl_agent/model.json")
        self.actions = [
            "Make the text more concise",
            "Enhance descriptions",
            "Improve flow and transitions",
            "Add more details",
            "Simplify language",
            "Make it more engaging"
        ]
        self.q_table = self._init_q_table()
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        
        self._load_model()
    
    def _init_q_table(self) -> Dict[str, List[float]]:
        return {
            "beginning": [0.0] * len(self.actions),
            "middle": [0.0] * len(self.actions),
            "end": [0.0] * len(self.actions)
        }
    
    def _get_state(self, content: str) -> str:
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return "beginning"
            
        if len(lines) <= 3:
            return "end"
            
        if len([line for line in lines[:3] if line.endswith(('.', '!', '?'))]) < 2:
            return "beginning"
            
        return "middle"
    
    def get_suggestion(self, content: str) -> Tuple[str, int]:
       
        state = self._get_state(content)
        
        if random.random() < self.exploration_rate:
            action_idx = random.randint(0, len(self.actions) - 1)
        else:
            action_idx = np.argmax(self.q_table[state])
        
        return self.actions[action_idx], action_idx
    
    def update_model(self, state: str, action_idx: int, reward: float, next_state: str):
       
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.learning_rate * td_error
        
        self.exploration_rate = max(self.min_exploration, 
                                  self.exploration_rate * self.exploration_decay)
    
    def save_model(self):
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, 'w') as f:
                json.dump({
                    'q_table': self.q_table,
                    'exploration_rate': self.exploration_rate
                }, f, indent=2)
            logger.info(f"Saved RL model to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving RL model: {e}")
    
    def _load_model(self):
        try:
            if self.model_path.exists():
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                    self.q_table = data.get('q_table', self._init_q_table())
                    self.exploration_rate = data.get('exploration_rate', self.exploration_rate)
                logger.info(f"Loaded RL model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading RL model: {e}")
            self.q_table = self._init_q_table()
            self.exploration_rate = 0.3
