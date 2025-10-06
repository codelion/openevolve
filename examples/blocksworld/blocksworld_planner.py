"""
Blocksworld domain with a random search planner.
This file contains both the simulator and the planner to be evolved.
"""
import random
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class State:
    """Immutable blocksworld state representation."""
    on: Tuple[Tuple[str, str], ...]  # (block, location) pairs, location can be block or 'table'
    clear: Tuple[str, ...]  # blocks with nothing on top
    holding: Optional[str] = None  # block being held (None if hand empty)
    
    def __hash__(self):
        return hash((self.on, self.clear, self.holding))


class BlocksworldSimulator:
    """Simple blocksworld simulator."""
    
    def __init__(self, blocks: List[str]):
        self.blocks = blocks
        self.all_locations = blocks + ['table']
    
    def make_state(self, on_dict: dict, holding: Optional[str] = None) -> State:
        """Create a state from an on-table dictionary."""
        on_tuples = tuple(sorted(on_dict.items()))
        clear = tuple(sorted(b for b in self.blocks if b not in [loc for _, loc in on_dict.items()] and b != holding))
        return State(on=on_tuples, clear=clear, holding=holding)
    
    def get_on_dict(self, state: State) -> dict:
        """Convert state to on-table dictionary."""
        return dict(state.on)
    
    def is_goal(self, state: State, goal: State) -> bool:
        """Check if state satisfies goal (ignoring holding)."""
        return state.on == goal.on
    
    def get_actions(self, state: State) -> List[Tuple[str, str]]:
        """Get all valid actions from current state."""
        actions = []
        on_dict = self.get_on_dict(state)
        
        if state.holding is None:
            # Can pick up any clear block
            for block in state.clear:
                actions.append(('pickup', block))
        else:
            # Can put down on table
            actions.append(('putdown', state.holding))
            # Can stack on any clear block (except the one holding)
            for block in state.clear:
                if block != state.holding:
                    actions.append(('stack', f"{state.holding}_{block}"))
        
        return actions
    
    def apply_action(self, state: State, action: Tuple[str, str]) -> Optional[State]:
        """Apply action to state, return new state or None if invalid."""
        action_type, param = action
        on_dict = self.get_on_dict(state)
        
        if action_type == 'pickup':
            block = param
            if state.holding is not None or block not in state.clear:
                return None
            new_on = {k: v for k, v in on_dict.items() if k != block}
            return self.make_state(new_on, holding=block)
        
        elif action_type == 'putdown':
            if state.holding is None or param != state.holding:
                return None
            new_on = dict(on_dict)
            new_on[state.holding] = 'table'
            return self.make_state(new_on, holding=None)
        
        elif action_type == 'stack':
            parts = param.split('_')
            if len(parts) != 2:
                return None
            block_to_stack, target_block = parts
            if state.holding != block_to_stack or target_block not in state.clear:
                return None
            new_on = dict(on_dict)
            new_on[block_to_stack] = target_block
            return self.make_state(new_on, holding=None)
        
        return None


# EVOLVE-BLOCK-START
def plan_blocks(initial_state: State, goal_state: State, simulator: BlocksworldSimulator,
                max_plan_length: int = 1000) -> Optional[List[Tuple[str, str]]]:
    """
    Random search planner for blocksworld.
    Tries random action sequences until it finds a solution.
    Runs indefinitely until solution found or timeout (controlled by evaluator).

    Args:
        initial_state: Starting state
        goal_state: Goal state
        simulator: Blocksworld simulator
        max_plan_length: Maximum length of each random plan attempt

    Returns:
        List of actions that solve the problem, or None if timeout
    """
    while True:  # Run until solution found or timeout
        state = initial_state
        plan = []

        # Try a random sequence of actions
        for _ in range(max_plan_length):
            if simulator.is_goal(state, goal_state):
                return plan

            actions = simulator.get_actions(state)
            if not actions:
                break

            action = random.choice(actions)
            next_state = simulator.apply_action(state, action)

            if next_state is not None:
                plan.append(action)
                state = next_state
            else:
                break

        # Check if we reached goal
        if simulator.is_goal(state, goal_state):
            return plan
# EVOLVE-BLOCK-END


def solve_problem(blocks: List[str], initial_on: dict, goal_on: dict) -> dict:
    """
    Main function to solve a blocksworld problem.
    Returns a dict with plan and success status.
    """
    sim = BlocksworldSimulator(blocks)
    initial_state = sim.make_state(initial_on)
    goal_state = sim.make_state(goal_on)
    
    plan = plan_blocks(initial_state, goal_state, sim)
    
    return {
        'success': plan is not None,
        'plan': plan,
        'plan_length': len(plan) if plan else 0
    }


if __name__ == "__main__":
    # Example problem: stack A on B on C
    blocks = ['A', 'B', 'C']
    initial = {'A': 'table', 'B': 'table', 'C': 'table'}
    goal = {'A': 'B', 'B': 'C', 'C': 'table'}
    
    result = solve_problem(blocks, initial, goal)
    print(f"Success: {result['success']}")
    print(f"Plan length: {result['plan_length']}")
    if result['plan']:
        print(f"Plan: {result['plan']}")