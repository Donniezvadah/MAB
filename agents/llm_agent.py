import openai
import numpy as np
from agents.base_agent import BaseAgent

class LLMAgent(BaseAgent):
    def __init__(self, n_arms: int, api_key: str, model: str = "o3-mini", prompt_template: str = None, horizon: int = 200):
        super().__init__(n_arms)
        self.api_key = api_key
        self.model = model
        self.horizon = horizon
        self.prompt_template = prompt_template or self.default_prompt_template()
        self.client = openai.OpenAI(api_key=self.api_key)
        self.history = []  # List of (action, reward) tuples

    def default_prompt_template(self):
        return (
            "You are playing a matrix game for {horizon} rounds. There are {n_arms} number of actions.\n"
            "At each round, you need to choose a policy; it specifies your probability of choosing each action.\n"
            "This policy should be a {n_arms}-dimensional vector, with the sum of its components equaling 1.\n"
            "After that, you are shown the reward for choosing each action. Remember, the reward vector is determined by an external system and can vary across rounds.\n"
            "It is not decided by what policies you have chosen. The reward vector is also {n_arms}-dimensional.\n"
            "You can adjust your policy based on the reward vectors for all previous rounds. You're required to provide your policy in numeric format.\n"
            "Your response's last line should be formatted as 'Policy: [your {n_arms}-dimensional policy]'.\n"
            "Let's think step by step. Explicitly examining history is important. Please explain how you chose the policy by guessing what reward you might receive for each action according to the history.\n"
            "Here is the history of actions and rewards so far: {history}.\n"
        )

    def select_action(self) -> int:
        prompt = self.prompt_template.format(
            n_arms=self.n_arms,
            horizon=self.horizon,
            history=self.format_history()
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=100
        )
        # Try to extract the policy vector from the response
        try:
            content = response.choices[0].message.content.strip()
            # Find the line starting with 'Policy:'
            policy_line = [line for line in content.splitlines() if line.strip().lower().startswith('policy:')]
            if not policy_line:
                raise ValueError("No policy line found in LLM response.")
            policy_str = policy_line[0].split(':', 1)[1].strip().replace('[', '').replace(']', '')
            policy = np.array([float(x) for x in policy_str.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()])
            if len(policy) != self.n_arms or not np.isclose(policy.sum(), 1.0):
                raise ValueError("Policy vector is not valid.")
            # Sample an arm according to the policy
            action = np.random.choice(self.n_arms, p=policy)
        except Exception:
            action = np.random.randint(self.n_arms)  # Fallback: random action
        return action

    def update(self, arm: int, reward: float):
        self.history.append((arm, reward))

    def reset(self):
        super().reset()
        self.history = []

    def format_history(self):
        # Format the history as a string for the prompt
        if not self.history:
            return "None yet."
        return "; ".join([f"Action: {a}, Reward: {r}" for a, r in self.history]) 