import os
import json
from typing import List
from openai import OpenAI
from models import SupportAction, SupportObservation, SupportTask, ActionType
from support_env import SupportEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("SUPPORT_ENV_TASK", "easy_triage")
BENCHMARK = os.getenv("SUPPORT_ENV_BENCHMARK", "support_triage")

def run_episode(env: SupportEnv, task_name: str, client: OpenAI) -> float:
    obs = env.reset()
    done = False
    rewards_history: List[float] = []
    
    # Strictly required START log
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
    
    while not done and env.current_step < env.max_steps:
        # Construct prompt from Observation
        system_prompt = (
            "You are a customer support agent. You must output EXACTLY a valid JSON representing your action. "
            "Valid action_type: 'assign_category', 'reply_to_customer', 'check_order_status', 'issue_refund'. "
            "For 'assign_category', provide 'category'. For 'reply_to_customer', provide 'reply_text'. "
            "For 'check_order_status' and 'issue_refund', provide 'order_id'.\n"
            f"Observation: {obs.model_dump_json()}"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            action_json = response.choices[0].message.content
            action_dict = json.loads(action_json)
            # Make sure no extra fields
            action = SupportAction(**action_dict)
        except Exception as e:
            # Fallback action if model fails
            action = SupportAction(action_type=ActionType.REPLY_TO_CUSTOMER, reply_text="What?")
            action_json = action.model_dump_json()

        # Step environment
        obs, reward, done, info = env.step(action)
        rewards_history.append(reward)
        
        # Log STEP exactly as requested
        # Format: [STEP] step=<n> action=<str> reward=<float> done=<bool> error=<msg|null>
        action_str = f"{action.action_type.value}()"
        reward_str = f"{reward:.2f}"
        done_str = "true" if done else "false"
        error_str = "null" if not info.get("error") else info["error"]
        
        print(f"[STEP] step={env.current_step} action={action_str} reward={reward_str} done={done_str} error={error_str}", flush=True)

    # Episode ended
    # Final score is the sum of rewards (bounded between 0 and 1 theoretically based on env)
    total_score = sum(rewards_history)
    success_str = "true" if total_score > 0 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_history)
    
    print(f"[END] success={success_str} steps={env.current_step} score={total_score:.2f} rewards={rewards_str}", flush=True)

    return total_score

def main():
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = SupportEnv()
    
    # We will try all 3 tasks as a baseline check.
    # The submission requires hitting 3 tasks. The runner might inject task via ENV so we run that.
    run_episode(env, env.task_name_str, client)

if __name__ == "__main__":
    main()
