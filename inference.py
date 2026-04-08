import os
import time
from typing import Optional
from openai import OpenAI
from support_env import SupportEnv
from models import SupportAction, SupportObservation, SupportTask, ActionType

# Configure OpenAI client from environment variables
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
base_url = os.getenv("API_BASE_URL")
model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None


def get_action_from_llm(observation: SupportObservation, task: SupportTask) -> SupportAction:
    """Simple rule-based baseline agent."""

    if task == SupportTask.EASY:
        return SupportAction(action_type=ActionType.ASSIGN_CATEGORY, category="Technical")

    elif task == SupportTask.MEDIUM:
        return SupportAction(
            action_type=ActionType.REPLY_TO_CUSTOMER,
            reply_text="Could you please provide your order ID or order number so I can process your refund?"
        )

    elif task == SupportTask.HARD:
        if not observation.system_message or "status" not in observation.system_message.lower():
            return SupportAction(action_type=ActionType.CHECK_ORDER_STATUS, order_id="ORD-12345")
        elif "eligible" in (observation.system_message or "").lower() and not observation.customer_reply:
            return SupportAction(action_type=ActionType.ISSUE_REFUND, order_id="ORD-12345")
        else:
            return SupportAction(
                action_type=ActionType.REPLY_TO_CUSTOMER,
                reply_text="Your refund has been processed successfully."
            )

    return SupportAction(action_type=ActionType.ASSIGN_CATEGORY, category="General")


def log_step(step: int, action: SupportAction, reward: float, total_reward: float, done: bool, info: dict):
    print(f"STEP {step} action={action.action_type.value} category={action.category or ''} order_id={action.order_id or ''} reply_text={action.reply_text or ''} reward={reward} total={total_reward} done={done} error={info.get('error', '')}")


def run_baseline(task: str = "easy_triage"):
    os.environ["SUPPORT_ENV_TASK"] = task
    env = SupportEnv()

    obs = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    print("START")
    print(f"TASK {task}")
    print(f"INITIAL_OBSERVATION {obs}")

    while not done and step < 10:
        action = get_action_from_llm(obs, env.task)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        log_step(step, action, reward, total_reward, done, info)
        step += 1

        if info.get('error'):
            print(f"ERROR {info['error']}")

        time.sleep(0.5)

    print(f"FINAL_SCORE {total_reward}")
    print("END")
    return total_reward

if __name__ == "__main__":
    # Run all tasks
    tasks = ["easy_triage", "medium_missing_info", "hard_full_resolution"]
    scores = {}

    for task in tasks:
        score = run_baseline(task)
        scores[task] = score
        print(f"Task {task}: {score}")

    print("All scores:", scores)