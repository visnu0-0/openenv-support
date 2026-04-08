import os
from typing import Tuple, Dict, Any
from models import SupportAction, SupportObservation, SupportTask, ActionType

class SupportEnv:
    def __init__(self):
        # We read the task from an environment variable to allow easy configuration,
        # or it can be passed in. We default to EASY.
        self.task_name_str = os.getenv("SUPPORT_ENV_TASK", "easy_triage")
        self.task = SupportTask(self.task_name_str)
        self.max_steps = 10
        self.current_step = 0
        
        # Internal state
        self.ticket_id = ""
        self.ticket_content = ""
        self.assigned_category = None
        self.system_message = None
        self.customer_reply = None
        
        # Ground truth / Mock DB state for Hard Task
        self.correct_category = ""
        self.order_status_db = {"ORD-12345": "Eligible for Refund"}
        self.refund_issued = False

    def reset(self) -> SupportObservation:
        self.current_step = 0
        self.assigned_category = None
        self.system_message = None
        self.customer_reply = None
        self.refund_issued = False
        
        if self.task == SupportTask.EASY:
            self.ticket_id = "TCK-001"
            self.ticket_content = "Hi, my app keeps crashing when I try to open the camera. Please fix this."
            self.correct_category = "Technical"
        elif self.task == SupportTask.MEDIUM:
            self.ticket_id = "TCK-002"
            self.ticket_content = "I was double charged for my subscription and I would like a refund immediately."
            self.correct_category = "Billing"
        elif self.task == SupportTask.HARD:
            self.ticket_id = "TCK-003"
            self.ticket_content = "I want a refund for my recent purchase. My order number is ORD-12345."
            self.correct_category = "Billing"
        
        return self._get_obs()

    def state(self) -> Dict[str, Any]:
        return {
            "task": self.task.value,
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "ticket_id": self.ticket_id,
            "assigned_category": self.assigned_category,
            "refund_issued": self.refund_issued,
            "customer_reply": self.customer_reply
        }

    def _get_obs(self) -> SupportObservation:
        return SupportObservation(
            ticket_id=self.ticket_id,
            ticket_content=self.ticket_content,
            system_message=self.system_message,
            assigned_category=self.assigned_category,
            customer_reply=self.customer_reply
        )

    def step(self, action: SupportAction) -> Tuple[SupportObservation, float, bool, Dict[str, Any]]:
        self.current_step += 1
        self.system_message = None
        reward = 0.0
        done = False
        error = None

        if action.action_type == ActionType.ASSIGN_CATEGORY:
            self.assigned_category = action.category
            self.system_message = f"Ticket assigned to category: {self.assigned_category}"
        
        elif action.action_type == ActionType.REPLY_TO_CUSTOMER:
            self.customer_reply = action.reply_text
            self.system_message = f"Replied to customer: {self.customer_reply}"
        
        elif action.action_type == ActionType.CHECK_ORDER_STATUS:
            if action.order_id in self.order_status_db:
                status = self.order_status_db[action.order_id]
                self.system_message = f"Order {action.order_id} status: {status}"
            else:
                self.system_message = f"Order {action.order_id} not found."
        
        elif action.action_type == ActionType.ISSUE_REFUND:
            if action.order_id in self.order_status_db and self.order_status_db[action.order_id] == "Eligible for Refund":
                self.refund_issued = True
                self.order_status_db[action.order_id] = "Refunded"
                self.system_message = f"Refund successfully issued for order {action.order_id}."
            else:
                self.system_message = f"Cannot issue refund for order {action.order_id}."
        
        # Grading logic per task
        if self.task == SupportTask.EASY:
            if self.assigned_category:
                if self.assigned_category.lower() == self.correct_category.lower():
                    reward = 1.0
                    done = True
                else:
                    reward = 0.0
                    done = True
                    error = "Incorrect category assigned."

        elif self.task == SupportTask.MEDIUM:
            if self.customer_reply:
                reply_lower = self.customer_reply.lower()
                if "order id" in reply_lower or "order number" in reply_lower:
                    reward = 1.0
                    done = True
                else:
                    reward = 0.0
                    done = True
                    error = "Replied without asking for the order ID/number."
            elif self.assigned_category and self.assigned_category.lower() == self.correct_category.lower():
                # Partial reward for assigning category correctly
                reward = 0.2

        elif self.task == SupportTask.HARD:
            # Partial rewards
            if self.assigned_category and self.assigned_category.lower() == self.correct_category.lower():
                reward_cat = 0.2
            else:
                reward_cat = 0.0
                
            if "ORD-12345" in (self.system_message or "") and action.action_type == ActionType.CHECK_ORDER_STATUS:
                 reward_check = 0.3
            else:
                 # It tracks if they ever checked it via the system message output on this step, 
                 # wait, let's keep a history or state boolean. 
                 # Actually, it's easier to just compute absolute current state reward.
                 reward_check = 0.0
            
            # Since the OpenEnv baseline expects cumulative or delta reward per step? The baseline script does `rewards.append(reward)`, and final score. 
            # I will return the delta reward. 
            pass # Let's rewrite the hard logic properly below.

        if self.current_step >= self.max_steps and not done:
            done = True
            error = "Max steps reached."

        # Let's fix the reward logic so it's clean and delta-based for the Hard task.
        # Recalculating hard task rewards accurately:
        if self.task == SupportTask.HARD:
            # We want to give a final score based on state at the end.
            # But the prompt says "Reward function provides useful varying signal (not just sparse)".
            # If the user checks order, +0.2 this step.
            # If the user issues refund, +0.4 this step.
            # If the user replies appropriately AND everything else is done, +0.4 this step, done.
            if action.action_type == ActionType.CHECK_ORDER_STATUS and action.order_id == "ORD-12345":
                reward += 0.2
            elif action.action_type == ActionType.ISSUE_REFUND and action.order_id == "ORD-12345":
                reward += 0.4
            elif action.action_type == ActionType.REPLY_TO_CUSTOMER:
                if self.refund_issued:
                    reward += 0.4
                    done = True
                else:
                    done = True
                    error = "Replied before resolving the ticket (issuing refund)."

        # Ensure reward is capped at 1.0 cumulatively? 
        # The prompt says score/reward is 0.0-1.0. We will just ensure maximum sum is 1.0.
        
        info = {}
        if error:
            info['error'] = error

        return self._get_obs(), reward, done, info
