from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field
import openenv_core

class SupportTask(str, Enum):
    EASY = "easy_triage"                     # Agent needs to categorize the ticket correctly
    MEDIUM = "medium_missing_info"           # Agent needs to ask for missing order id
    HARD = "hard_full_resolution"            # Agent needs to check order status and issue refund

class ActionType(str, Enum):
    ASSIGN_CATEGORY = "assign_category"
    REPLY_TO_CUSTOMER = "reply_to_customer"
    CHECK_ORDER_STATUS = "check_order_status"
    ISSUE_REFUND = "issue_refund"

class SupportAction(BaseModel):
    action_type: ActionType = Field(description="The type of action to perform.")
    category: Optional[str] = Field(default=None, description="The category to assign (used with assign_category). Valid options: 'Billing', 'Technical', 'General'")
    reply_text: Optional[str] = Field(default=None, description="The text to send to the customer (used with reply_to_customer).")
    order_id: Optional[str] = Field(default=None, description="The order ID to check or refund (used with check_order_status or issue_refund).")

class SupportObservation(BaseModel):
    ticket_id: str = Field(description="The ID of the current support ticket.")
    ticket_content: str = Field(description="The text content of the problem reported by the customer.")
    system_message: Optional[str] = Field(default=None, description="Feedback from the system after an action (e.g., order status or tool execution result).")
    assigned_category: Optional[str] = Field(default=None, description="The currently assigned category of the ticket.")
    customer_reply: Optional[str] = Field(default=None, description="The most recent message sent to the customer.")

class SupportReward(BaseModel):
    score: float = Field(default=0.0, description="The score achieved so far [0.0, 1.0].")
    done: bool = Field(default=False, description="Whether the episode is complete.")
    error: Optional[str] = Field(default=None, description="Any error message if an invalid action was taken.")
