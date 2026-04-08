---
title: Support Triage Env
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Client Support Ticket Triage (OpenEnv)

This environment simulates a real-world customer support agent processing support tickets. It implements a complete OpenEnv setup including tasks ranging from easy to hard, with automated grading and appropriate baseline models.

## Motivation & Domain
Customer Support is one of the most practical and immediate applications of autonomous agents. Simulating this environment requires the agent to demonstrate reading comprehension, API/Tool use, multi-step reasoning, and procedural instruction following. It is highly practical and directly applicable as a real-world task limit.

## Tasks & Difficulty
We provide 3 tasks that test different levels of capability:
- **Easy (`easy_triage`)**: The agent must read a customer ticket ("My app is crashing") and use `assign_category` to route it to the correct department (e.g. "Technical").
- **Medium (`medium_missing_info`)**: A customer wants a refund but does not provide an order number. The agent must realize the missing info and use `reply_to_customer` to ask for the order ID.
- **Hard (`hard_full_resolution`)**: The customer supplies an order ID explicitly requesting a refund. The agent must use `check_order_status` on that ID, verify if it is eligible, use `issue_refund`, and then follow up with a `reply_to_customer` affirming the action. 

## Action Space
Uses standard Pydantic typed inputs.
`action_type` string literals: `"assign_category"`, `"reply_to_customer"`, `"check_order_status"`, `"issue_refund"`. Additional fields include `category`, `reply_text`, and `order_id` which are utilized conditionally based on action type.

## Observation Space
Provides continuous streaming of internal state:
`ticket_id`, `ticket_content`, `system_message` (from previous actions), `assigned_category`, `customer_reply`. 

## Baseline Quickstart
Ensure your `.env` contains:
```env
API_BASE_URL="<your_provider_url>"
MODEL_NAME="<your_model>"
HF_TOKEN="<your_hf_token>"
```
Install requirements:
```bash
pip install -r requirements.txt
```

Run baseline:
```bash
python inference.py
```

Run OpenEnv locally:
```bash
docker build -t openenv-support .
docker run -p 7860:7860 openenv-support
```
