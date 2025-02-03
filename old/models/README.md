# Models

This folder stores the trained AI models for demand prediction and control logic.

## Structure:
- `demand_predictor.pkl`: A pre-trained model for predicting energy demand (example format: Pickle).
- `optimizer_agent.onnx`: The reinforcement learning or optimization model for battery and appliance control.

**Usage**:
1. Place your pre-trained models in this folder.
2. Reference these models in your scripts (e.g., `predict.py` or `control_agent.py`).
3. Do not push large model files to GitHub. Instead, provide download links in the main `README.md`.
