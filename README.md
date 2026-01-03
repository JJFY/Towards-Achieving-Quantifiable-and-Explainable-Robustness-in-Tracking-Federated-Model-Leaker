# Towards-Achieving-Quantifiable-and-Explainable-Robustness-in-Tracking-Federated-Model-Leaker
These are the codes for the paper: Towards Achieving Quantifiable and Explainable Robustness in Tracking Federated Model Leaker: A Reed-Solomon Codes-Based Approach


main.py runs federated learning with per-client watermark insertion, model aggregation, and tracking, logging all results and saving global and client models.
Fed.py implements the federated learning algorithms.
RS64.py implements the RS code algorithms (generation and error correction).
dataClass.py is for code embedding datasets composition.
function.py implements some function used in main.py.
generate_rscode.py is used to generate the code for each client.
resnet.py is the model structure used in main.py.
