# DreamerV3 Implementation Plan

## 1. Core Components
- **RSSM (Recurrent State Space Model)**:
    - Deterministic state (GRU).
    - Stochastic state (Discrete categorical latents).
    - Transition predictor (Prior).
    - Representation predictor (Posterior).
- **Encoder**: CNN (for image observations) or MLP (for vector observations) mapping to RSSM input.
- **Decoders**:
    - Image Decoder (CNN).
    - Reward Predictor (MLP).
    - Continue Predictor (MLP).
- **Actor-Critic**:
    - Actor: Proposes actions in latent space.
    - Critic: Predicts values of latent states.

## 2. Key Techniques
- **Symlog Transformation**: Applied to rewards and values.
- **Discrete Latents**: Using `independent_categorical` or similar.
- **KL Balancing**: 0.1 for KL loss.
- **Two-Hot Encoding / Value Rescaling**: For rewards and values.

## 3. Files to Create
- `networks.py`: Contains all NN modules.
- `rssm.py`: Categorical RSSM implementation.
- `utils.py`: Math utilities (symlog, symexp, etc.).
- `agent.py`: Dreamer agent wrapper.
- `train.py`: Main execution script.
