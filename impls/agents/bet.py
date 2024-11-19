from typing import Any, Dict, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GPT, GPTConfig, Identity
from einops import rearrange
from functools import partial
import ml_collections
import distrax

class BehaviorTransformerAgent(flax.struct.PyTreeNode):
    """Behavior Transformer agent using a GPT model."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    cluster_centers: jnp.ndarray
    have_fit_kmeans: bool

    def loss(self, batch, grad_params):
        """Compute the loss for behavior cloning using transformer architecture."""
        # Prepare the input sequence
        observations = batch['observations']  # Shape: (batch_size, seq_length, obs_dim)
        actions = batch['actions']  # Shape: (batch_size, seq_length, act_dim)
        goals = batch.get('goals', None)  # Optional

        # Pass through GPT model
        gpt_output = self.network.select('gpt')(observations, params=grad_params)

        # Map GPT output to classification logits and offset predictions
        cbet_preds = self.network.select('map_to_preds')(gpt_output, params=grad_params)
        cbet_logits, cbet_offsets = jnp.split(
            cbet_preds, [self.config.n_clusters], axis=-1
        )
        cbet_offsets = rearrange(
            cbet_offsets, 'N T (K A) -> N T K A', K=self.config.n_clusters
        )

        # Compute probabilities
        cbet_probs = nn.softmax(cbet_logits, axis=-1)

        # Sample cluster centers
        N, T, _ = cbet_probs.shape
        sampled_centers_idx = jax.random.categorical(self.rng, cbet_logits, axis=-1)
        sampled_offsets = cbet_offsets[jnp.arange(N)[:, None], jnp.arange(T), sampled_centers_idx]
        centers = self.cluster_centers[sampled_centers_idx]

        # Reconstruct actions
        reconstructed_actions = centers + sampled_offsets

        if actions is not None:
            # Find closest cluster centers for true actions
            action_bins = self.find_closest_cluster(actions)
            true_offsets = actions - self.cluster_centers[action_bins]

            predicted_offsets = cbet_offsets[jnp.arange(N)[:, None], jnp.arange(T), action_bins]

            # Compute losses
            offset_loss = jnp.mean((predicted_offsets - true_offsets) ** 2)
            cbet_loss = self.focal_loss(
                logits=cbet_logits.reshape(-1, self.config.n_clusters),
                targets=action_bins.reshape(-1),
                gamma=self.config.gamma
            )
            total_loss = cbet_loss + self.config.offset_loss_multiplier * offset_loss

            loss_info = {
                'classification_loss': cbet_loss,
                'offset_loss': offset_loss,
                'total_loss': total_loss,
            }

            if not self.have_fit_kmeans:
                total_loss = 0.0

            return total_loss, loss_info
        else:
            return None, {}

    @jax.jit
    def update(self, batch):
        """Update the agent's parameters."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.loss(batch, grad_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, seed=None):
        """Sample actions using the trained transformer model."""
        gpt_output = self.network.select('gpt')(observations)

        cbet_preds = self.network.select('map_to_preds')(gpt_output)
        cbet_logits, cbet_offsets = jnp.split(
            cbet_preds, [self.config.n_clusters], axis=-1
        )
        cbet_offsets = rearrange(
            cbet_offsets, 'N T (K A) -> N T K A', K=self.config.n_clusters
        )

        cbet_probs = nn.softmax(cbet_logits, axis=-1)

        # Sample cluster centers
        sampled_centers_idx = jax.random.categorical(seed, cbet_logits, axis=-1)
        sampled_offsets = cbet_offsets[jnp.arange(observations.shape[0])[:, None], jnp.arange(cbet_probs.shape[1]), sampled_centers_idx]
        centers = self.cluster_centers[sampled_centers_idx]

        actions = centers + sampled_offsets

        return actions

    def find_closest_cluster(self, action_seq):
        """Find the closest cluster centers for given actions."""
        N, T, _ = action_seq.shape
        flattened_actions = action_seq.reshape(-1, self.config.act_dim)
        cluster_center_distance = jnp.sum(
            (flattened_actions[:, None, :] - self.cluster_centers[None, :, :]) ** 2, axis=-1
        )
        closest_cluster_center = jnp.argmin(cluster_center_distance, axis=-1)
        discretized_action = closest_cluster_center.reshape(N, T)
        return discretized_action

    @staticmethod
    def focal_loss(logits, targets, gamma):
        """Compute the focal loss."""
        log_probs = nn.log_softmax(logits, axis=-1)
        targets_one_hot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
        probs = jnp.exp(log_probs)
        focal_weight = (1 - probs) ** gamma
        loss = -jnp.sum(targets_one_hot * focal_weight * log_probs, axis=-1)
        return jnp.mean(loss)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Initialize the Behavior Transformer Agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Define GPT model
        gpt_def = GPT(config.gpt_config)

        # Define mapping from GPT output to CBET predictions
        map_to_preds_def = MLP(
            hidden_dims=[],
            output_dim=(config.act_dim + 1) * config.n_clusters
        )

        network_info = dict(
            gpt=(gpt_def, ex_observations),
            map_to_preds=(map_to_preds_def, gpt_def.apply({'params': {}}, ex_observations)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items() if v[1] is not None}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config.lr)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize cluster centers (placeholder)
        cluster_centers = jnp.zeros((config.n_clusters, config.act_dim))

        return cls(
            rng,
            network=network,
            config=config,
            cluster_centers=cluster_centers,
            have_fit_kmeans=False
        )
    def get_config():
        config = ml_collections.ConfigDict(
            dict(
                # Agent hyperparameters
                agent_name='bet',  # Agent name
                lr=3e-4,  # Learning rate
                batch_size=1024,  # Batch size
                gpt_config=GPTConfig(
                    block_size=1024,
                    input_dim=256,
                    output_dim=256,
                    n_layer=12,
                    n_head=12,
                    n_embd=768,
                    dropout=0.1
                ),  # GPT configuration
                n_clusters=10,  # Number of clusters
                gamma=2.0,  # Focal loss gamma
                offset_loss_multiplier=1.0,  # Offset loss multiplier
                act_dim=2,  # Action dimension
            )
        )
        return config