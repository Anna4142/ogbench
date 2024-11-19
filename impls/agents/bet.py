from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from functools import partial
from einops import rearrange
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GPT # Import GPT from networks.py
from flax import linen as nn


class BETAgent(flax.struct.PyTreeNode):
    """Behavior Transformer (BET) agent using a GPT model."""

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

        if actions is not None:
            N, T, _ = actions.shape

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
        N, T, _ = observations.shape
        sampled_centers_idx = jax.random.categorical(seed, cbet_logits, axis=-1)
        sampled_offsets = cbet_offsets[jnp.arange(N)[:, None], jnp.arange(T), sampled_centers_idx]
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
        """Initialize the BETAgent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Define GPT model using parameters from config
        gpt_def = GPT(config)

        # Define mapping from GPT output to CBET predictions
        features = config.hidden_dims + [(config.act_dim + 1) * config.n_clusters]
        map_to_preds_def = MLP(
            hidden_dims=features,
            activate_final=False,
            layer_norm=config['layer_norm']

        )

        network_info = dict(
            gpt=(gpt_def, ex_observations),
            map_to_preds=(map_to_preds_def, None),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items() if v[1] is not None}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config.lr)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Initialize cluster centers (should be set after fitting KMeans on actions)
        cluster_centers = jnp.zeros((config.n_clusters, config.act_dim))

        return cls(
            rng,
            network=network,
            config=config,
            cluster_centers=cluster_centers,
            have_fit_kmeans=False
        )


def get_config():
    """Get the default configuration for the BETAgent."""
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters
            # Agent hyperparameters
            agent_name='bet',  # Agent name
            lr=3e-4,  # Learning rate
            batch_size=1024,  # Batch size
            n_clusters=10,  # Number of clusters
            gamma=2.0,  # Focal loss gamma
            offset_loss_multiplier=1.0,  # Offset loss multiplier
            act_dim=2,  # Action dimension
            discount=0.99,  # Discount factor
            hidden_dims=[256, 256],
            discrete=False,  # Whether the action space is discrete
            layer_norm=True

            # GPT configuration parameters
            block_size=1024,  # Maximum sequence length
            input_dim=256,  # Input dimension
            output_dim=256,  # Output dimension
            n_layer=12,  # Number of transformer layers
            n_head=12,  # Number of attention heads
            n_embd=768,  # Embedding dimension
            dropout=0.1,  # Dropout rate

            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
