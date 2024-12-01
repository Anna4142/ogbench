"""Decision Transformer implementation."""
from typing import Any, Optional, Sequence
import dataclasses
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import ml_collections
import optax
from flax.training import train_state
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


@dataclasses.dataclass
class DecisionTransformerConfig:
    """Configuration for Decision Transformer model."""
    max_length: int  # Maximum sequence length
    state_dim: int  # State dimensionality
    action_dim: int  # Action dimensionality
    n_layer: int  # Number of transformer layers
    n_head: int  # Number of attention heads
    n_embd: int  # Embedding dimension
    dropout: float  # Dropout rate
    action_tanh: bool  # Whether to use tanh on actions


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    config: DecisionTransformerConfig

    def setup(self):
        assert self.config.n_embd % self.config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Dense(3 * self.config.n_embd)
        # output projection
        self.c_proj = nn.Dense(self.config.n_embd)
        # regularization
        self.resid_dropout = nn.Dropout(self.config.dropout)

        self.n_head = self.config.n_head
        self.n_embd = self.config.n_embd
        self.dropout = self.config.dropout

    def __call__(self, x, mask=None, deterministic=True):
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scale = 1.0 / jnp.sqrt(k.shape[-1])
        att = (q @ jnp.swapaxes(k, -2, -1)) * scale

        # causal mask to ensure that attention is only applied to the left in the input sequence
        causal_mask = jnp.triu(jnp.ones((T, T)), k=1)
        if mask is not None:
            causal_mask = causal_mask | mask

        att = jnp.where(causal_mask[None, None, :, :], float('-inf'), att)
        att = nn.softmax(att, axis=-1)
        att = self.resid_dropout(att, deterministic=deterministic)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y, deterministic=deterministic)
        return y


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    config: DecisionTransformerConfig

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.attn = CausalSelfAttention(self.config)
        self.ln2 = nn.LayerNorm()
        self.mlp = nn.Sequential([
            nn.Dense(4 * self.config.n_embd),
            nn.gelu,
            nn.Dense(self.config.n_embd),
            nn.Dropout(self.config.dropout)
        ])

    def __call__(self, x, mask=None, deterministic=True):
        x = x + self.attn(self.ln1(x), mask=mask, deterministic=deterministic)
        x = x + self.mlp(self.ln2(x), deterministic=deterministic)
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer implementation."""
    config: DecisionTransformerConfig

    def setup(self):
        self.token_emb = nn.Dense(self.config.n_embd)
        self.pos_emb = nn.Embed(
            num_embeddings=self.config.max_length,
            features=self.config.n_embd
        )

        self.state_encoder = nn.Dense(self.config.n_embd)
        self.ret_encoder = nn.Dense(self.config.n_embd)
        self.action_encoder = nn.Dense(self.config.n_embd)

        self.blocks = [Block(self.config) for _ in range(self.config.n_layer)]

        self.ln_f = nn.LayerNorm()
        self.action_head = nn.Dense(self.config.action_dim)

        self.seq_len = 0

    def get_mask(self, states, actions, returns_to_go, padded=True):
        """Create attention mask based on which tokens are padded."""
        if padded:
            attention_mask = jnp.where(states != 0, 0, 1)  # assume 0 is pad token
            attention_mask = attention_mask.sum(-1) > 0
        else:
            attention_mask = None
        return attention_mask

    def __call__(self, states, actions, returns_to_go, timesteps=None, deterministic=True):
        batch_size = states.shape[0]
        self.seq_len = states.shape[1]

        if timesteps is None:
            timesteps = jnp.arange(states.shape[1])

        # Get attention mask
        attention_mask = self.get_mask(states, actions, returns_to_go)

        # Encode each modality
        state_embeddings = self.state_encoder(states)
        action_embeddings = self.action_encoder(actions)
        returns_embeddings = self.ret_encoder(returns_to_go.reshape(-1, 1))
        returns_embeddings = returns_embeddings.reshape(batch_size, -1, self.config.n_embd)

        # Time embedding
        time_embeddings = self.pos_emb(timesteps)

        # Combine all embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Stack modalities
        stacked_inputs = jnp.stack(
            (returns_embeddings, state_embeddings, action_embeddings), axis=1
        ).reshape(batch_size, 3 * self.seq_len, self.config.n_embd)

        # Run through transformer blocks
        x = stacked_inputs
        for block in self.blocks:
            x = block(x, mask=attention_mask, deterministic=deterministic)
        x = self.ln_f(x)

        # Get predictions
        action_preds = self.action_head(x[:, 1::3])  # predict only action outputs

        if self.config.action_tanh:
            action_preds = nn.tanh(action_preds)

        return action_preds


class DecisionTransformerAgent(flax.struct.PyTreeNode):
    """Decision Transformer agent implementation."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def train_loss(self, batch, grad_params=None):
        """Compute training loss."""
        states = batch['observations']
        actions = batch['actions']
        returns_to_go = batch['returns_to_go']
        timesteps = batch.get('timesteps', None)

        action_preds = self.network.select('dt')(
            states, actions, returns_to_go, timesteps=timesteps
        )

        loss = jnp.mean((action_preds - actions) ** 2)

        return loss, {'train_loss': loss}

    @jax.jit
    def update(self, batch):
        """Update agent parameters."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.train_loss(batch, grad_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def get_action(self, states, actions, returns_to_go, timesteps=None):
        """Get actions from the model."""
        action_preds = self.network.select('dt')(
            states, actions, returns_to_go, timesteps=timesteps
        )
        return action_preds[:, -1]  # return last prediction

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new Decision Transformer agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        dt_config = DecisionTransformerConfig(
            max_length=config.max_length,
            state_dim=ex_observations.shape[-1],
            action_dim=ex_actions.shape[-1],
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd,
            dropout=config.dropout,
            action_tanh=config.action_tanh
        )

        dt_def = DecisionTransformer(dt_config)

        # Initialize network
        network_info = {
            'dt': (dt_def, (ex_observations, ex_actions, jnp.zeros_like(ex_observations[:, :, 0:1])))
        }
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config.lr)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=config)


def get_default_config():
    """Get default configuration."""
    config = ml_collections.ConfigDict()

    # Model configuration
    config.max_length = 20  # Maximum sequence length
    config.n_layer = 4  # Number of transformer layers
    config.n_head = 4  # Number of attention heads
    config.n_embd = 128  # Embedding dimension
    config.dropout = 0.1  # Dropout rate
    config.action_tanh = True  # Use tanh activation on actions

    # Training configuration
    config.lr = 1e-4  # Learning rate
    config.batch_size = 64  # Batch size
    config.train_steps = 100000  # Number of training steps

    

    return config