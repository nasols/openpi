import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override
import numpy as np 
from typing import Generic, TypeVar
import torch

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi05_config
from openpi.models import tokenizer as _tokenizer
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi import transforms as _transforms

from functools import partial

ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)



logger = logging.getLogger("openpi")

def _gen_sample_action(action_horizon, action_dim) -> _model.Actions:
        t = jnp.arange(action_horizon)
        action = -jnp.sin((t / action_horizon) * (jnp.pi/2)) 
        action = jnp.broadcast_to(action[None, :, None], (1, action_horizon, action_dim))
        return action

def _noise_around_action(action, rng : at.KeyArrayLike): 
    """
    Given a action chunk we produce noise patterns around it. 
    The noise pattern is a normal distribution with mean at the action chunk and a fixed or varying standard deviation. 
    If varying, we increase the deviation over time, i.e. at the beginning of the action chunk we have low noise and at the end we have high noise.
    """

    action_horizon = action.shape[1]
    action_dim = action.shape[2]
    t = jnp.arange(action_horizon)
    
    noise = jax.random.normal(rng, action.shape) 
    time_weights = (t / action_horizon) * 0.5 + 0.1 # Linearly increasing weights from 0.1 to 0.6
    noise = noise * time_weights[None, :, None] # Scale noise by time weights
    x_t = action + noise

    return x_t

def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)

@jax.vmap
def lef_to_right_map(x, input_mask, attn_mask): 
    seq_len = input_mask.sum()
    shift = -seq_len.astype(int)

    x = jnp.roll(x, shift, axis=0)
    input_mask = jnp.roll(input_mask, shift)
    attn_mask = jnp.roll(attn_mask, shift, axis=(0, 1))

    return x, input_mask, attn_mask

@jax.vmap
def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # Due to vmap, this is operating in a single example (not batch level).
    assert x.ndim == 2
    assert input_mask.ndim == 1
    assert attn_mask.ndim == 2
    assert x.shape[0] == input_mask.shape[0]
    assert attn_mask.shape[0] == attn_mask.shape[1], attn_mask.shape
    seqlen = jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1
    x = jnp.roll(x, -seqlen, axis=0)
    input_mask = jnp.roll(input_mask, -seqlen, axis=0)
    attn_mask = jnp.roll(attn_mask, -seqlen, axis=(0, 1))
    return x, input_mask, attn_mask


@jax.vmap
def _pack_sequence_by_first_true(tokens, mask):
    """Left-pack a contiguous True span in `mask` while preserving token order."""
    shift = jnp.argmax(mask).astype(int)
    packed_tokens = jnp.roll(tokens * mask, -shift)
    packed_mask = jnp.roll(mask, -shift)
    return packed_tokens, packed_mask

@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b *ad"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "*b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    original_shape = pos.shape # (32, 15)
    logger.log(level=103, msg=f"FROM POSEMB_SINCOS: Original position shape: {original_shape}")
    pos_flat = pos.reshape(-1)
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2) # 512 arr
    period = min_period * (max_period / min_period) ** fraction  # 512 arr 
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos_flat, # (480,)
        1.0 / period * 2 * jnp.pi, # (512)
        precision=jax.lax.Precision.HIGHEST,
    )
    emb = jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
    emb = emb.reshape(*original_shape, embedding_dim) # (32, 15, 1024)
    return emb


class Pi05(_model.BaseModel):
    """
    Class for the Pi_0.5 model!!
    """
    def __init__(self, config: pi05_config.Pi05Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = True
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.config = config
        self.model_type = config.model_type
        self.ki_mode = config.ki_mode
        self.ki_fast_loss_weight = config.ki_fast_loss_weight
        self.hi_mode = config.hi_mode
        # For hierarchical mode, use same weight as FAST loss (both are text generation)
        self.ki_subtask_loss_weight = config.ki_fast_loss_weight 
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX( # Init PaliGemma model
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # Create tokenizer for text generation
        #[TODO Gets in the way of running training, belongs to HI-Robot implementation]
        #self.tokenizer = _tokenizer.PaligemmaTokenizer(max_len=config.max_token_len)

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        # Debugging purpose - should be removed later due to overhead 
        # self.tokenizer = _tokenizer.PaligemmaTokenizer(max_len=self.config.max_token_len)

        # Guided inference
        self.max_delay = 5

    def _compute_fast_loss(
        self, 
        prefix_out: at.Float[at.Array, "b s emb"], 
        observation: _model.Observation
    ) -> at.Float[at.Array, ""]:
        """Compute FAST token prediction loss for Knowledge Insulation.
            
        Args:
            prefix_out: Hidden states from VLM forward pass (batch, seq_len, embed_dim)
                        This includes embeddings for ALL tokens: images + text
            observation: Observation containing tokenized_prompt and token_loss_mask
            
        Returns:
            Scalar loss value for FAST token prediction
        """
        if observation.tokenized_prompt is None or observation.token_loss_mask is None:
            return jnp.array(0.0)
        
        # Extract target tokens for next-token prediction (shift by 1)
        ## [1, 2, 3, 4, 5] -> [2, 3, 4, 5]
        target_tokens = observation.tokenized_prompt[:, 1:]  # Shape: (batch, seq_len-1)
        loss_mask = observation.token_loss_mask[:, 1:]  # Shape: (batch, seq_len-1)
        
        # Extract only the text token embeddings from prefix_out
        # prefix_out contains: [image_tokens, text_tokens]
        text_token_len = observation.tokenized_prompt.shape[1]
        text_embeddings = prefix_out[:, -text_token_len:]  # Last text_token_len positions

        
        # Decode hidden states to vocabulary logits
        # Use text_embeddings[:, :-1] to predict next token (no prediction for last token)
        # Memory-efficient: Use einsum with optimal contraction order
        embedding_table = self.PaliGemma.llm.embedder["input_embedding"]
        logits = jnp.einsum(
            'bse,ve->bsv', 
            text_embeddings[:, :-1], 
            embedding_table,
            optimize='optimal'
        )  # Shape: (batch, seq_len-1, vocab_size)
        

        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, target_tokens
        )  # Shape: (batch, seq_len-1)
        
        # Show predicted vs target FAST tokens
        predicted_tokens = jnp.argmax(logits, axis=-1)
        # Find first few positions where loss_mask is True
        mask_indices = jnp.where(loss_mask[0])[0][:5]

        # Apply loss mask: only compute loss on FAST tokens
        masked_loss = per_token_loss * loss_mask
        
        # Normalize by number of masked tokens per sample
        sum_masked_loss = jnp.sum(masked_loss, axis=-1)  # Sum over sequence
        num_tokens = jnp.clip(jnp.sum(loss_mask, axis=-1), 1)  # Avoid division by zero
        batch_loss = sum_masked_loss / num_tokens  # Normalize per sample
        
        # Average over batch
        return jnp.mean(batch_loss)

    def _compute_subtask_loss(self, prefix_out: at.Float[at.Array, "b s emb"], subtask_tokens: at.Int[at.Array, "b s"], subtask_mask: at.Bool[at.Array, "b s"]) -> at.Float[at.Array, ""]:
        """Compute subtask prediction loss with teacher forcing.
        
        Args:
            prefix_out: Hidden states from VLM (batch, seq_len, embed_dim)
                        Contains: [image_tokens, decomposition_prompt_tokens, GT_subtask_tokens]
            subtask_tokens: Ground truth subtask tokens (batch, subtask_len)
                            Full sequence WITHOUT BOS (e.g., ["pick", "up", "the", "cube"])
            subtask_mask: Boolean mask for valid subtask tokens (batch, subtask_len)
            
        Returns:
            Scalar loss for subtask prediction
        """
        if subtask_tokens is None:
            return jnp.array(0.0)  
        
        # For next-token prediction, we need the hidden state BEFORE each token to predict it
        # Extract N+1 positions: the position before first token + all N subtask token positions
        # Example: to predict ["pick", "up", "the"], we need hidden states at ["Subtask:", "pick", "up"]
        text_token_len = subtask_tokens.shape[1]
        text_embeddings = prefix_out[:, -(text_token_len + 1):]  # (batch, text_len+1, emb)
        
        # Target is the full subtask (no shift needed since we have the extra position)
        target_tokens = subtask_tokens  # (batch, seq_len)
        
        # Decode to vocabulary logits
        # Use all N positions (from before first token to before last token) to predict N tokens
        embedding_table = self.PaliGemma.llm.embedder["input_embedding"]
        logits = jnp.einsum(
            'bse,ve->bsv', 
            text_embeddings[:, :-1],  # Exclude last position, use N positions before each target
            embedding_table,
            optimize='optimal'
        )  # Shape: (batch, seq_len, vocab_size)
        
        # Predict tokens and show for debugging
        predicted_token_ids = jnp.argmax(logits, axis=-1)

        # Compute loss
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens)
        valid_mask = subtask_mask.astype(jnp.float32)  # No shift needed
        masked_loss = per_token_loss * valid_mask
        
        mean_loss = jnp.mean(jnp.sum(masked_loss, axis=-1) / jnp.clip(jnp.sum(valid_mask, axis=-1), 1))
        return mean_loss

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask) # indicating global tokens vs padding
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, "b *s"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, "s"],
        at.Float[at.Array, "b *s emb"] | None, # Allows adarms to be shaped [b, ad, emb], i.e. (batch, 15, 1024) for per-token time embedding.
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        b = noisy_actions.shape[0] if len(noisy_actions.shape) == 3 else 1
        embed_dim = self.action_in_proj.out_features #1024 
        
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, embed_dim, min_period=4e-3, max_period=4.0) # (32, 15, 1024)
        
        # time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb 

       
       # CONTINUE: Check time shape when not running guided inference. Or in normal mode. Seems to have too large shape 
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        logger.log(level=103, msg=f"ADRAMS SHAPE: {adarms_cond.shape}")
        # adarms_cond = adarms_cond.reshape(tokens.shape)
        return tokens, input_mask, ar_mask, adarms_cond
    
    @override
    def compute_loss_ki(
        self, rng:at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train:bool=False
    ) -> at.Float[at.Array, "*b ah"]: 
        
        """
        Computes the combined loss for Knowledge Insulation. 
        Works in two parts. 
        Part 1: Predicts FAST tokens (tokenized action chunk) using the VLM only and computing loss agains ground truht. 
        Part 2: Uses the action expert with attention to the VLM latent tokens to predict continuous actions. 
        Importantly, the KV cache is detached before computing the contiuous actions, this prevents gradients from flowing from the action expert to the VLM.

        The Observation structure: 
        {
            "state": jnp.array (1, 32),
            "tokenized_prompt": jnp.array (1, 200), <-- tokenized text prompt, contains the original prompt + tokenized actions
            "tokenized_prompt_mask": jnp.array (1, 200),
            "token_loss_mask": jnp.array (1, 200)
            "base_0_rgb": jnp.array (1, 224, 224, 3)
            "left_wrist_0_rgb": jnp.array (1, 224, 224, 3)
            "right_wrist_0_rgb": jnp.array (1, 224, 224, 3)
            "base_0_rgb": jnp.array (1,)
            "left_wrist_0_rgb": jnp.array (1,)
            "right_wrist_0_rgb": jnp.array (1,)
        }

        The "actions" input is the original continuous action chunk of shape (1,15,32). 
        This is for part 1 tokenized and compared to the FAST tokens, and in part 2 compared normally. 
        """
        
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2] # Actions typically shape [1, 15, 32]
        noise = jax.random.normal(noise_rng, actions.shape) # <-- See if we can start from a previous action, not totally random
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001 # Sample time from Beta distribution, shape [1]
        time_expanded = time[..., None, None] # Expand to [1, 1, 1] for broadcasting
        x_t = time_expanded * noise + (1 - time_expanded) * actions # Diffuse the actions based on time
        ## x_t = [time, time, time] * [1, 15, 32] + [1-time, 1-time, 1-time] * [1, 15, 32] -> [1, 15, 32]
        ## At t=0, x_t is just the original actions. At t=1, x_t is pure noise. In between, it's a mix.
        u_t = noise - actions # The "velocity" we want to predict: how to denoise x_t back to actions

        #############################################################################
        ######## PART ONE: FAST LOSS - Compute VLM forward and cache results ########
        #############################################################################

        first_subtask_idx = jnp.argmax(observation.action_token_mask, axis=1)
        base_prompt_mask = (
            jnp.arange(self.max_token_len)[None, :] < first_subtask_idx[:, None]
        ) & observation.tokenized_prompt_mask
        base_prompt_tokens = jnp.where(base_prompt_mask, observation.tokenized_prompt, 0)
        
        
        gt_fast_only, gt_fast_mask = _pack_sequence_by_first_true(
            observation.tokenized_prompt,
            observation.action_token_mask,
        )
        
        base_prompt_obs = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=base_prompt_tokens,
            tokenized_prompt_mask=base_prompt_mask,  # Only valid for base
            token_ar_mask=None,
            token_loss_mask=None,
        )

        # Teacher-forced FAST prediction: [images + base prompt] + [GT FAST tokens]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(base_prompt_obs)
        gt_fast_embeddings = self.PaliGemma.llm(gt_fast_only, method="embed")

        full_prefix_tokens = jnp.concatenate([prefix_tokens, gt_fast_embeddings], axis=1)
        full_prefix_mask = jnp.concatenate([prefix_mask, gt_fast_mask], axis=1)
        fast_ar_mask = jnp.ones(gt_fast_embeddings.shape[1], dtype=bool)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, fast_ar_mask])

        prefix_attn_mask = make_attn_mask(full_prefix_mask, full_ar_mask)
        prefix_positions = jnp.cumsum(full_prefix_mask, axis=1) - 1

        (prefix_out_FAST, _), kv_cache = self.PaliGemma.llm(
                [full_prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=prefix_positions,
        )

        # Compute FAST text loss over teacher-forced FAST tokens.
        per_token_fast_loss = optax.softmax_cross_entropy_with_integer_labels(
            jnp.einsum(
                'bse,ve->bsv',
                prefix_out_FAST[:, -(gt_fast_only.shape[1] + 1):-1],
                self.PaliGemma.llm.embedder["input_embedding"],
                optimize='optimal',
            ),
            gt_fast_only,
        )
        masked_fast_loss = per_token_fast_loss * gt_fast_mask.astype(jnp.float32)
        FAST_loss = jnp.mean(
            jnp.sum(masked_fast_loss, axis=-1)
            / jnp.clip(jnp.sum(gt_fast_mask, axis=-1), 1)
        )



        #############################################################################
        ###### PART TWO: ACTION LOSS - Reuse KV cache to avoid recomputing VLM ######
        #############################################################################

        # Stop gradients on KV cache to prevent action loss from updating VLM
        kv_cache_detached = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        # Compute suffix tokens (noisy actions + time embedding)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        ## suffix_mask indicating which tokens are part of the suffix
        ## suffix_ar_mask indicating the autoregressive attention between tokens 

        # Create attention masks for suffix tokens
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        
        prefix_attn_mask_for_suffix = einops.repeat(full_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        ## Converts prefix_mask from shape (b, prefix length) to (b, suffix length, prefix length) for attention between suffix tokens and cached prefix tokens
        
        full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(full_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        # Forward pass with cached KV: only compute action expert
        # Pass [None, suffix_tokens] to skip VLM recomputation
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],  # Skip VLM, use cached KV
            mask=full_attn_mask, 
            positions=suffix_positions,
            kv_cache=kv_cache_detached,  # Reuse stopped-gradient cache
            adarms_cond=[None, adarms_cond],
        ) 
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        continuous_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        # Combined loss: both components contribute to final loss
        total_loss = continuous_loss + self.ki_fast_loss_weight * FAST_loss
        return total_loss
    
    @override
    def compute_loss_hierarchical(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> any : #at.Float[at.Array, "*b ah"]:
        """
        Compute loss for hierarchical policy training with TEACHER FORCING.
        
        Part 1: Predict subtask from decomposition prompt (trains VLM to generate subtasks)
        Part 2: Use action prompt with embedded ground truth subtask for action generation (teacher forcing)
        
        Expected observation fields (provided by data pipeline):
            - tokenized_prompt: Decomposition prompt (e.g., "Decompose task into subtask. Task: {task}, State: {state}")
            - gt_subtask_tokens: Ground truth subtask text (e.g., "Pick up red block")
            - gt_action_prompt_tokens: Action prompt with subtask embedded (e.g., "What should robot do? State: {state}, Task: {subtask}, Action:")
        
        We expect observation to contain a tokenized subtask which is the ground truth subtask that the VLM should produce. 
        The Observation structure: 
        {
            "state": jnp.array (1, 32),
            "base_0_rgb": jnp.array (1, 224, 224, 3)
            "left_wrist_0_rgb": jnp.array (1, 224, 224, 3)
            "right_wrist_0_rgb": jnp.array (1, 224, 224, 3)
            "base_0_rgb": jnp.array (1,)
            "left_wrist_0_rgb": jnp.array (1,)
            "right_wrist_0_rgb": jnp.array (1,)

            "tokenized_prompt": jnp.array (1, 200), <-- tokenized decomposition prompt to make subtask, i.e. "Task:{prompt}, State:{state}, Subtask:"
            "tokenized_prompt_mask": jnp.array (1, 200),
            
            "subtask_tokens": <-- Tokenized prompt to make actions, i.e. "Task: {subtask}, State:{state}, Action:" 
            "subtask_mask": 
            "subtask_gt_tokens": <-- Tokenized the ground truth subtask text, i.e. "Move forward", used in loss function
            "subtask_gt_mask":
        }
        """

        # Prepare noisy actions for diffusion
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        #################################################################
        ######## PART ONE: Subtask Prediction Loss (VLM Training) #######
        #################################################################
        
        # Teacher forcing: LLM predicts GT subtask tokens given images + base prompt
        # tokenized_prompt format: "Task: xxx; State: xxx; Subtask: xxx;\nAction: xxx;"
        # Masks indicate: subtask_token_mask marks subtask tokens, action_token_mask marks action tokens
        # We train LLM to do next-token prediction on subtask given:
        #   [images] + [base prompt "Task: xxx; State: xxx; Subtask: "] + [GT subtask tokens]
        
        # Step 1: Extract base prompt as strict prefix before the subtask span.
        # This guarantees trailing "Action: " tokens are excluded.
        first_subtask_idx = jnp.argmax(observation.subtask_token_mask, axis=1)
        base_prompt_mask = (
            jnp.arange(self.max_token_len)[None, :] < first_subtask_idx[:, None]
        ) & observation.tokenized_prompt_mask
        base_prompt_tokens = jnp.where(base_prompt_mask, observation.tokenized_prompt, 0)
        
        
        # Step 2: Extract GT subtask tokens and left-pack them for aligned teacher forcing.
        # The mask from tokenization is a contiguous span, so rolling by first True index
        # yields [subtask_tokens..., padding...].
        gt_subtask_only, gt_subtask_mask = _pack_sequence_by_first_true(
            observation.tokenized_prompt,
            observation.subtask_token_mask,
        )
        
        # Step 3: Create observation with only base prompt for embed_prefix
        base_prompt_obs = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=base_prompt_tokens,
            tokenized_prompt_mask=base_prompt_mask,  # Only valid for base
            token_ar_mask=None,
            token_loss_mask=None,
        )
        
        # Step 4: Embed images + base prompt using embed_prefix
        # This returns:
        #   - prefix_tokens: [img_embeds, base_prompt_embeds] with shape [b, img_seq_len + base_len, emb_dim]
        #   - prefix_mask: validity mask [b, img_seq_len + base_len]
        #   - prefix_ar_mask: ar_mask with False for all (bidirectional prefix attention)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(base_prompt_obs)
        
        # Step 5: Embed GT subtask tokens separately for next-token prediction
        gt_subtask_embeddings = self.PaliGemma.llm(gt_subtask_only, method="embed")
        
        # Step 6: Use packed subtask mask from step 2.
        
        # Step 7: Concatenate: [images + base_prompt] + [gt_subtask]
        full_prefix_tokens = jnp.concatenate([prefix_tokens, gt_subtask_embeddings], axis=1)
        full_prefix_mask = jnp.concatenate([prefix_mask, gt_subtask_mask], axis=1)
        
        # Step 8: Build AR mask for next-token prediction:
        #   - Images and base prompt: False (bidirectional)
        #   - GT subtask: True (causal - each token can only attend to prior tokens)
        subtask_ar_mask = jnp.ones(gt_subtask_embeddings.shape[1], dtype=bool)
        full_ar_mask = jnp.concatenate([prefix_ar_mask, subtask_ar_mask])
        
        prefix_attn_mask = make_attn_mask(full_prefix_mask, full_ar_mask)
        prefix_positions = jnp.cumsum(full_prefix_mask, axis=1) - 1
        
        (prefix_out_pred, _), _ = self.PaliGemma.llm(
            [full_prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )
        
        # Compute loss on all subtask tokens
        subtask_loss = self._compute_subtask_loss(prefix_out_pred, gt_subtask_only, gt_subtask_mask)

        #########################################################################################
        ######## PART TWO: Action Loss with Ground Truth Action Prompt (Teacher Forcing) ########
        #########################################################################################
        
        # Create observation with action prompt that has ground truth subtask embedded
        # Data pipeline provides this pre-formatted
        teacher_forced_obs = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=observation.subtask_tokens,  # Action prompt with subtask
            tokenized_prompt_mask=observation.subtask_gt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=None,
        )
        
        # Process VLM with ground truth action prompt
        gt_prefix_tokens, gt_prefix_mask, gt_prefix_ar_mask = self.embed_prefix(teacher_forced_obs)
        gt_prefix_attn_mask = make_attn_mask(gt_prefix_mask, gt_prefix_ar_mask)
        gt_prefix_positions = jnp.cumsum(gt_prefix_mask, axis=1) - 1
        
        (_, _), kv_cache_gt = self.PaliGemma.llm(
            [gt_prefix_tokens, None],
            mask=gt_prefix_attn_mask,
            positions=gt_prefix_positions,
        )

        kv_cache_detached = jax.tree.map(jax.lax.stop_gradient, kv_cache_gt)
        
        # Compute suffix tokens (noisy actions)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            teacher_forced_obs, x_t, time
        )
        
        # Create attention masks
        """
        Explaining the attention masks: 
            - Suffix_attn_mask is created from the suffix_mask and suffix_ar_mask 
            |- suffix_mask indicates which tokens in the suffix are actual tokens and not padding
            |- suffix_ar_mask indicates the autoregressive attention between all the suffix tokens.
                |- the ar-mask is structured as a boolean array, where True indicates that previous tokens cannot attend to i
                    and False indicates the token can attend to the same tokens as the previous token. 
            Now the combined suffix_attn_mask indicates which suffix tokens can attend to which other *suffix* tokens. 

            We further want to make a attention mask indicating how the suffix tokens can attend to the prefix tokens. 
            We allow the suffix to attend to all prefix tokens, so we repeat the prefix_mask for each suffix.
            This creates the *prefix_attn_mask_for_suffix*. 
            We then concatenate these to create the *full_attn_mask*        
        """
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_for_suffix = einops.repeat(
            gt_prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(gt_prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        
        # Forward pass with ground truth subtask context
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache_detached,  # Cache from GROUND TRUTH subtask
            adarms_cond=[None, adarms_cond],
        )
        
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        total_loss = action_loss + self.ki_subtask_loss_weight * subtask_loss

        return total_loss

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:

        
        #### HACK - Lets us use the compute loss function call always depending on the model type 
        if self.hi_mode and not self.ki_mode:
            # If hi_mode enabled, we expect the prompt in obs to look like "Task: {task}; State: {state}; Subtask: {subtask};\n Action: " with masks indicating tokens that are the subtask tokens.   
            return self.compute_loss_hierarchical(rng, observation, actions, train=train)
        elif self.ki_mode and not self.hi_mode:
            # If ki_mode enabled, we expect the prompt in obs to look like "Task: {task}; State: {state};\n Action: {FAST tokens}" with masks indicating which tokens are the FAST tokens.  
            return self.compute_loss_ki(rng, observation, actions, train=train)

        elif self.ki_mode and self.hi_mode: 
            # If both hi and ki enabled, we expect the prompt in obs to look like "Task: {task}; State: {state}; Subtask: {subtask};\n Action: {FAST tokens}" with masks indicating which tokens are the subtask and which are the FAST tokens.
            # We here compute loss over both subtask and FAST tokens, and the tokens in both cases are produced autoregressively. 
            pass 
        
        elif self.config.guided_inference: 
            batch_shape = actions.shape[:-2]
            preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
            delay = jax.random.randint(delay_rng, batch_shape, 0, self.max_delay)
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001 # Returns one value for time.
            u_t = noise - actions

            prefix_action_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]
            time_expanded = time[..., None, None]
            
            time_delay = jnp.where(prefix_action_mask, 0.0, time[:, None])  # Creates array where the first "delay" actions have time=0 (no noise) and the rest have the sampled time value. 
            
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time_delay)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            loss = jnp.square(v_t - u_t)
            postfix_action_mask = ~prefix_action_mask[:, :, None]
            loss = jnp.sum(loss*postfix_action_mask, axis=-1) / (jnp.sum(postfix_action_mask, axis=-1) + 1e-8) # computing the mean over only the action postfix tokens, i.e. the predicted tokens
            return loss 



        else: 
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
            """
            Explaination of the following code: 
            First they generate noise in the shape of the action. 
            Then they add noise to that action acording to the sampled time, so x_t is a noised action at some time between 1.0 and 0.0. 1 is noise. 
            Then u_t is produced, which is the desired velocity to predict.  
            """
            batch_shape = actions.shape[:-2]
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
            time_expanded = time[..., None, None]
            time = jnp.repeat(time[:, None], self.action_horizon, axis=1)#.reshape(1,-1)
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            """
            Explaination of the following code:
            They embed the prefix (images + tokenized prompt + state) and the suffix (noisy actions + time).
            They then create attention masks, indicating which tokens are tokens and not padding, autoregressive mask indicating which tokens can attend to other tokens (previous ones), and then combine these to create the full attention mask.
            Positions are the indicies of the actual tokens. So if the input_mask is [1, 1, 1, 0, 0] the positions become [1, 2, 3, 3, 3]-1 = [0, 1, 2, 2, 2]. So the tokens have index 0, 1, 2. We here do assume right padding.
            We call on the Paligemma LLM using both the backbone (prefix) and the action expert (suffix) to generate suffix_out, which we further use to generate our velocity field v_t trough a linear projection. 
            This v_t is the prediction we want to match u_t. 
            """
            
            
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return jnp.mean(jnp.square(v_t - u_t), axis=-1)
        


    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:

        num_steps = 10

        observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps        
        batch_size = observation.state.shape[0]

        if noise is None:
            # noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            
            # noise = jnp.zeros((batch_size, self.action_horizon, self.action_dim))
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))*1
            # noise = noise.at[:, :3, 6].set(0.0)

            # pattern = jnp.where(
            #     jnp.arange(self.action_horizon) % 2 == 0,
            #     -1.0,
            #     1.0
            # )

            # alternating_arr = jnp.broadcast_to(
            #     pattern[None, :, None],
            #     (batch_size, self.action_horizon, self.action_dim)
            # )
            # noise += alternating_arr
            

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry, _):
            x_t, time = carry
            

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )

            assert prefix_out is None

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            # v_t = v_t.at[:, :3, 6].set(0.0)

            x_next = x_t + dt * v_t
            t_next = time + dt

            # values to record
            record = (x_t, v_t, time)

            return (x_next, t_next), record

        init_carry = (noise, 1.0)

        (x_final, _), (x_hist, v_hist, t_hist) = jax.lax.scan(
            step,
            init_carry,
            xs=None,
            length=num_steps,
        )
        
        hist = {
            "x_hist": x_hist,
            "x_final": x_final,
            "v_hist": v_hist,
            "t_hist": t_hist,
        }

        return x_final, hist

    @override
    # def sample_actions(
    #     self,
    #     rng: at.KeyArrayLike,
    #     observation: _model.Observation,
    #     *,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    #     noise: at.Float[at.Array, "b ah ad"] | None = None,
    #     d: int | at.Int[at.Array, ""] = 4,
    #     s: int | at.Int[at.Array, ""] = 5,
    #     A_prev: _model.Actions | None = None,
    # ) -> _model.Actions:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    #     # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]
    #     if noise is None:
    #         noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    #     # first fill KV cache with a forward pass of the prefix
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    #     def step(carry):
    #         x_t, time = carry
    #         suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time, batch_size)
    #         )
    #         # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
    #         # other
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
    #         # prefix tokens
    #         prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
    #         # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    #         assert full_attn_mask.shape == (
    #             batch_size,
    #             suffix_tokens.shape[1],
    #             prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #         )
    #         # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    #         (prefix_out, suffix_out), _ = self.PaliGemma.llm(
    #             [None, suffix_tokens],
    #             mask=full_attn_mask,
    #             positions=positions,
    #             kv_cache=kv_cache,
    #             adarms_cond=[None, adarms_cond],
    #         )
    #         assert prefix_out is None
    #         v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    #         return x_t + dt * v_t, time + dt

    #     def cond(carry):
    #         x_t, time = carry
    #         # robust to floating-point error
    #         return time >= -dt / 2

    #     x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    #     return x_0
    



    @override
    def guided_inference(
            self, 
            rng: at.KeyArrayLike, 
            observation: _model.Observation,
            *, 
            num_steps: int | at.Int[at.Array, ""] = 10, 
            noise: at.Float[at.Array, "b ah ad"] | None = None,
            d: int | at.Int[at.Array, ""] = 4,
            s: int | at.Int[at.Array, ""] = 5,
            j: int | at.Int[at.Array, ""] = 0,
            A_prev: at.Float[ArrayT, ""] | None = None,

    ): 
        """
        Implementation of real-time chunking from physical intelligence paper. 
        
        ## Args: 
        - d : number of actions that are executed during the time it takes to run inference 
        - s : number of actions that are executed before new inference run starts
        - j : number of actions from the previous action chunk that are pinned when predicting the new chunk 
        - A_prev : the previous action chunk sliced by [:, s:, :] 
        """
        
        observation = _model.preprocess_observation(None, observation, train=False)
        
        batch_size = observation.state.shape[0]
        dt = -1.0 / num_steps
        H = self.action_horizon 
        i = jnp.arange(H) # Time steps within the action horizon

        ## Building soft-maxing weights ###
        ci = jnp.where(
            jnp.logical_and(i >= d, i < H-s), 
            (H-s-i) / (H-s-d+1), 
            0.0
        ) #Should be as long as the condition d<= i < H-s over the index array i 
        
        W = jnp.where(
            i < d, 1.0,                                                 # 1 for i < d 
            jnp.where(
                jnp.logical_and((i >= d), (i < H-s)), 
                ci*((jnp.exp(ci)-1)/(jnp.exp(jnp.ones_like(ci))-1)),    # Expression for i in [d, H-s]
                0.0                                                     # 0 for i >= H - s
            )
        )
        ##################################

        ### Handle A_prev ###
        if A_prev is None:
            A_prev = jnp.zeros((batch_size, self.action_horizon, self.action_dim))
            W = jnp.zeros_like(W) # If no previous actions, set weights to zero to disable guidance

        else: 
            A_prev = jnp.asarray(A_prev)

        pad_width = max(0, H - A_prev.shape[1]) # Calculate how much padding is needed if A_prev has fewer than H steps
        A_prev = jnp.pad(A_prev, ((0,0), (0,pad_width), (0, 0))) # Right-pad A_prev with zeros to be H-length 
        #####################

        if noise is None:
            t = jnp.arange(self.action_horizon) 
            # noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
            
            # noise = jnp.zeros((batch_size, self.action_horizon, self.action_dim))

            ## SCALED RANDOM NORMAL NOISE ##

            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))*1
            # noise = _noise_around_action(A_prev, rng)
            
            ## PINNING NOISE TO ZERO ##
            # noise = noise.at[:, :, 6].set(
            #     jnp.where(
            #         (t<s),          # mask 
            #         0.0,            # If true  
            #         noise[:, :, 6]  # If fale 
            #     )
            # )

            ## PINNING NOISE TO PREV ACTION BEGINNING ## 
            pin_mask = (t < j)[None, :, None] # Shape (1, action_horizon, 1) with True for indices < j
            # noise = noise.at[:, :, :].set(
            #     jnp.where(
            #         pin_mask,                  # mask
            #         A_prev[:, :, :],        # if true
            #         noise[:, :, :]          # if false
            #         )
            #     ) 

            noise = jnp.where(pin_mask, A_prev, noise) 
        
            
            ## SPECIFIC NOISE PATTERN ## 
            # pattern = jnp.where(
            #     jnp.arange(self.action_horizon) % 2 == 0,
            #     -1.0,
            #     1.0
            # )

            # alternating_arr = jnp.broadcast_to(
            #     pattern[None, :, None],
            #     (batch_size, self.action_horizon, self.action_dim)
            # )
            # noise += alternating_arr



        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry, _):
            x_t, time = carry 

            def denoiser(x_t): 
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, jnp.broadcast_to(time, batch_size)
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )

                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                x_1 = x_t - time*v_t
                
                return x_1, v_t

            x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
            e = (A_prev - x_1) * W[:, None] # Could be diag depending on shapes 

            r2 = (time**2) / ((1-time)**2 + time**2)

            pinv_correction = vjp_fun(e)[0]
            c = jnp.nan_to_num((time)/(1-time), posinf=5.0) ## A max weight tolerance, should be input
            guidance_weight = jnp.minimum(c / r2, 5.0) # Guidance weight with a max cap to prevent extreme values
            v_corrected = v_t - guidance_weight * pinv_correction

            
            # Pinning each prediction to the previous action chunk ## 
            t = jnp.arange(self.action_horizon)
            pin_mask = (t < j)[None, :, None] # Shape (1, action_horizon, 1) with True for indices < j
            v_corrected = jnp.where(pin_mask, 0.0, v_corrected)
            x_t = x_t + dt * v_corrected
            x_t = jnp.where(pin_mask, A_prev, x_t)

            time += dt

            # values to record
            record = (x_t, v_t, time)

            return (x_t, time), record 
        
        def cond(carry):
            _, time, _ = carry
            return time >= -dt / 2
        
        init_carry = (noise, 1.0) 
        (x_1, _), (x_hist, v_hist, t_hist) = jax.lax.scan( 
            f=step, 
            init=init_carry,
            xs=None, 
            length=num_steps
        )

        hist = {
            "x_final": x_1,
            "x_hist": x_hist,
            "v_hist": v_hist,
            "t_hist": t_hist,
        }


        return x_1, hist
        


    @at.typecheck
    def _generate_subtask(
        self, 
        rng: at.KeyArrayLike,
        observation: _model.Observation, 
        original_prompt: str | None = None, 
        max_tokens: int = 200,
        eos_token_id: int = 1, 
        temperature: float = 0.0,
        debug_top_k: int = 5,
        ) -> tuple[at.Int[at.Array, "b s"], at.Bool[at.Array, "b s"]]: 
        """Generate subtask tokens using VLM autoregressive generation with KV caching.
        
        NOTE: The observation should contain the decomposition prompt already tokenized.
        The decomposition prompt should be formatted like:
        
        This method generates token IDs that should be decoded externally using a tokenizer.
        
        Args:
            rng: Random key for sampling
            observation: Observation with images and DECOMPOSITION prompt tokenized
            original_prompt: Original high-level task (for logging only)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            eos_token_id: Token ID for end-of-sequence
            debug_top_k: Number of top candidates to log per decode step (set <=0 to disable)
        Returns:
            List of generated token IDs (not including prompt tokens)
            List of boolean values indicating whether each generated token is a subtask token
        """


        batch_size = observation.tokenized_prompt.shape[0]
        
        
        # Get initial prefix embeddings (images + DECOMPOSITION prompt from observation)
        prefix_tokens_embedding, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        
        # prefix_tokens_embedding, prefix_mask, prefix_attn_mask = left_to_right_align(prefix_tokens_embedding, prefix_mask, prefix_attn_mask)

        prefill_size = prefix_tokens_embedding.shape[1] # How many tokens in the prefix, should be shape (batch_size, 200)
        prefix_len = jnp.sum(prefix_mask, axis=-1) # How many tokens in the prefix (i.e not 0), should be shape (batch_size,)
        text_len = jnp.sum(observation.tokenized_prompt_mask, axis=-1).astype(jnp.int32)
        # Exclude image tokens from decode attention; allow only task text + generated tokens.
        text_start = prefix_len.astype(jnp.int32) - text_len

        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Initial forward pass to get KV cache
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens_embedding, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )
        #kv_cache = jax.tree.map(lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, 1), (0, 0), (0, 0))), kv_cache)
        last_idx = jnp.clip(prefix_len.astype(jnp.int32) - 1, 0)
        gather_idx = last_idx[:, None, None]
        last_embedding = jnp.take_along_axis(prefix_out, gather_idx, axis=1)
        last_logits = self.PaliGemma.llm(last_embedding, method="decode")  
        last_logits = jax.nn.log_softmax(last_logits, axis=-1)  
        output_tokens = jnp.zeros((batch_size, max_tokens), dtype=jnp.int32)

        
        def fixed_prefix_mask(start_idx, end_idx, length:int, dtype=jnp.bool_) -> jnp.ndarray:
            idx = jnp.arange(length)[None, None, :] 
            return jnp.logical_and(
                idx >= start_idx[:, None, None],
                idx < end_idx[:, None, None]
            ) # Creates array of True up to idx and False after

        for step in range(max_tokens) :
            rng, rng_step = jax.random.split(rng)

            if debug_top_k > 0:
                k = min(debug_top_k, int(last_logits.shape[-1]))
                top_log_probs, top_token_ids = jax.lax.top_k(last_logits[:, 0, :], k)
                top_probs = jnp.exp(top_log_probs)
            

            token = jax.lax.cond(
                temperature > 0,
                lambda _: jax.random.categorical(rng_step, last_logits/ temperature, axis=-1),
                lambda _: jnp.argmax(last_logits, axis=-1),
                operand=None,
            ) # (batch_size, 1)

            output_tokens = jnp.put_along_axis(output_tokens, jnp.broadcast_to(step, (token.shape[0], 1)), token, axis=-1, inplace=False)
            eos = jnp.all(jnp.any(token == eos_token_id, axis=-1))

            token_embedding = self.PaliGemma.llm(token, method="embed")  # (batch_size, 1, emb_dim)
            positions = (prefix_len[:, ] + step + 1).reshape(batch_size, 1)  # (batch_size, 1)

            mask_length = kv_cache[0].shape[2] + 1 # i.e. 968, 969, 970 ...             
            end_idx = prefix_len + step + 1
            mask = fixed_prefix_mask(jnp.zeros_like(end_idx), end_idx, mask_length) # (batch_size, 1, mask_length)
            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [token_embedding, None],
                mask=mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, None],
            )

            last_embedding = prefix_out[:, -1:]
            last_logits = self.PaliGemma.llm(last_embedding, method="decode")
            last_logits = jax.nn.log_softmax(last_logits, axis=-1)

            if bool(eos): 
                
                break


        # Return token-level mask expected by policy (shape: batch, max_tokens).
        eos_hits = output_tokens == eos_token_id
        eos_cumsum = jnp.cumsum(eos_hits.astype(jnp.int32), axis=1)
        generated_token_mask = (output_tokens != 0) & (eos_cumsum == 0)

        return output_tokens, generated_token_mask



        # generated_tokens = []
        
        # # Current sequence length (for position tracking)
        # current_length = prefix_tokens_embedding.shape[1]
        
        # # Autoregressive generation loop with KV caching
        # for step in range(max_tokens):
        #     # Get logits for the LAST position (which predicts the next token)
        #     last_embedding = prefix_out[:, -1:, :]  # Shape: (batch, 1, emb_dim)
        #     logits = jnp.einsum('bte,ve->btv', last_embedding, embedding_table)  # (batch, 1, vocab_size)
            
        #     if temperature > 0:
        #         rng, sample_rng = jax.random.split(rng)
        #         next_token_id = jax.random.categorical(
        #             sample_rng, logits[0, 0] / temperature
        #         )
        #     else:
        #         # Greedy decoding
        #         next_token_id = jnp.argmax(logits[0, 0])
            
        #     next_token_int = int(next_token_id)
            
        #     # Check for EOS token (1) or newline (108) to stop generation early
        #     if next_token_int == 1 or next_token_int == 108:
        #         if next_token_int == 1:
        #             logger.log(level=103, msg=f"[HI-Robot] EOS token reached at step {step}")
        #         else:
        #             logger.log(level=103, msg=f"[HI-Robot] Newline token reached at step {step}")
        #         break

        #     generated_tokens.append(next_token_int)
            
        #     # Embed the new token
        #     next_token_embedding = self.PaliGemma.llm(
        #         jnp.array([[next_token_int]]), method="embed"
        #     )  # Shape: (1, 1, emb_dim)
            
        #     # Update position for the new token
        #     next_position = jnp.array([[current_length]], dtype=jnp.int32)
        #     current_length += 1
            
        #     # Create attention mask for new token (can attend to all previous tokens)
        #     # New token attends to: all prefix tokens + all previously generated tokens + itself
        #     new_attn_mask = jnp.ones((1, 1, current_length), dtype=jnp.bool_)
            
        #     # Forward pass for ONLY the new token, reusing KV cache
        #     (new_out, _), kv_cache = self.PaliGemma.llm(
        #         [next_token_embedding, None],
        #         mask=new_attn_mask,
        #         positions=next_position,
        #         kv_cache=kv_cache,  # Reuse cached keys/values from previous tokens
        #     )
            
        #     # Update prefix_out for next iteration
        #     prefix_out = new_out
        
        # # logger.log(level=103, msg=f"[HI-Robot] Generated {len(generated_tokens)} tokens: {generated_tokens}")
        # generated_tokens_mask = [True] * len(generated_tokens) + [False] * (max_tokens - len(generated_tokens))
        # generated_tokens = generated_tokens + [0] * (max_tokens - len(generated_tokens))  # Pad to max_tokens with zeros
        # return generated_tokens, generated_tokens_mask

