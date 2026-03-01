import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi import transforms as _transforms




logger = logging.getLogger("openpi")


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


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.config = config
        self.model_type = config.model_type
        self.knowledge_insulation = config.knowledge_insulation
        self.ki_fast_loss_weight = config.ki_fast_loss_weight
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
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
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def compute_fast_loss(
        self, 
        prefix_out: at.Float[at.Array, "b s emb"], 
        observation: _model.Observation
    ) -> at.Float[at.Array, ""]:
        """Compute FAST token prediction loss for Knowledge Insulation.
        
        Memory-optimized version that avoids creating large one-hot tensors.
        
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
        
        # Use optax's efficient cross-entropy with integer labels
        # This avoids creating massive one-hot tensors (batch × seq_len × 256K vocab)
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, target_tokens
        )  # Shape: (batch, seq_len-1)
        
        # Apply loss mask: only compute loss on FAST tokens
        masked_loss = per_token_loss * loss_mask
        
        # Normalize by number of masked tokens per sample
        sum_masked_loss = jnp.sum(masked_loss, axis=-1)  # Sum over sequence
        num_tokens = jnp.clip(jnp.sum(loss_mask, axis=-1), 1)  # Avoid division by zero
        batch_loss = sum_masked_loss / num_tokens  # Normalize per sample
        
        # Average over batch
        return jnp.mean(batch_loss)

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
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:# WE DO NOT USE PI0 SO WE DO NOT EVER RUN THIS BLOCK
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        if self.knowledge_insulation == False:
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
        
        elif self.knowledge_insulation == True:
            # Knowledge Insulation: Memory-optimized implementation with KV cache reuse
            # Two losses computed from two forward passes, but VLM computation reused via cache
            
            logger.log(level=103, msg="[DEBUG] Running KI loss-computation (memory-optimized)")
            
            #############################################################################
            ######## PART ONE: FAST LOSS - Compute VLM forward and cache results ########
            #############################################################################

            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            # prefix attention mask. Separates the image+prompt+state tokens into a bi-directional block 
            ## and the FAST tokens into an autoregressive block.
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask) 
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1 
            
            # Forward pass through VLM: get outputs AND KV cache
            # Gradients will flow through prefix_tokens for FAST loss
            (prefix_out_FAST, _), kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None],  # Only VLM expert
                mask=prefix_attn_mask,
                positions=prefix_positions,
            ) 
            
            # Compute FAST loss (gradients flow to VLM parameters)
            FAST_loss = self.compute_fast_loss(prefix_out_FAST, observation) 

            logger.log(level=103, msg=f"[DEBUG] FAST loss computed: {FAST_loss:.4f}")

            #############################################################################
            ###### PART TWO: ACTION LOSS - Reuse KV cache to avoid recomputing VLM ######
            #############################################################################

            # Stop gradients on KV cache to prevent action loss from updating VLM
            kv_cache_detached = jax.tree.map(jax.lax.stop_gradient, kv_cache)
            
            # Compute suffix tokens (noisy actions + time embedding)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
            
            # Create attention masks for suffix tokens
            # suffix_attn_mask: how suffix tokens attend to each other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # prefix_attn_mask: how suffix tokens can attend to cached prefix
            prefix_attn_mask_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            # Combined mask: suffix can attend to [cached_prefix, suffix]
            full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
            
            # Positions for suffix tokens (continue from where prefix ended)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            
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
            logger.log(level=103, msg=f"[DEBUG] Continuous action loss computed: {jnp.mean(continuous_loss):.4f}")
            # Combined loss: both components contribute to final loss
            total_loss = continuous_loss + self.ki_fast_loss_weight * FAST_loss
            return total_loss


    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    @at.typecheck
    def generate_subtask(
        self, 
        rng: at.KeyArrayLike,
        observation: _model.Observation, 
        original_prompt: str, 
        max_tokens: int = 20,
        temperature: float = 0.7,
    ) -> str: 
        """Generate subtask using VLM autoregressive text generation.
        
        This method passes the observation through the VLM to decompose the 
        high-level task into a more specific sub-task. It uses greedy decoding
        to generate tokens autoregressively.
        
        Args:
            rng: Random key for sampling (if using temperature > 0)
            observation: Preprocessed observation with images and tokenized prompt
            original_prompt: The original high-level goal/prompt text
            max_tokens: Maximum number of tokens to generate for the subtask
            temperature: Sampling temperature (0 = greedy, >0 = stochastic)
            
        Returns:
            Generated subtask text (token IDs as they need external decoding)
        """
        
        logger.log(level=103, msg=f"[HI-Robot] Generating subtask for prompt: {original_prompt}")
        
        # Get initial prefix embeddings (images + decomposition prompt)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # Extract embedding table from VLM for decoding logits to tokens
        embedding_table = self.PaliGemma.llm.embedder["input_embedding"]
        generated_token_ids = []
        
        # Autoregressive generation loop - iteratively predict next token
        for step in range(max_tokens):
            # Create attention mask for current sequence
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            
            # Forward pass through VLM (only vision-language expert, no action expert)
            (prefix_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, None],  # None for action expert
                mask=prefix_attn_mask,
                positions=positions,
            )
            
            # Decode last hidden state to vocabulary logits
            last_embedding = prefix_out[:, -1:, :]  # Shape: (batch, 1, emb_dim)
            logits = jnp.einsum('bte,ve->btv', last_embedding, embedding_table)  # (batch, 1, vocab_size)
            
            # Sample or greedily select next token
            if temperature > 0:
                rng, sample_rng = jax.random.split(rng)
                next_token_id = jax.random.categorical(
                    sample_rng, logits[0, 0] / temperature
                )
            else:
                # Greedy decoding
                next_token_id = jnp.argmax(logits[0, 0])
            
            next_token_id = int(next_token_id)
            
            # Check for EOS token (1) or newline (108) to stop generation early
            if next_token_id == 1:
                logger.log(level=103, msg=f"[HI-Robot] EOS token reached at step {step}")
                break
                
            generated_token_ids.append(next_token_id)
            logger.log(level=103, msg=f"[HI-Robot] Step {step}: Generated token ID {next_token_id}")
            
            # Embed the new token and append to sequence for next iteration
            next_token_embedding = self.PaliGemma.llm(
                jnp.array([[next_token_id]]), method="embed"
            )  # Shape: (1, 1, emb_dim)
            
            # Extend the sequence
            prefix_tokens = jnp.concatenate([prefix_tokens, next_token_embedding], axis=1)
            prefix_mask = jnp.concatenate([prefix_mask, jnp.ones((1, 1), dtype=jnp.bool_)], axis=1)
            prefix_ar_mask = jnp.concatenate([prefix_ar_mask, jnp.array([False])])  # Causal for generated
        
        logger.log(level=103, msg=f"[HI-Robot] Generated token IDs: {generated_token_ids}")
        
        # Return token IDs - decoding happens externally in policy layer with its tokenizer
        return generated_token_ids

