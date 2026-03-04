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


class Pi05(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = True
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.config = config
        self.model_type = config.model_type
        self.knowledge_insulation = config.knowledge_insulation
        self.ki_fast_loss_weight = config.ki_fast_loss_weight
        self.hierarchical_mode = config.hierarchical_mode
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
        
        jax.debug.print("\n🔍 FAST LOSS COMPUTATION:")
        jax.debug.print("  Total prefix_out length: {L}", L=prefix_out.shape[1])
        jax.debug.print("  Tokenized prompt length: {T}", T=text_token_len)
        jax.debug.print("  Extracting last {T} positions for text", T=text_token_len)
        jax.debug.print("  Text tokens (first 10): {t}", t=observation.tokenized_prompt[0][:10])
        jax.debug.print("  Loss mask (first 10): {m}", m=observation.token_loss_mask[0][:10])
        jax.debug.print("  Num tokens with loss mask=True: {n}", n=jnp.sum(observation.token_loss_mask[0]))
        
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
        jax.debug.print("  First 5 FAST token positions: {idx}", idx=mask_indices)
        jax.debug.print("  PREDICTED FAST tokens: {p}", p=predicted_tokens[0, mask_indices])
        jax.debug.print("  GROUND TRUTH tokens:   {t}", t=target_tokens[0, mask_indices])
        
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
        
        # For clarity: show what the model is learning
        jax.debug.print("\n📝 SUBTASK PREDICTION:")
        jax.debug.print("  FULL SUBTASK (target): {x}", x=target_tokens[0][:10])  # What we want to generate
        jax.debug.print("  PREDICTED (output):    {x}\n", x=predicted_token_ids[0][:10])  # What model predicts
        
        # Compute loss
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_tokens)
        valid_mask = subtask_mask.astype(jnp.float32)  # No shift needed
        jax.debug.print("valid tokens per ex: {x}", x=jnp.sum(valid_mask, axis=-1))
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
        
        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
       
        # time MLP (for adaRMS)
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
       
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
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
        
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        ## Combines the prefix mask (true for tokens that are part of the input, false else.)
        ### and the prefix autoregressive mask (indicates which tokens can attent to which.)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1 # Position indices for prefix tokens, used for relative positional embeddings

        # Forward pass through VLM 
        (prefix_out_FAST, _), kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None],  # Only VLM expert
                mask=prefix_attn_mask,
                positions=prefix_positions,
        )   

        # Compute FAST loss based on VLM output and observation tokens
        FAST_loss = self._compute_fast_loss(prefix_out_FAST, observation) 

        jax.debug.print("\n📊 KNOWLEDGE INSULATION - PART 1 (FAST LOSS):")
        jax.debug.print("  prefix_out shape: {s}", s=prefix_out_FAST.shape)
        jax.debug.print("  FAST loss: {loss}", loss=FAST_loss)

        logger.log(level=103, msg=f"[DEBUG] FAST loss: {FAST_loss}")

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
        
        prefix_attn_mask_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        ## Converts prefix_mask from shape (b, prefix length) to (b, suffix length, prefix length) for attention between suffix tokens and cached prefix tokens
        
        full_attn_mask = jnp.concatenate([prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1)
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
        
        # Teacher forcing: concatenate GT subtask to decomposition prompt
        # Input becomes: [images, "Task: X, State: Y, Subtask:", "Pick", "up", "the", "cube"]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # Embed GT subtask tokens and concatenate
        if observation.subtask_gt_tokens is not None:
            logger.log(level=103, msg="[DEBUG] Embedding ground truth subtask tokens for teacher forcing.")

            subtask_embeddings = self.PaliGemma.llm(observation.subtask_gt_tokens, method="embed")
            
            # All tokens for generating subtask. This is the prefix + task decomposition prompt + embedded subtask
            prefix_tokens = jnp.concatenate([prefix_tokens, subtask_embeddings], axis=1)

            # Indicating what tokens are actual tokens and which are padding. 
            prefix_mask = jnp.concatenate([prefix_mask, observation.subtask_gt_mask], axis=1)

            # Subtask tokens use causal attention (can't see future tokens)
            # Set to True to enable causal attention within subtask
            subtask_ar_mask = jnp.array([True] * subtask_embeddings.shape[1])
            prefix_ar_mask = jnp.concatenate([prefix_ar_mask, subtask_ar_mask])
        
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        (prefix_out_pred, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        jax.debug.print("\n🔍 PREDICTED SUBTASK TOKENS: {x}", x=prefix_out_pred.shape)
        
        # Compute loss on all subtask tokens
        subtask_loss = self._compute_subtask_loss(prefix_out_pred, observation.subtask_gt_tokens, observation.subtask_gt_mask)
        logger.log(level=103, msg=f"[DEBUG] Subtask prediction loss: {subtask_loss}")

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
        logger.log(level=103, msg=f"[DEBUG] Action loss: {jnp.mean(action_loss):.4f}")

        total_loss = action_loss + self.ki_subtask_loss_weight * subtask_loss

        return total_loss


    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        
        #### HACK - Lets us use the compute loss function call always depending on the model type 
        if self.hierarchical_mode: 
            logger.log(level=103, msg="[DEBUG] Computing hierarchical loss with teacher forcing.")
            return self.compute_loss_hierarchical(rng, observation, actions, train=train)
        elif self.knowledge_insulation:
            logger.log(level=103, msg="[DEBUG] Computing loss with knowledge insulation.")
            return self.compute_loss_ki(rng, observation, actions, train=train)
        

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

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

    # [TODO] Current implementation runs an error due to the tokenizer being instansiated twice, once in data transforms pipeline and a second time in the Pi0 model instance. 
    # JAX only accepts one instance so either find a workaround using the data pipeline tokenizer or something else to be able to run HI-Robot
    # This todo is only affecting the HI-Robot part of the pipeline. Maybe not even crucial for training. 

    @at.typecheck
    def _generate_subtask(
        self, 
        rng: at.KeyArrayLike,
        observation: _model.Observation, 
        original_prompt: str | None = None, 
        max_tokens: int = 20,
        temperature: float = 0.7,
        ) -> list[int]: 
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
            
        Returns:
            List of generated token IDs (not including prompt tokens)
        """
        if original_prompt is not None:
            logger.log(level=103, msg=f"[HI-Robot] Generating subtask for prompt: {original_prompt}")

        # Get initial prefix embeddings (images + DECOMPOSITION prompt from observation)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        
        # Initial forward pass to get KV cache
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )
        
        embedding_table = self.PaliGemma.llm.embedder["input_embedding"]
        generated_tokens = []
        
        # Current sequence length (for position tracking)
        current_length = prefix_tokens.shape[1]
        
        # Autoregressive generation loop with KV caching
        for step in range(max_tokens):
            # Get logits for the LAST position (which predicts the next token)
            last_embedding = prefix_out[:, -1:, :]  # Shape: (batch, 1, emb_dim)
            logits = jnp.einsum('bte,ve->btv', last_embedding, embedding_table)  # (batch, 1, vocab_size)
            
            if temperature > 0:
                rng, sample_rng = jax.random.split(rng)
                next_token_id = jax.random.categorical(
                    sample_rng, logits[0, 0] / temperature
                )
            else:
                # Greedy decoding
                next_token_id = jnp.argmax(logits[0, 0])
            
            next_token_int = int(next_token_id)
            
            # Check for EOS token (1) or newline (108) to stop generation early
            if next_token_int == 1 or next_token_int == 108:
                if next_token_int == 1:
                    logger.log(level=103, msg=f"[HI-Robot] EOS token reached at step {step}")
                else:
                    logger.log(level=103, msg=f"[HI-Robot] Newline token reached at step {step}")
                break

            generated_tokens.append(next_token_int)
            
            # Embed the new token
            next_token_embedding = self.PaliGemma.llm(
                jnp.array([[next_token_int]]), method="embed"
            )  # Shape: (1, 1, emb_dim)
            
            # Update position for the new token
            next_position = jnp.array([[current_length]], dtype=jnp.int32)
            current_length += 1
            
            # Create attention mask for new token (can attend to all previous tokens)
            # New token attends to: all prefix tokens + all previously generated tokens + itself
            new_attn_mask = jnp.ones((1, 1, current_length), dtype=jnp.bool_)
            
            # Forward pass for ONLY the new token, reusing KV cache
            (new_out, _), kv_cache = self.PaliGemma.llm(
                [next_token_embedding, None],
                mask=new_attn_mask,
                positions=next_position,
                kv_cache=kv_cache,  # Reuse cached keys/values from previous tokens
            )
            
            # Update prefix_out for next iteration
            prefix_out = new_out
        
        logger.log(level=103, msg=f"[HI-Robot] Generated {len(generated_tokens)} tokens: {generated_tokens}")
        
        return generated_tokens

