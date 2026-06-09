import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi05_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

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

class Pi05Experimental(_model.BaseModel):

    def __init__(self, config: pi05_config.Pi05Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
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
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True   # This attribute gets automatically set by model.train() and model.eval().

        self.num_steps = 10

    def suffix_sum(self, arr, axis):
        return jnp.flip(
            jnp.cumsum(
                jnp.flip(
                    arr,
                    axis=axis
                ),
                axis=axis
            ),
            axis=axis
        )

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
        if not self.pi05:
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

    def get_denoiser(
            self,
            observation,
            batch_size
        ):
        observation = _model.preprocess_observation(None, observation, train=False)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def infer_velocity(t, x_t):
            # Revert to train-time convention x_1=noise, x_0=clean
            time = 1.0 - t

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            negative_v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            v_t = - negative_v_t
            return v_t
        
        return infer_velocity

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        unnorm_q01,
        unnorm_q99,
        # num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        
        def unnorm(x):
            return (x + 1.0) / 2.0 * (unnorm_q99 - unnorm_q01 + 1e-6) + unnorm_q01

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / self.num_steps
        batch_size = observation.state.shape[0]
        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry, _):
            time, x_t = carry
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

            x_next = x_t + dt * v_t
            time_next = time + dt
            next_carry = (time_next, x_next)

            x_t_unnorm = unnorm(x_t)
            record = (time, x_t_unnorm)

            return next_carry, record

        final_carry, records = jax.lax.scan(
            f=step,
            init=(1.0, noise),
            xs=None,
            length=self.num_steps,
        )

        t_0, x_0 = final_carry
        t_hist, x_hist = records

        # Append final time/state to the recorded scan history.
        t_hist = jnp.concatenate(
            [t_hist, jnp.asarray(t_0)[None]],
            axis=0,
        )
        x_0_unnorm = unnorm(x_0)
        x_hist = jnp.concatenate(
            [x_hist, x_0_unnorm[None, ...]],
            axis=0,
        )

        info = {
            't_hist': t_hist,
            'x_hist': x_hist
        }

        return x_0, info
    
    def sample_inpaint(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        unnorm_q01,
        unnorm_q99,
        Y,
        W,
        guidance_weights,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        
        dt = 1.0 / self.num_steps
        batch_size = observation.state.shape[0]
        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        infer_velocity = self.get_denoiser(
            observation=observation,
            batch_size=batch_size
        )

        def unnorm(x):
            return (x + 1.0) / 2.0 * (unnorm_q99 - unnorm_q01 + 1e-6) + unnorm_q01
        
        def step(carry, step_idx):
            time, x_t = carry

            def clean_estimate(x):
                v = infer_velocity(time, x)
                x_clean = x + (1.0 - time) * v
                x_clean_unnorm = unnorm(x_clean)
                return x_clean_unnorm, v

            x_1_hat_unnorm, pullback, v_t = jax.vjp(clean_estimate, x_t, has_aux=True)

            cotangent = - (x_1_hat_unnorm - Y[None, :, :]) * W[None, :, :]

            v_t_guidance, = pullback(cotangent)

            v_t_full = v_t + guidance_weights[step_idx] * v_t_guidance

            x_next = x_t + dt * v_t_full

            time_next = time + dt
            next_carry = (time_next, x_next)

            x_t_unnorm = unnorm(x_t)
            record = (time, x_t_unnorm)

            return next_carry, record


        final_carry, records = jax.lax.scan(
            f=step,
            init=(0.0, noise),
            xs=jnp.arange(self.num_steps),
            length=self.num_steps,
        )

        t_1, x_1 = final_carry
        t_hist, x_hist = records

        # Append final time/state to the recorded scan history.
        t_hist = jnp.concatenate(
            [t_hist, jnp.asarray(t_1)[None]],
            axis=0,
        )
        x_1_unnorm = unnorm(x_1)
        x_hist = jnp.concatenate(
            [x_hist, x_1_unnorm[None, ...]],
            axis=0,
        )

        info = {
            't_hist': t_hist,
            'x_hist': x_hist
        }

        return x_1, info
    
    def sample_inpaint_abs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        unnorm_q01,
        unnorm_q99,
        Y,
        W,
        guidance_weights,
        q_obs,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        
        dt = 1.0 / self.num_steps
        batch_size = observation.state.shape[0]
        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        infer_velocity = self.get_denoiser(
            observation=observation,
            batch_size=batch_size
        )

        def unnorm(x):
            return (x + 1.0) / 2.0 * (unnorm_q99 - unnorm_q01 + 1e-6) + unnorm_q01
        
        def step(carry, step_idx):
            time, x_t = carry

            def clean_estimate(x):
                v = infer_velocity(time, x)
                x_clean = x + (1.0 - time) * v
                x_clean_unnorm = unnorm(x_clean)
                return x_clean_unnorm, v
            
            x_1_hat_unnorm, pullback, v_t = jax.vjp(clean_estimate, x_t, has_aux=True)

            Q = q_obs[None, None, :] + 1.0/15.0 * jnp.cumsum(x_1_hat_unnorm, axis=1)
            Errors = (Q - Y[None, :, :]) * W[None, :, :]
            cotangent = - 1.0/15.0 * self.suffix_sum(Errors, axis=1)
            
            v_t_guidance, = pullback(cotangent)
            v_t_full = v_t + guidance_weights[step_idx] * v_t_guidance

            x_next = x_t + dt * v_t_full
            time_next = time + dt
            next_carry = (time_next, x_next)

            x_t_unnorm = unnorm(x_t)
            record = (time, x_t_unnorm)

            return next_carry, record


        final_carry, records = jax.lax.scan(
            f=step,
            init=(0.0, noise),
            xs=jnp.arange(self.num_steps),
            length=self.num_steps,
        )

        t_1, x_1 = final_carry
        t_hist, x_hist = records

        # Append final time/state to the recorded scan history.
        t_hist = jnp.concatenate(
            [t_hist, jnp.asarray(t_1)[None]],
            axis=0,
        )
        x_1_unnorm = unnorm(x_1)
        x_hist = jnp.concatenate(
            [x_hist, x_1_unnorm[None, ...]],
            axis=0,
        )

        info = {
            't_hist': t_hist,
            'x_hist': x_hist,
        }

        return x_1, info    

    def sample_inpaint_pose(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        unnorm_q01,
        unnorm_q99,
        Y_pos,
        W_pos,
        guidance_weights,
        q_obs,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        
        dt = 1.0 / self.num_steps
        batch_size = observation.state.shape[0]
        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        infer_velocity = self.get_denoiser(
            observation=observation,
            batch_size=batch_size
        )

        def unnorm(x):
            return (x + 1.0) / 2.0 * (unnorm_q99 - unnorm_q01 + 1e-6) + unnorm_q01

        def step(carry, step_idx):
            time, x_t = carry

            def clean_estimate(x):
                v = infer_velocity(time, x)
                x_clean = x + (1.0 - time) * v
                x_clean_unnorm = unnorm(x_clean)
                return x_clean_unnorm, v
            
            x_1_hat_unnorm, pullback, v_t = jax.vjp(clean_estimate, x_t, has_aux=True)

            # shape (B, 15, 7)
            Q = q_obs[None, None, :] + 1.0/15.0 * jnp.cumsum(x_1_hat_unnorm[:, :, :7], axis=1)

            # shape (B, 15, 3)
            P_pos = jax.vmap(
                jax.vmap(forward_pos, in_axes=0, out_axes=0),
                in_axes=0, out_axes=0,
            )(Q)

            # shape (B, 15, 3, 7)
            J_pos = jax.vmap(
                jax.vmap(get_jacobian_pos, in_axes=0, out_axes=0),
                in_axes=0, out_axes=0,
            )(Q[:, :, :7])

            # shape (B, 15, 7)
            Errors = jnp.einsum(
                'ij, bij, bijq -> biq',
                W_pos, (P_pos - Y_pos[None, :, :]), J_pos
            )

            # shape (B, 15, 32)
            zero_pad = jnp.zeros((Errors.shape[0], 15, 32-7), dtype=jnp.float32)
            Errors_padded = jnp.concatenate(
                [Errors, zero_pad],
                axis=2,
            )

            cotangent = - 1.0/15.0 * self.suffix_sum(Errors_padded, axis=1)

            v_t_guidance, = pullback(cotangent)

            v_t_full = v_t + guidance_weights[step_idx] * v_t_guidance

            x_next = x_t + dt * v_t_full

            time_next = time + dt
            next_carry = (time_next, x_next)

            x_t_unnorm = unnorm(x_t)
            L = 0.5 * jnp.linalg.norm(W_pos * (P_pos[0] - Y_pos), ord='fro')**2     # NOTE: not batched
            record = (time, x_t_unnorm, L)

            return next_carry, record


        final_carry, records = jax.lax.scan(
            f=step,
            init=(0.0, noise),
            xs=jnp.arange(self.num_steps),
            length=self.num_steps,
        )

        t_1, x_1 = final_carry
        t_hist, x_hist, L_hist = records

        # Append final time/state to the recorded scan history.
        t_hist = jnp.concatenate(
            [t_hist, jnp.asarray(t_1)[None]],
            axis=0,
        )
        x_1_unnorm = unnorm(x_1)
        x_hist = jnp.concatenate(
            [x_hist, x_1_unnorm[None, ...]],
            axis=0,
        )
        # Final guidance loss
        Q_final = q_obs[None, None, :] + 1.0/15.0 * jnp.cumsum(x_1_unnorm[:, :, :7], axis=1)
        P_pos_final = jax.vmap(jax.vmap(forward_pos, in_axes=0, out_axes=0), in_axes=0, out_axes=0,)(Q_final)
        L_final = 0.5 * jnp.linalg.norm(W_pos * (P_pos_final[0] - Y_pos), ord='fro')**2
        L_hist = jnp.concatenate(
            [L_hist, jnp.asarray(L_final)[None]],
            axis=0,
        )

        info = {
            't_hist': t_hist,
            'x_hist': x_hist,
            'L_hist': L_hist,
        }

        return x_1, info
    
    def sample_RTC_and_CG(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        unnorm_q01,
        unnorm_q99,
        q_obs,      # (32,)     excluding gripper
        Y_RTC,      # (H, 32)   full abs-space joint angles + gripper, time-shifted and padded
        w_RTC,      # (H,)
        g_RTC,      # (num_steps,)
        w_gripper_RTC,  # float
        Y_CG_pos,       # (H, 3)
        W_CG_pos,       # (H, 3)
        g_CG,       # (num_steps,)
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """
        Computes RTC loss in abs joint space.
        Currently supports CG in xyz-Cartesian positional space
        """

        dt = 1.0 / self.num_steps
        batch_size = observation.state.shape[0]
        
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        infer_velocity = self.get_denoiser(
            observation=observation,
            batch_size=batch_size
        )

        def unnorm(x):
            return (x + 1.0) / 2.0 * (unnorm_q99 - unnorm_q01 + 1e-6) + unnorm_q01

        def get_RTC_cotangent(
                Q,              # (B, H, 7)
                Gripper,        # (B, H)
            ):
            # (B, H, 7)
            Errors_Q = (Q - Y_RTC[None, :, :7]) * w_RTC[None, :, None]
            sum_Errors_Q = 1.0/15.0 * self.suffix_sum(Errors_Q, axis=1)

            # (B, H)
            Errors_Gripper = (Gripper - Y_RTC[None, :, 7]) * w_RTC[None, :] * w_gripper_RTC

            # (B, H, 32-7-1)
            zero_pad = jnp.zeros((Errors_Q.shape[0], 15, 32-7-1), dtype=jnp.float32)

            # (B, H, 32)
            cotangent = - jnp.concatenate(
                [
                    sum_Errors_Q,               # (B, H, 7)
                    Errors_Gripper[:, :, None], # (B, H, 1)
                    zero_pad,                   # (B, H, 32-7-1)
                ],
                axis=-1,
            )
            return cotangent

        def get_CG_cotangent(
                Q,              # (B, H, 7)
            ):
            # shape (B, H, 3)
            P_pos = jax.vmap(
                jax.vmap(forward_pos, in_axes=0, out_axes=0),
                in_axes=0, out_axes=0,
            )(Q)

            # shape (B, H, 3, 7)
            J_pos = jax.vmap(
                jax.vmap(get_jacobian_pos, in_axes=0, out_axes=0),
                in_axes=0, out_axes=0,
            )(Q)

            # shape (B, H, 7)
            Errors = jnp.einsum(
                'ij, bij, bijq -> biq',
                W_CG_pos, (P_pos - Y_CG_pos[None, :, :]), J_pos
            )
            sum_Errors = 1.0/15.0 * self.suffix_sum(Errors, axis=1)

            # shape (B, H, 32)
            zero_pad = jnp.zeros((Errors.shape[0], 15, 32-7), dtype=jnp.float32)
            cotangent = - jnp.concatenate(
                [
                    sum_Errors,               # (B, H, 7)
                    zero_pad,                 # (B, H, 32-7)
                ],
                axis=-1,
            )
            return cotangent

        def step(carry, step_idx):
            time, x_t = carry

            def clean_estimate(x):
                v = infer_velocity(time, x)
                x_clean = x + (1.0 - time) * v
                x_clean_unnorm = unnorm(x_clean)
                return x_clean_unnorm, v
            
            x_1_hat_unnorm, pullback, v_t = jax.vjp(clean_estimate, x_t, has_aux=True)

            # shape (B, H, 7)
            Q = q_obs[None, None, :] + 1.0/15.0 * jnp.cumsum(x_1_hat_unnorm[:, :, :7], axis=1)
            
            # shape (B, H)
            Gripper = x_1_hat_unnorm[:, :, 7]

            # shape (B, H, 32)
            cotangent = (
                  get_RTC_cotangent(Q, Gripper) * g_RTC[step_idx]
                + get_CG_cotangent(Q)           * g_CG[step_idx]
            )

            v_t_guidance, = pullback(cotangent)
            v_t_full = v_t + v_t_guidance

            x_next = x_t + dt * v_t_full
            time_next = time + dt
            next_carry = (time_next, x_next)

            x_t_unnorm = unnorm(x_t)
            record = (time, x_t_unnorm)

            return next_carry, record

        final_carry, records = jax.lax.scan(
            f=step,
            init=(0.0, noise),
            xs=jnp.arange(self.num_steps),
            length=self.num_steps,
        )

        t_1, x_1 = final_carry
        t_hist, x_hist = records

        # Append final time/state to the recorded scan history.
        t_hist = jnp.concatenate(
            [t_hist, jnp.asarray(t_1)[None]],
            axis=0,
        )
        x_1_unnorm = unnorm(x_1)
        x_hist = jnp.concatenate(
            [x_hist, x_1_unnorm[None, ...]],
            axis=0,
        )

        info = {
            't_hist': t_hist,
            'x_hist': x_hist,
        }

        return x_1, info
    













def rpy_to_R(rpy):
    """
    URDF fixed-axis RPY:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    r = rpy[0]
    p = rpy[1]
    y = rpy[2]

    cr = jnp.cos(r)
    sr = jnp.sin(r)
    cp = jnp.cos(p)
    sp = jnp.sin(p)
    cy = jnp.cos(y)
    sy = jnp.sin(y)

    return jnp.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=jnp.float32,
    )


def make_T(xyz, rpy):
    R = rpy_to_R(rpy)
    xyz = xyz.astype(jnp.float32)

    upper = jnp.concatenate([R, xyz[:, None]], axis=1)
    lower = jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

    return jnp.concatenate([upper, lower], axis=0)


def make_Rz_joint(theta):
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    return jnp.array(
        [
            [c, -s, 0.0, 0.0],
            [s,  c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )


JOINT_XYZ = jnp.array(
    [
        [0.0, 0.0, 0.333],
        [0.0, 0.0, 0.0],
        [0.0, -0.316, 0.0],
        [0.0825, 0.0, 0.0],
        [-0.0825, 0.384, 0.0],
        [0.0, 0.0, 0.0],
        [0.088, 0.0, 0.0],
    ],
    dtype=jnp.float32,
)

JOINT_RPY = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [-jnp.pi / 2.0, 0.0, 0.0],
        [jnp.pi / 2.0, 0.0, 0.0],
        [jnp.pi / 2.0, 0.0, 0.0],
        [-jnp.pi / 2.0, 0.0, 0.0],
        [jnp.pi / 2.0, 0.0, 0.0],
        [jnp.pi / 2.0, 0.0, 0.0],
    ],
    dtype=jnp.float32,
)


FIXED_TCP_T = (
    make_T(
        jnp.array([0.0, 0.0, 0.107], dtype=jnp.float32),
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    )
    @ make_T(
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([0.0, 0.0, -jnp.pi / 4.0], dtype=jnp.float32),
    )
    @ make_T(
        jnp.array([0.0, 0.0, 0.150], dtype=jnp.float32),
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    )
)

FIXED_JOINT_T = jnp.concatenate(
    [
        jax.vmap(make_T)(JOINT_XYZ, JOINT_RPY),
        FIXED_TCP_T[None, :, :],
    ],
    axis=0,
)

FIXED_JOINT_R = FIXED_JOINT_T[:7, :3, :3]
FIXED_JOINT_P = FIXED_JOINT_T[:7, :3, 3]

FIXED_TCP_R = FIXED_JOINT_T[7, :3, :3]
FIXED_TCP_P = FIXED_JOINT_T[7, :3, 3]


def forward_pos(q):
    """
    Forward geometry for Franka Panda.

    Input:
        q: shape (7,), dtype float32

    Output:
        pos: shape (3,)
              [x, y, z]
    """
    def step(T, inputs):
        fixed_T_i, q_i = inputs
        T_next = T @ fixed_T_i @ make_Rz_joint(q_i)
        return T_next, None

    T, _ = jax.lax.scan(
        step,
        jnp.eye(4, dtype=jnp.float32),
        (FIXED_JOINT_T[:7], q),
    )

    T = T @ FIXED_JOINT_T[7]

    return T[:3, 3]


def postmul_Rz(R, theta, out=None):
    c = jnp.cos(theta)
    s = jnp.sin(theta)

    # out = R @ Rz(theta)
    col0 = c * R[:, 0] + s * R[:, 1]
    col1 = -s * R[:, 0] + c * R[:, 1]
    col2 = R[:, 2]

    return jnp.stack([col0, col1, col2], axis=1)


def get_jacobian_pos(q):
    """
    Geometric Jacobian for Franka Panda TCP.

    Input:
        q: shape (7,), dtype float32

    Output:
        J: shape (3, 7)     maps qdot to TCP linear velocity in world frame
    """
    def step(carry, inputs):
        R, p = carry
        fixed_R_i, fixed_p_i, q_i = inputs

        R_joint = R @ fixed_R_i
        p_next = p + R @ fixed_p_i

        # Joint origin in world
        joint_p_i = p_next

        # Joint z-axis in world
        joint_z_i = R_joint[:, 2]

        # Continue through actuated revolute joint
        R_next = postmul_Rz(R_joint, q_i)

        return (R_next, p_next), (joint_p_i, joint_z_i)

    (R, p), (joint_p, joint_z) = jax.lax.scan(
        step,
        (
            jnp.eye(3, dtype=jnp.float32),
            jnp.zeros(3, dtype=jnp.float32),
        ),
        (FIXED_JOINT_R, FIXED_JOINT_P, q),
    )

    # Apply fixed TCP offset
    p_ee = p + R @ FIXED_TCP_P

    r = p_ee[None, :] - joint_p

    # Linear part: z_i cross (p_ee - p_i)
    J_pos = jnp.cross(joint_z, r).T

    # Angular part: z_i
    J_rot = joint_z.T

    J = jnp.concatenate([J_pos, J_rot], axis=0)

    return J[:3, :]     # ONLY POS







### Pose helpers
# # FIXME make JAX compatible

# # Pre-compute joint transformations at initialization
# def rpy_to_R(rpy):
#     """
#     URDF fixed-axis RPY:
#         R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
#     """
#     r = rpy[0]
#     p = rpy[1]
#     y = rpy[2]

#     cr = jnp.cos(r)
#     sr = jnp.sin(r)
#     cp = jnp.cos(p)
#     sp = jnp.sin(p)
#     cy = jnp.cos(y)
#     sy = jnp.sin(y)

#     R = jnp.empty((3, 3), dtype=jnp.float32)

#     R[0, 0] = cy * cp
#     R[0, 1] = cy * sp * sr - sy * cr
#     R[0, 2] = cy * sp * cr + sy * sr

#     R[1, 0] = sy * cp
#     R[1, 1] = sy * sp * sr + cy * cr
#     R[1, 2] = sy * sp * cr - cy * sr

#     R[2, 0] = -sp
#     R[2, 1] = cp * sr
#     R[2, 2] = cp * cr

#     return R

# def make_T(xyz, rpy):
#     T = jnp.eye(4, dtype=jnp.float32)

#     R = rpy_to_R(rpy)

#     T[0, 0] = R[0, 0]
#     T[0, 1] = R[0, 1]
#     T[0, 2] = R[0, 2]

#     T[1, 0] = R[1, 0]
#     T[1, 1] = R[1, 1]
#     T[1, 2] = R[1, 2]

#     T[2, 0] = R[2, 0]
#     T[2, 1] = R[2, 1]
#     T[2, 2] = R[2, 2]

#     T[0, 3] = xyz[0]
#     T[1, 3] = xyz[1]
#     T[2, 3] = xyz[2]

#     return T

# def make_Rz_joint(theta):
#     c = jnp.cos(theta)
#     s = jnp.sin(theta)

#     T = jnp.eye(4, dtype=jnp.float32)

#     T[0, 0] = c
#     T[0, 1] = -s
#     T[1, 0] = s
#     T[1, 1] = c

#     return T

# JOINT_XYZ = jnp.array(
#     [
#         [0.0, 0.0, 0.333],
#         [0.0, 0.0, 0.0],
#         [0.0, -0.316, 0.0],
#         [0.0825, 0.0, 0.0],
#         [-0.0825, 0.384, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.088, 0.0, 0.0],
#     ],
#     dtype=jnp.float32,
# )
# JOINT_RPY = jnp.array(
#     [
#         [0.0, 0.0, 0.0],
#         [-jnp.pi / 2.0, 0.0, 0.0],
#         [jnp.pi / 2.0, 0.0, 0.0],
#         [jnp.pi / 2.0, 0.0, 0.0],
#         [-jnp.pi / 2.0, 0.0, 0.0],
#         [jnp.pi / 2.0, 0.0, 0.0],
#         [jnp.pi / 2.0, 0.0, 0.0],
#     ],
#     dtype=jnp.float32,
# )
# FIXED_JOINT_T = jnp.empty((8, 4, 4), dtype=jnp.float32)
# for i in range(7):
#     FIXED_JOINT_T[i] = make_T(JOINT_XYZ[i], JOINT_RPY[i])
# FIXED_JOINT_T[7] = (    # TCP offset
#     make_T(
#         jnp.array([0.0, 0.0, 0.107], dtype=jnp.float32),
#         jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
#     )
#     @ make_T(
#         jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
#         jnp.array([0.0, 0.0, -jnp.pi / 4.0], dtype=jnp.float32),
#     )
#     @ make_T(
#         jnp.array([0.0, 0.0, 0.150], dtype=jnp.float32),
#         jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
#     )
# )

# def forward_pos(q):
#     """
#     Forward geometry for Franka Panda.

#     Input:
#         q: shape (7,), dtype float32

#     Output:
#         pos: shape (3,)
#               [x, y, z]
#     """
#     T = (
#           FIXED_JOINT_T[0] @ make_Rz_joint(q[0])
#         @ FIXED_JOINT_T[1] @ make_Rz_joint(q[1])
#         @ FIXED_JOINT_T[2] @ make_Rz_joint(q[2])
#         @ FIXED_JOINT_T[3] @ make_Rz_joint(q[3])
#         @ FIXED_JOINT_T[4] @ make_Rz_joint(q[4])
#         @ FIXED_JOINT_T[5] @ make_Rz_joint(q[5])
#         @ FIXED_JOINT_T[6] @ make_Rz_joint(q[6])
#         @ FIXED_JOINT_T[7]
#     )

#     return T[:3, 3]




# def postmul_Rz(R, theta, out):
#     c = jnp.cos(theta)
#     s = jnp.sin(theta)

#     # out = R @ Rz(theta)
#     for i in range(3):
#         r0 = R[i, 0]
#         r1 = R[i, 1]
#         r2 = R[i, 2]

#         out[i, 0] = c * r0 + s * r1
#         out[i, 1] = -s * r0 + c * r1
#         out[i, 2] = r2


# FIXED_JOINT_R = jnp.empty((7, 3, 3), dtype=jnp.float32)
# FIXED_JOINT_P = jnp.empty((7, 3), dtype=jnp.float32)

# for i in range(7):
#     FIXED_JOINT_R[i] = FIXED_JOINT_T[i, :3, :3]
#     FIXED_JOINT_P[i] = FIXED_JOINT_T[i, :3, 3]

# FIXED_TCP_R = FIXED_JOINT_T[7, :3, :3].copy()
# FIXED_TCP_P = FIXED_JOINT_T[7, :3, 3].copy()


# def get_jacobian_pos(q):
#     """
#     Geometric Jacobian for Franka Panda TCP.

#     Input:
#         q: shape (7,), dtype float32

#     Output:
#         J: shape (3, 7)     maps qdot to TCP linear velocity in world frame
#     """

#     joint_p = jnp.empty((7, 3), dtype=jnp.float32)
#     joint_z = jnp.empty((7, 3), dtype=jnp.float32)

#     R = jnp.eye(3, dtype=jnp.float32)
#     # R_joint = jnp.empty((3, 3), dtype=jnp.float32)
#     R_next = jnp.empty((3, 3), dtype=jnp.float32)

#     p = jnp.zeros(3, dtype=jnp.float32)

#     for i in range(7):
#         # matmul3(R, FIXED_JOINT_R[i], R_joint)
#         R_joint = R @ FIXED_JOINT_R[i]

#         px = p[0] + R[0, 0] * FIXED_JOINT_P[i, 0] + R[0, 1] * FIXED_JOINT_P[i, 1] + R[0, 2] * FIXED_JOINT_P[i, 2]
#         py = p[1] + R[1, 0] * FIXED_JOINT_P[i, 0] + R[1, 1] * FIXED_JOINT_P[i, 1] + R[1, 2] * FIXED_JOINT_P[i, 2]
#         pz = p[2] + R[2, 0] * FIXED_JOINT_P[i, 0] + R[2, 1] * FIXED_JOINT_P[i, 1] + R[2, 2] * FIXED_JOINT_P[i, 2]

#         # Joint origin in world
#         joint_p[i, 0] = px
#         joint_p[i, 1] = py
#         joint_p[i, 2] = pz

#         # Joint z-axis in world
#         joint_z[i, 0] = R_joint[0, 2]
#         joint_z[i, 1] = R_joint[1, 2]
#         joint_z[i, 2] = R_joint[2, 2]

#         # Continue through actuated revolute joint:
#         postmul_Rz(R_joint, q[i], R_next)

#         for r in range(3):
#             for c in range(3):
#                 R[r, c] = R_next[r, c]

#         p[0] = px
#         p[1] = py
#         p[2] = pz

#     # Apply fixed TCP offset:
#     p_ee_x = p[0] + R[0, 0] * FIXED_TCP_P[0] + R[0, 1] * FIXED_TCP_P[1] + R[0, 2] * FIXED_TCP_P[2]
#     p_ee_y = p[1] + R[1, 0] * FIXED_TCP_P[0] + R[1, 1] * FIXED_TCP_P[1] + R[1, 2] * FIXED_TCP_P[2]
#     p_ee_z = p[2] + R[2, 0] * FIXED_TCP_P[0] + R[2, 1] * FIXED_TCP_P[1] + R[2, 2] * FIXED_TCP_P[2]

#     J = jnp.empty((6, 7), dtype=jnp.float32)

#     for i in range(7):
#         zx = joint_z[i, 0]
#         zy = joint_z[i, 1]
#         zz = joint_z[i, 2]

#         rx = p_ee_x - joint_p[i, 0]
#         ry = p_ee_y - joint_p[i, 1]
#         rz = p_ee_z - joint_p[i, 2]

#         # Linear part: z_i cross (p_ee - p_i)
#         J[0, i] = zy * rz - zz * ry
#         J[1, i] = zz * rx - zx * rz
#         J[2, i] = zx * ry - zy * rx

#         # Angular part: z_i
#         J[3, i] = zx
#         J[4, i] = zy
#         J[5, i] = zz

#     return J[:3, :]     # ONLY POS




