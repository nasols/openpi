import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download

import traceback, os

logger = logging.getLogger("openpi")

class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, ki_mode:bool = False, hi_mode:bool=False):
        self._max_len = max_len

        # traceback.print_stack(limit=30)

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._FAST_tokenizer = None 
        self._FAST_skip_tokens = 128
        self.ki_mode = ki_mode
        if ki_mode: 
            self._FAST_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        
        self.hi_mode = hi_mode

    
    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._tokenizer.vocab_size() - 1 - self._FAST_skip_tokens - tokens
        

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
    
    def tokenize_subtask(self, prompt:str, state:np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        """
        Function used to tokenize the prompt for producing subtasks in the HI-robot pipeline. 
        Output is the tokens for 'Task: xxx;\nSubtask: ' 
        NOTE! Can simply change between using and not using the state in the prompt. Test what works. 
        Used in inference to produce the subtask, i.e. the tokens we pass to the llm. 
        By doing so we can then use the 'build_tokenized_prompt' function for action generation.
        """

        assert isinstance(prompt, str), f"Expected prompt to be a string, got {type(prompt)}"
        assert state is not None, "State cannot be None for subtask tokenization --> Only supports Pi05"
        assert isinstance(state, np.ndarray), f"Expected state to be a numpy array, got {type(state)}"

        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        full_prompt = f"Task: {cleaned_text}; State: {state_str};\nSubtask: "
        tokens = self._tokenizer.encode(full_prompt, add_bos=True, add_eos=False)

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    def tokenize_raw_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize raw text WITHOUT BOS or special formatting.
        
        Used for teacher forcing where the text will be concatenated to an existing
        sequence that already has BOS token.
        
        Args:
            text: Raw text to tokenize (e.g., "pick up the cube")
            
        Returns:
            tokens: Token IDs (padded to max_len)
            mask: Boolean mask indicating valid tokens
        """
        cleaned_text = text.strip().replace("_", " ").replace("\n", " ")
        # Encode WITHOUT add_bos - no special tokens
        tokens = self._tokenizer.encode(cleaned_text, add_bos=False)
        
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
    
    def decode(self, tokens: np.ndarray) -> str:
        # Decode tokens to text
        decoded_text = self._tokenizer.decode(tokens.tolist())
        return decoded_text

    def build_tokenized_prompt(
            self, 
            prompt:str, 
            state:np.ndarray, 
            subtask:str|None=None, 
            actions:np.ndarray|None=None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main function for tokenizing and building a full prompt for training. 
        Expects input to be strings, i.e. not tokenized. 
        During inference, the inputs would be tokenized as we dont de-tokenize the subtask after generation, we keep it at token-level. 
        This function returns: 
        'tokens'        - the full list of tokens representing the prompt, state, subtask and actions (depending on use case)
        'token_mask'    - indicating tokens and padding, boolean list, True = token, False = padding. 
        'ar_mask'       - indicating which tokens should attend to which. True=causal attention, False=bidirectional attention. 
        'loss_mask'     - indicating which tokens should be included in loss calc. Are true for FAST tokens (if present) and tokens representing the subtask (if present)
        """
        cleaned_prompt = prompt.strip().replace("_", " ").replace("\n", " ")

        # ========= High level task building =========
        # ============================================
        if self.hi_mode: 
            sub_prompt = f"Task: {cleaned_prompt}; State: {state}; Subtask: "
            prompt_tokens = self._tokenizer.encode(sub_prompt, add_bos=True, add_eos=False)
            ar_mask = [True]*len(prompt_tokens) # Causal attention for the main prompt up to subtask generation
            loss_mask = [False]*len(prompt_tokens) # No loss on prompt and state tokens
            subtask_token_mask = [False]*len(prompt_tokens) # Indicating that these tokens are not subtask tokens
            action_token_mask = [False]*len(prompt_tokens) # Indicating that these tokens are not action tokens
            token_mask = [True]*len(prompt_tokens) # Mask for all non-padding tokens
        
        else: 
            sub_prompt = f"Task: {cleaned_prompt}; State: {state};\nAction: "
            prompt_tokens = self._tokenizer.encode(sub_prompt, add_bos=True, add_eos=False)
            ar_mask = [True]*len(prompt_tokens) # Causal attention for the entire prompt including "Action: ".
            loss_mask = [False]*len(prompt_tokens) # No loss on prompt and state tokens
            subtask_token_mask = [False]*len(prompt_tokens) # Indicating that these tokens are not subtask tokens
            action_token_mask = [False]*len(prompt_tokens) # Indicating that these tokens are not action tokens
            token_mask = [True]*len(prompt_tokens) # Mask for all non-padding tokens
        # ============================================
        # ============================================

        # ========= Low level task building ==========
        # ============================================
        # If subtask is None, as in the case when using HI-mode but not yet generated subtask, we want to return the tokens for 
        ## "Task:xxx; State: xxx; Subtask: " and the we run a pass to build the subtask. 
        ## Post generating subtask, we use the inference function that runs at token level. 
        if subtask is not None: # If subtask is not None, i.e. the string version of a subtask is given, we are training the model, the G.T subtask is given. 
            cleaned_subtask = subtask.strip().replace("_", " ").replace("\n", " ")
            subtask_tokens = self._tokenizer.encode(cleaned_subtask, add_bos=False, add_eos=False)
            
            tokens = prompt_tokens + subtask_tokens # Now tokens looks like "Task: xxx; State: xxx; Subtask: xxx"
            ar_mask += [True]*len(subtask_tokens) # We have causal attention up to action prediction. 
            subtask_loss_mask = loss_mask + [True]*len(subtask_tokens) # Loss on subtask tokens for subtask prediction
            action_loss_mask = loss_mask + [False]*len(subtask_tokens) # No loss on subtask tokens for action prediction
            subtask_token_mask = subtask_token_mask + [True]*len(subtask_tokens) # Mask indicating which tokens are subtask tokens
            action_token_mask = action_token_mask + [False]*len(subtask_tokens) # Mask indicating which tokens are action tokens
            token_mask += [True]*len(subtask_tokens) # Mask for all non-padding tokens
        
        elif subtask is None: 
            tokens = prompt_tokens
            subtask_loss_mask = loss_mask.copy()
            action_loss_mask = loss_mask.copy()
            

        # ============================================
        # ============================================

        # ========== Actions task building ===========
        # ============================================
        if actions is None and subtask is not None: # KI-Mode deactivated -- Inference ? 
            # At this point, tokens looks like "Task: xxx; State: xxx; Subtask: xxx" and we are done with building the prompt. We can return the tokenized prompt and masks.
            # The tokenized prompt should not include the actions as we are running inference. 
            # Before this point we expect the model to have predicted a subtask and it is fed into this function. 
            # Or we are running training without KI-mode, i.e. no actions in the training data. 
            ## In this case we are not predicting FAST tokens, and they are not provided. We want the prompt to end with "Action: " and the action expert to produce continuous tokens. 
            action_tokens = self._tokenizer.encode("\nAction: ", add_bos=False, add_eos=True) # We will use the presence of "Action: " in the decoded text to determine whether the model is trying to predict actions or not.
            tokens += action_tokens # Now tokens looks like "Task: xxx; State: xxx; Subtask: xxx \n Action: " 
            ar_mask += [True]*len(action_tokens) # Causal attention for the entire prompt including "Action: ".
            subtask_loss_mask += [False]*len(action_tokens) # No loss on "Action: " tokens for subtask prediction
            action_loss_mask += [False]*len(action_tokens) # No loss on "Action:
            subtask_token_mask += [False]*len(action_tokens) # "Action: " tokens are not subtask tokens
            action_token_mask += [False]*len(action_tokens) # "Action: " tokens are not action tokens
            token_mask += [True]*len(action_tokens) # Mask for all non-padding tokens
  
        elif actions is not None and self._FAST_tokenizer is not None: # KI-mode activated -- Training ?
            
            # If we are training and KI-Mode is activated, we expect observation to include actions, i.e. continuous action vectors. 
            # We then tokenizre these using the FAST tokenizer and append these to the tokens and build loss masks. 
            # During inference in this function, actions will never be provided. Look to prior if statement. 
            
            FAST_action_tokens = self._FAST_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(FAST_action_tokens)
            action_tokens =  action_tokens_in_pg.tolist() + self._tokenizer.encode("|", add_eos=True, add_bos=False)

            tokens += action_tokens # Now tokens looks like "Task: xxx; State: xxx; Subtask: xxx \n Action: <FAST_ACTION_TOKENS> |"
            ar_mask += [True]*len(action_tokens) 
            subtask_loss_mask += [False]*len(action_tokens) # No loss on action tokens for subtask prediction
            action_loss_mask += [True]*len(action_tokens) # Loss on action tokens for action prediction
            subtask_token_mask += [False]*len(action_tokens) # Action tokens are not subtask tokens
            action_token_mask += [True]*len(action_tokens) # Mask indicating which tokens are action tokens
            token_mask += [True]*len(action_tokens) # Mask for all non-padding tokens

        # ============================================
        # ============================================

        # ================= Padding ==================
        # ============================================
            

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens += padding
            token_mask += padding
            ar_mask += padding
            subtask_loss_mask += padding
            action_loss_mask += padding
            subtask_token_mask += padding
            action_token_mask += padding
        else: 
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            subtask_loss_mask = subtask_loss_mask[: self._max_len]
            action_loss_mask = action_loss_mask[: self._max_len]
            subtask_token_mask = subtask_token_mask[: self._max_len]
            action_token_mask = action_token_mask[: self._max_len]

        assert len(tokens)              == self._max_len, "Arrays have to be of length max_len after padding/truncation"
        assert len(token_mask)          == self._max_len, "Arrays have to be of length max_len after padding/truncation"
        assert len(ar_mask)             == self._max_len, "Arrays have to be of length max_len after padding/truncation"
        assert len(subtask_loss_mask)   == self._max_len, "Arrays have to be of length max_len after padding/truncation"
        assert len(action_loss_mask)    == self._max_len, "Arrays have to be of length max_len after padding/truncation"
        assert len(subtask_token_mask)  == self._max_len, "Arrays have to be of length max_len after padding/truncation"
        assert len(action_token_mask)   == self._max_len, "Arrays have to be of length max_len after padding/truncation"
        
        tokens = np.asarray(tokens)
        token_mask = np.asarray(token_mask, dtype=bool)
        ar_mask = np.asarray(ar_mask, dtype=int)
        subtask_loss_mask = np.asarray(subtask_loss_mask)
        action_loss_mask = np.asarray(action_loss_mask)
        subtask_token_mask = np.asarray(subtask_token_mask)
        action_token_mask = np.asarray(action_token_mask)

        return tokens, token_mask, ar_mask, subtask_loss_mask, action_loss_mask, subtask_token_mask, action_token_mask
    
    def build_tokenized_prompt_inference(
            self,
            prompt_tokens: np.ndarray,
            prompt_mask: np.ndarray,
            subtask_tokens: np.ndarray,
            subtask_mask: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build action-generation prompt on token level for inference.

        Expects:
            - prompt_tokens/prompt_mask from decomposition prompt tokenization,
              i.e. "Task: ...; State: ...; Subtask: "
            - subtask_tokens/subtask_mask generated by subtask decoder.

        Returns:
            - tokens/token_mask for "Task: ...; State: ...; Subtask: <generated>;\nAction: "
        """

        prompt_tokens = np.asarray(prompt_tokens).reshape(-1)
        prompt_mask = np.asarray(prompt_mask, dtype=bool).reshape(-1)
        subtask_tokens = np.asarray(subtask_tokens).reshape(-1)
        subtask_mask = np.asarray(subtask_mask, dtype=bool).reshape(-1)

        prompt_valid = prompt_tokens[prompt_mask].tolist()
        subtask_valid = subtask_tokens[subtask_mask].tolist()
        action_tokens = self._tokenizer.encode("\nAction: ", add_bos=False, add_eos=True)

        tokens = prompt_valid + subtask_valid + action_tokens
        token_mask = [True] * len(tokens)

        if len(tokens) < self._max_len:
            pad = [0] * (self._max_len - len(tokens))
            tokens += pad
            token_mask += [False] * len(pad)
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask, dtype=bool)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step, args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore())
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
