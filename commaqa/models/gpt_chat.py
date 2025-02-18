import logging
import time
import os
from functools import lru_cache
from openai import OpenAI
import openai
from diskcache import Cache
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit
client = OpenAI()

logger = logging.getLogger(__name__)


cache = Cache(os.path.expanduser("~/.cache/gpt3calls"))






def openai_call(
    prompt,
    engine,
    temperature,
    max_tokens,
    top_p,
    frequency_penalty,
    presence_penalty,
    stop,
    n,
    best_of,
    logprobs,
):
    function = client.chat.completions.create
    return function(
        messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
        model=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
    )





class GPTChatGenerator:
    def __init__(
        self,
        engine="gpt-3.5-turbo-instruct",
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
        retry_after_n_seconds=None,
        n=1,
        best_of=1,
        logprobs=0,
        remove_method="first",
    ):
        self.engine = engine
        self.logprobs = logprobs
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop
        self.temperature = temperature
        self.retry_after_n_seconds = retry_after_n_seconds
        self.remove_method = remove_method
        self.model_tokens_limit = 3500

    def generate_text_sequence(self, prompt):
        """
        :param input_text:
        :return: returns a sequence of tuples (string, score) where lower score is better
        """
        # GPT3 can't handle trailing white-space
        prompt = prompt.rstrip()

        prompt = fit_prompt_into_given_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_tokens,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            tokenizer_model_name="gpt2",  # did this before tiktoken was released.
            last_is_test_example=True,
        )
        #import pdb; pdb.set_trace()

        arguments = {
            "engine": self.engine,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "n": self.n,
            "best_of": self.best_of,
            "logprobs": self.logprobs,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
        }
        if self.best_of is not None:
            arguments["best_of"] = self.best_of


            
        response = openai_call(**arguments)




        output_seq_score = []

        for index, choice in enumerate(response.choices):
            output_seq_score.append((choice.message.content, index))

        return sorted(output_seq_score, key=lambda x: x[1])