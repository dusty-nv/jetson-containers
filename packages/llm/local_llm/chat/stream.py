#!/usr/bin/env python3
import threading

from local_llm.utils import ends_with_token


class StreamingResponse():
    """
    Asynchronous output token iterator returned from a model's generate() function.
    Use it to stream the reply from the LLM as they are decoded token-by-token:
    
        ```
        response = model.generate("Once upon a time,")
        
        for token in response:
            print(token, end='', flush=True)
        ```
    
    To terminate processing prematurely, call the .stop() function, which will stop the model
    from generating additional output tokens.  Otherwise tokens will continue to be filled.
    """
    def __init__(self, model, input, **kwargs):
        super().__init__()
        
        self.model = model
        self.input = input
        self.event = threading.Event()
        self.kwargs = kwargs
        self.kv_cache = kwargs.get('kv_cache', None)
        
        self.stopping = False  # set if the user requested early termination
        self.stopped = False   # set when generation has actually stopped
        
        self.output_tokens = []  # accumulated output tokens so far
        self.output_text = ''    # detokenized output text so far
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped:
            stop_tokens = self.kwargs.get('stop_tokens', [self.model.tokenizer.eos_token_id])
            if not ends_with_token(self.output_tokens, stop_tokens, self.model.tokenizer):
                self.output_tokens.append(self.model.tokenizer.eos_token_id) # add EOS if necessary
                return self.get_message_delta()
            raise StopIteration
            
        self.event.wait()
        self.event.clear()

        return self.get_message_delta()
        
    def stop(self):
        self.stopping = True

    def get_message_delta(self):
        message = self.model.tokenizer.decode(self.output_tokens, skip_special_tokens=False) #, clean_up_tokenization_spaces=None
        delta = message[len(self.output_text):]
        self.output_text = message
        return delta