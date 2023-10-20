#!/usr/bin/env python3
import threading


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
        #self.queue = queue.Queue()
        self.event = threading.Event()
        self.stopped = False
        self.kwargs = kwargs
        self.kv_cache = kwargs.get('kv_cache', None)
        
        self.output_tokens = []
        self.output_text = ''
        
    def __iter__(self):
        return self

    def __next__(self):
        if self.stopped:
            if len(self.output_tokens) == 0 or self.output_tokens[-1] != self.model.tokenizer.eos_token_id:
                self.output_tokens.append(self.model.tokenizer.eos_token_id) # add EOS if necessary
                return self.get_message_delta()
            raise StopIteration
            
        self.event.wait()
        self.event.clear()

        return self.get_message_delta()
        
    def stop(self):
        self.stopped = True

    def get_message_delta(self):
        message = self.model.tokenizer.decode(self.output_tokens, skip_special_tokens=False) #, clean_up_tokenization_spaces=None
        delta = message[len(self.output_text):]
        self.output_text = message
        return delta