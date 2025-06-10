#!/usr/bin/env python3
import logging

from local_llm import Plugin, LocalLM, ChatHistory
from local_llm.utils import ImageTypes, print_table


class ChatQuery(Plugin):
    """
    Plugin that feeds incoming text or ChatHistory to LLM and generates the reply.
    It can either internally manage the ChatHistory, or that can be done externally.
    
    Inputs:  (str or list[str]) -- one or more text prompts
             (dict) -- an existing ChatEntry dict
             (ChatHistory) -- use latest entry from chat history
     
    Outputs:  channel 0 (str) -- the partially-generated output text, token-by-token
              channel 1 (str) -- the partially-generated output text, word-by-word
              channel 2 (str) -- the entire final output text, after generation is complete
              channel 3 (StreamingReponse) -- the stream iterator from generate()    
              channel 4 (ndarray) -- the last CLIP image features/embeddings
    """
    OutputToken = 0
    OutputWords = 1
    OutputFinal = 2
    OutputStream = 3
    OutputImageEmbedding = 4
    
    def __init__(self, model="meta-llama/Llama-2-7b-chat-hf", **kwargs):
        """
        Parameters:
        
          model (str|LocalLM) -- model name/path or loaded LocalLM model instance
          
        kwargs (these get passed to LocalLM.from_pretrained() if model needs loaded)
        
          api (str) -- the model backend API to use:  'auto_gptq', 'awq', 'mlc', or 'hf'
                       if left as None, it will attempt to be automatically determined.
          quant (str) -- for AWQ or MLC, either specify the quantization method,
                         or the path to the quantized model (AWQ and MLC API's only)
          vision_model (str) -- for VLMs, override the vision embedding model (CLIP)
                                otherwise, it will use the CLIP variant from the config.
                                
        kwargs (these get passed to ChatHistory initializer)
        
          chat_template (str|dict) -- either a chat template dict, or the name of the 
                                       chat template to use like 'llama-2', 'vicuna-v1'
                                       If None, will attempt to determine model type.        
          system_prompt (str) -- set the default system prompt
                                 if None, will use system prompt from the template.
                                 
        kwargs (generation parameters)

          max_new_tokens (int) -- the number of tokens to output in addition to the prompt (default: 128)
          min_new_tokens (int) -- force the model to generate a set number of output tokens (default: -1)
          do_sample (bool) -- if True, temperature/top_p will be used.  Otherwise, greedy search (default: False)
          repetition_penalty -- the parameter for repetition penalty. 1.0 means no penalty (default: 1.0)  
          temperature (float) -- randomness token sampling parameter (default=0.7, only used if do_sample=True)
          top_p (float) -- if set to float < 1 and do_sample=True, only the smallest set of most probable tokens
                           with probabilities that add up to top_p or higher are kept for generation (default 0.95)  

        kwargs (other)

          print_stats (bool) -- if True, generation performance will be printed to the terminal after EOS.
                                This also gets enabled if --debug or --verbose is used, and passed to ChatHistory.          
        """
        super().__init__(output_channels=5, **kwargs)

        if isinstance(model, str):
            self.model = LocalLM.from_pretrained(model, **kwargs)
        else:
            self.model = model
         
        self.stream = None
        self.history = ChatHistory(self.model, **kwargs)

        self.max_context_len = self.model.config.max_length
        self.max_new_tokens = kwargs.get('max_new_tokens', 128)
        self.min_new_tokens = kwargs.get('min_new_tokens', -1)
        self.wrap_tokens = kwargs.get('wrap_tokens', 512)
        
        self.do_sample = kwargs.get('do_sample', False)
        self.repetition_penalty = kwargs.get('repetition_penalty', 1.0)
        self.temperature = kwargs.get('temperature', 0.7)
        self.top_p = kwargs.get('top_p', 0.95)
            
        self.print_stats = kwargs.get('print_stats', kwargs.get('debug', False))
        
        #warmup_query = '2+2 is '
        #logging.debug(f"Warming up LLM with query '{warmup_query}'")
        #logging.debug(f"Warmup response:  '{self.model.generate(warmup_query, streaming=False)}'")
     
    @property
    def chat_history(self):
        return self.history.to_list()
        
    @property
    def chat_tokens(self):
        return self.history.num_tokens
        
    @property
    def chat_state(self):
        return self.chat_history, self.chat_tokens, self.max_context_len
        
    def config(self, **kwargs):
        """
        Config plugin settings from expanded dict
        """
        if 'max_new_tokens' in kwargs:
            self.max_new_tokens = int(kwargs['max_new_tokens'])
            
        if 'vision_scaling' in kwargs:
            self.model.vision_scaling = kwargs['vision_scaling']
        
    def process(self, input, **kwargs):
        """
        Generate the reply to a prompt or the latest ChatHistory.
        
        Parameters:
        
          input (str|dict|ChatHistory) -- either a string prompt from the user,
                                          a ChatEntry dict, or ChatHistory dict.
                                          
        Returns:
        
          The generated text (token by token), if input was a string or dict.
          If input was a ChatHistory, returns the streaming iterator/generator.
        """
        self.config(**kwargs)
        
        if input is None:
            return
            
        if isinstance(input, list):
            for x in input:
                self.process(x, **kwargs)
            return

        if self.interrupted:
            return
            
        # handle some special commands
        if isinstance(input, str):
            x = input.lower()
            if any([x == y for y in ('/reset', '/clear', 'reset', 'clear')]):
                self.history.reset()
                return
        
        # add prompt to chat history
        if isinstance(input, str) or isinstance(input, dict) or isinstance(input, ImageTypes):
            self.history.append(role='user', msg=input)
            chat_history = self.history
        elif isinstance(input, ChatHistory):
            chat_history = input  # TODO also recieve chat history as list for cross-process
        else:
            raise TypeError(f"ChatQuery plugin expects inputs of type str, dict, image, or ChatHistory (was {type(input)})")

        # images should be followed by text prompts
        if 'image' in chat_history[-1] and 'text' not in chat_history[-1]:
            logging.debug("image message, waiting for user prompt")
            return
        
        # get the latest chat embeddings
        embedding, position = chat_history.embed_chat(
            max_tokens=self.model.config.max_length - self.max_new_tokens,
            wrap_tokens=self.wrap_tokens
        )
        
        # output vision features
        if chat_history.image_embedding is not None:
            self.output(chat_history.image_embedding, ChatQuery.OutputImageEmbedding)
            
        # start generating output
        self.stream = self.model.generate(
            embedding, 
            streaming=True, 
            kv_cache=chat_history.kv_cache,
            stop_tokens=chat_history.template.stop,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            **kwargs
        )
        
        # output the stream iterator on channel 3
        self.output(self.stream, ChatQuery.OutputStream)

        # output the generated tokens on channel 0
        bot_reply = chat_history.append(role='bot', text='')
        words = ''
        
        for token in self.stream:
            if self.interrupted:
                logging.debug(f"LLM interrupted, terminating request early")
                self.stream.stop()
                
            # sync the reply with the entire text, so that multi-token
            # unicode unicode sequences are detokenized and decoded together
            bot_reply.text = self.stream.output_text
            
            # output stream of raw tokens
            self.output(token, ChatQuery.OutputToken)
            
            # if a space was added, emit new word(s)
            words += token
            last_space = words.rfind(' ')
            
            if last_space >= 0:
                self.output(words[:last_space+1], ChatQuery.OutputWords)
                if last_space < len(words) - 1:
                    words = words[last_space+1:]
                else:
                    words = ''
            
        if len(words) > 0:
            self.output(words, ChatQuery.OutputWords)
            
        bot_reply.text = self.stream.output_text
        self.history.kv_cache = self.stream.kv_cache
        self.stream = None
        
        # output the final generated text on channel 2
        self.output(bot_reply.text, ChatQuery.OutputFinal)
    
        if self.print_stats:
            print_table(self.model.stats)
    '''
    def interrupt(self, clear_inputs=True, block=True):
        """
        Interrupt any ongoing/pending processing, and optionally clear the input queue.
        
        """
        super().interrupt(clear_inputs=clear_inputs)
        
        while self.stream is not None:
            self.stream.stop()
    '''
    