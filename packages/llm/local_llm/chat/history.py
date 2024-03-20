#!/usr/bin/env python3
import os
import json
import inspect
import logging
import numpy as np

from .templates import ChatTemplate, ChatTemplates
from ..utils import AttributeDict, ImageExtensions, ImageTypes, replace_text, print_table


def ChatEntry(role='user', msg=None, **kwargs):
    """
    Create a chat entry consisting of a text message, image, ect as input.  
    
    Parameters:
    
      role (str) -- The chat's turn template to apply, typically 'user' or 'bot'.
                    The role should have a corresponding entry in the active ChatTemplate.
                    
      msg (str|image) -- If a string, it should either contain text or a path
                         to a txt, json, or image file that will be loaded.

                         If an image, can be a np.ndarray, torch.Tensor, or PIL.Image.
                                  
                         If a dict, it will be passed through as the ChatEntry.
                                  
                         The embedding type of 'message' will attempt to be automatically
                         determined as text/image/ect, but if not possible, you should
                         explicitly set its type by passing it in via kwargs instead.
     
    kwargs:
    
       For messages whose embedding type is unable to be determined automatically, or to create
       a chat entry containing multiple message types, pass them in via kwargs like this:
       
         `entry = ChatEntry(role='user', text='abc', image='xyz.jpg')`
         
    Returns:
    
       A dict that has keys for 'role', 'text', 'image', ect.  This will return an AttributeDict,
       so you can access it like entry.role, entry.text, and so on.
    """    
    entry = AttributeDict(role=role, **kwargs)
    
    if msg is not None:
        entry[ChatHistory.embedding_type(msg)] = msg
        
    return entry
    
    
class ChatHistory():
    """
    Multimodal chat history that can contain a mix of media including text/images.
    
    ChatHistory objects can be indexed like a list of chat entry dicts,
    where each entry dict may have keys for 'text', 'image', 'role', ect.
    
       `chat_history[n]` will return the n-th chat entry

    Each type of media has a different embedding function (e.g. LLM's typically 
    do text token embedding internally, and images use CLIP + projection layers). 
    From these, it assembles the embedding for the entire chat as input to the LLM.
    
    It uses templating to add the required special tokens as defined by different
    model architectures.  In normal 2-turn chat, there are 'user' and 'bot' roles
    defined, but arbitrary roles can be added, each with their own template.
    The system prompt can also be configured through the chat template.
    
    TODO:  better caching of system prompt embeddings/ect
    """
    def __init__(self, model, chat_template=None, system_prompt=None, **kwargs):
        """
        Parameters:
           
           model (LocalLM) -- the model instance used for embeddings
           
           chat_template (str|dict) -- either a chat template dict, or the name of the 
                                       chat template to use like 'llama-2', 'vicuna-v1'
                                       If None, will attempt to determine model type.
                                  
           system_prompt (str) -- set the default system prompt
                                  if None, will use system prompt from the template.
                                  
           print_stats (bool) -- if True, generation performance will be printed to the terminal after EOS.
                                 This also gets enabled by default if --debug or --verbose is used.
        """
        self.model = model
        self.kv_cache = None
        
        if not chat_template:
            self.template = ChatTemplate(model)
            if self.template is None:
                raise RuntimeError(f"Couldn't automatically determine model type from {model.config.name}, please set the --chat-template argument")
            logging.info(f"using chat template '{self.template.name}' for model {model.config.name}")
        elif isinstance(chat_template, str):
            if os.path.isfile(chat_template):
                with open(chat_template) as template_file:
                    self.template = AttributeDict(json.load(template_file))
            else:
                self.template = AttributeDict(ChatTemplates[chat_template])
        elif isinstance(chat_template, dict):
            self.template = AttributeDict(template)
        else:
            raise TypeError(f"chat_template should be a str or dict (was {type(chat_template)})")
            
        if 'stop' in self.template:
            if not isinstance(self.template.stop, list):
                self.template.stop = [self.template.stop]
                
            for i, stop in enumerate(self.template.stop):
                if isinstance(stop, str):
                    self.template.stop[i] = self.model.tokenizer(stop, add_special_tokens=False, return_tensors='np').input_ids.squeeze().tolist()
        else:
            self.template.stop = [self.model.tokenizer.eos_token_id]
         
        #self.template.stop = [x for x in self.template.stop if x >= 0]  # filter out ignored stop tokens
        logging.info(f"model '{self.model.config.name}', chat template '{self.template.name}' stop tokens:  {self.model.tokenizer.batch_decode(self.template.stop)} -> {self.template.stop}")      

        if system_prompt:
            self.template['system_prompt'] = system_prompt

        self.embedding_functions = {}
        
        self.register_embedding('text', self.embed_text)
        self.register_embedding('dict', self.embed_dict)
        self.register_embedding('image', self.embed_image)
    
        self.print_stats = kwargs.get('print_stats', kwargs.get('debug', False))
        
        self.reset()

    @property
    def num_tokens(self):
        """
        Return the number of tokens used by the chat so far.
        embed_chat() needs to have been called for this to be upated,
        because otherwise the input wouldn't have been tokenized yet.
        """
        position = 0
        for n, entry in enumerate(self.entries):
            keys = self.valid_entry_keys(entry)
            for key in keys:
                embed_key = key + '_embedding'
                if embed_key in entry:
                    position += entry[embed_key].shape[1]
        return position
        
    def __len__(self):
        """
        Returns the number of entries in the chat history
        """
        return len(self.entries)
        
    def __getitem__(self, entry):
        """
        Return the n-th chat entry with the subscript indexing operator
        """
        return self.entries[entry]
        
    def append(self, role='user', msg=None, **kwargs):
        """
        Add a chat entry consisting of a text message, image, ect.
        See the ChatEntry() function for description of arguments.
        This can also accept an existing ChatEntry dict as msg.
        """
        if isinstance(msg, dict):
            self.entries.append(msg)
        else:
            self.entries.append(ChatEntry(role, msg, **kwargs))
        return self.entries[-1]

    def reset(self, add_system_prompt=True, wrap_tokens=None):
        """
        Reset the chat history, and optionally add the system prompt to the new chat.
        """
        if wrap_tokens:
            wrap_entry = self.find_wrap_entry(wrap_tokens)
            if wrap_entry:
                logging.warning(f"Wrapping chat to keep the most recent {len(self.entries)-wrap_entry} messages")
                self.entries = self.entries[wrap_entry:]
            else:
                logging.warning(f"Chat history overflow couldn't find previous chat entry to wrap to (clearing chat)")
                self.entries = []
        else:
            self.entries = []

        self.kv_cache = None
        self.image_embedding = None
        
        if add_system_prompt and 'system' in self.template:
            self.append(role='system', text=self.template['system_prompt'])
     
    def to_list(self):
        """
        Serialize the history to a list of dicts, where each dict is a chat entry
        with the non-critical keys removed (suitable for web transport, ect)
        """
        history = []
        
        for entry in self.entries:
            keys = self.valid_entry_keys(entry, is_embedding=False)
            
            if not keys:
                continue
                
            history.append({key: entry[key] for key in keys})
            
        return history
        
    @property
    def system_prompt(self):
        """
        Get the system prompt, the typically hidden instruction at the beginning
        of the chat like "You are a curious and helpful AI assistant, ..."
        """
        return self.template.get('system_prompt', '')
        
    @system_prompt.setter
    def system_prompt(self, instruction):
        """
        Set the system prompt instruction string and reset the chat history.
        TODO make it so this doesn't reset the chat history, but uncaches it.
        """
        self.template['system_prompt'] = instruction
        self.reset()
        
    def embed(self, input, type=None, **kwargs):
        """
        Get the embedding for a general input (text, image, ect)
        
        The embedding type is typically 'text', 'image', and will attempted
        to be determined automatically if it isn't explicitly specified. 
        Paths that end in image extensions are assumed to be an image, 
        otherwise strings are treated as text.
        
        The kwargs are passed through to the input type's embedding function.
        """
        if not type:
            type = self.embedding_type(input)
            
        if type not in self.embedding_functions:
            raise ValueError(f"type '{type}' did not have an embedding registered")
            
        return self.embedding_functions[type].func(input, **kwargs)
        
    def embed_text(self, text, template=None, use_cache=False, return_tokens=False, **kwargs):
        """
        Get the text embedding after applying the template for 'user', 'bot', ect.
        """
        if template:
            text = replace_text(template, {'${MESSAGE}': text})

        if return_tokens:
            embedding = self.model.tokenize(text)
        else:
            embedding = self.model.embed_text(text, use_cache=use_cache)
            
        logging.debug(f"embedding text {embedding.shape} {embedding.dtype} -> ```{text}```".replace('\n', '\\n'))
        
        return embedding
    
    def embed_dict(self, dict, **kwargs):
        """
        Get the embedding of a chat entry dict that can contain multiple embedding types.
        """
        embeddings = []
        
        for key, value in dict.items():
            if value is None:
                continue
            if key in self.embedding_functions:
                embeddings.append(self.embed(value, type=key, **kwargs))
                
        if len(embeddings) == 0:
            raise ValueError("dict did not contain any entries with valid embedding types")
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return np.concatenate(embeddings, axis=1)
                
    def embed_image(self, image, template=None, **kwargs):
        """
        Given an image, extract features and perfom the image embedding.
        This uses the CLIP encoder and a projection model that
        maps it into the embedding space the model expects.
        
        This is only applicable to vision VLM's like Llava and Mini-GPT4,
        and will throw an exception if model.has_vision is False.
        """
        embeddings = [] 

        if template: # get the text embedding for the template prefix
            template = template.split("${MESSAGE}")[0]
            
            if len(template) > 0:
                embeddings.append(self.embed_text(template, use_cache=True))
                logging.debug(f"image template:  ```{template}```")

        image_outputs = self.model.embed_image(image, return_tensors='np', return_dict=True)
        self.image_embedding = image_outputs.image_embeds
        
        embeddings.append(image_outputs.embedding)
        embeddings.append(self.embed_text('\n', use_cache=True))
        
        embeddings = np.concatenate(embeddings, axis=1)
        
        if self.print_stats:
            print_table(self.model.vision.stats)
            
        logging.debug(f"embedding image {embeddings.shape} {embeddings.dtype}")
        
        return embeddings

    def embed_chat(self, use_cache=True, max_tokens=None, wrap_tokens=None, **kwargs):
        """
        Assemble the embedding of either the latest or entire chat.
        
        If use_cache is true (the default), and only the new embeddings will be returned.
        If use_cache is set to false, then the entire chat history will be returned.
        
        The kwargs are passed to the embedding functions - for example, return_tokens=True
        will return tokens for the chat rather than embeddings.
        
        This function returns an (embedding, position) tuple, where the embedding array
        contains the new embeddings (or tokens) from the chat, and position is the current
        overall position in the history (up to the model's context window length)
        
        If the number of tokens in the chat history exceeds the length given in `max_tokens` argument
        (which is typically the model's context window, minus the max generation length),
        then the chat history will drop all but the latest `wrap_tokens`, starting with a user prompt.
        If `max_tokens` is provided but `wrap_tokens` is not, then the overflow tokens will be truncated.
        """
        embeddings = []
        position = 0
        
        num_user_prompts = 0
        open_user_prompt = False

        for i, entry in enumerate(self.entries):
            keys = self.valid_entry_keys(entry)
            
            if not keys:
                logging.warning(f"chat entry {i} had no valid/registered keys ({entry.keys()})")
                continue
                
            for key in keys:
                if 'first' in self.template and entry['role'] == 'user' and num_user_prompts == 0:
                    role_template = self.template['first']  # if the first non-system message has a different template
                else:
                    if entry.role not in self.template:
                        raise RuntimeError(f"chat template {self.template_name} didn't have an entry for role={entry.role}")
                    role_template = self.template[entry.role]
                    if open_user_prompt:
                        role_template = role_template[role_template.find('${MESSAGE}'):] # user prompt needs closed out from an image
                        open_user_prompt = False
                
                embed_key = key + '_embedding'
                cached = embed_key in entry and use_cache
                
                if logging.getLogger().isEnabledFor(logging.DEBUG) and not cached:
                    logging.debug(f"processing chat entry {i}  role='{entry.role}' template='{role_template}' open_user_prompt={open_user_prompt} cached={'true' if cached else 'false'} {key}='{entry[key] if isinstance(entry[key], str) else type(entry[key])}'".replace('\n', '\\n'))
                
                if use_cache:
                    if embed_key not in entry: # TODO  and entry.role != 'bot'  -- only compute bot embeddings when needed
                        entry[embed_key] = self.embed(entry[key], type=key, template=role_template, **kwargs)
                        
                        # bot message already included in kv_cache, except trailing template
                        # TODO handle bot generation prompt and trailing template
                        if entry.role != 'bot':
                            embeddings.append(entry[embed_key])
                            use_cache = False  # all entries after this need to be included
                            if key == 'image':
                                open_user_prompt = True  # image is inside first half of a user prompt
                        else:
                            position += entry[embed_key].shape[1]
                    else:
                        position += entry[embed_key].shape[1]
                else:
                    if embed_key not in entry:
                        entry[embed_key] = self.embed(entry[key], type=key, template=role_template, **kwargs)
                        if key == 'image':
                            open_user_prompt = True
                    embeddings.append(entry[embed_key])
                
                if entry['role'] == 'user' and key == 'text':
                    num_user_prompts += 1
                
        embeddings = np.concatenate(embeddings, axis=1) #, position
        
        '''
        if max_tokens and position + embeddings.shape[1] > max_tokens:
            if wrap_tokens:
                self.reset(wrap_tokens=wrap_tokens)
                embeddings, position = self.embed_chat(use_cache=False, max_tokens=max_tokens, wrap_tokens=wrap_tokens, **kwargs)
                logging.warning(f"Chat overflow, max history lenth {max_tokens} tokens exceeded (keeping the most recent {embeddings.shape[1]} tokens)")
            else:
                logging.warning(f"Truncating chat history overflow to {max_tokens} tokens")
                return embeddings[:,:max_tokens,:], position
        '''
        
        return embeddings, position      

    def tokenize(self, use_cache=True, **kwargs):
        return self.embed_chat(use_cache=use_cache, return_tokens=True, **kwargs)
        
    def valid_entry_keys(self, entry, is_embedding=True):
        keys = []
        
        for key in entry:       
            if key.endswith('_embedding') or entry[key] is None:
                continue
                
            if is_embedding and key not in self.embedding_functions:
                continue
              
            #if exclude and key in exclude:
            #    continue
            
            keys.append(key)
            
        return keys
        
    def register_embedding(self, type, func):
        params = inspect.signature(func).parameters
        self.embedding_functions[type] = AttributeDict(
            func=func,
            uses_template=len(params) > 1 #'role_template' in params
        )
        
    @staticmethod
    def embedding_type(input):
        if isinstance(input, str):
            if input.endswith(ImageExtensions):
                return 'image'
            else:
                return "text" 
        elif isinstance(input, ImageTypes):
            return 'image'
        elif isinstance(input, list) and len(input) > 0 and isinstance(input[0], str):
            return 'text'
        else:
            raise ValueError(f"couldn't find type of embedding for {type(input)}, please specify the 'type' argument")
            
    def find_wrap_entry(self, wrap_tokens):
        """
        Find the oldest entry from which the chat doesn't exceed the number of wrap_tokens,
        and that the entry should be a user query.  This is used to keep those more recent
        chat entries when the history overflows past the max context window of the model.
        """
        position = 0
        for n in range(len(self.entries)-1, -1, -1):
            entry = self.entries[n]
            keys = self.valid_entry_keys(entry)
            for key in keys:
                embed_key = key + '_embedding'
                if embed_key in entry:
                    position += entry[embed_key].shape[1]
                    if position >= wrap_tokens:
                        for i in range(n+1, len(self.entries)):
                            if self.entries[i].role == 'user':
                                return i
            
