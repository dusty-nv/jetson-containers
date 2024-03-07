#!/usr/bin/env python3
import time
import logging

from datetime import datetime
from local_llm import Plugin, StopTokens


class EventFilter(Plugin):
    """
    Plugin that filters output text from LLM and triggers events when it matches/changes
    """
    def __init__(self, filters=None, server=None, **kwargs):
        """
        Parameters:
        
          filters (str or list[str]) -- see parse_filters() function
          kwargs (dict) -- these get passed to the Plugin constructor
        """
        super().__init__(threaded=False, **kwargs)

        self.tags = None
        self.prompt = None
        self.history = []
        self.server = server
        self.filter_type = 'or'
        
        self.parse_filters(filters)
        
        if self.server:
            self.server.add_message_handler(self.on_websocket)
        
    def process(self, text, prompt=None, **kwargs):
        """
        Detect if the criteria for an event filters occurred in the incoming text.
        
        Parameters:
        
          input (str) -- text to filter/search against
                                          
        Returns:
        
          Event dict if a new event occurred, false otherwise
        """
        filters = self.filter_text(text, self.filters, op=self.filter_type)
        
        if not text or not filters:
            if self.history and 'end' not in self.history[-1]:
                self.on_event_end(self.history[-1])
            return
            
        new_event = False
        
        if not self.history:
            new_event = True
        elif 'end' in self.history[-1]:
            new_event = True
        elif self.history[-1]['filters'] != filters:
            new_event = True
            self.on_event_end(self.history[-1])
            
        if new_event:
            return self.on_event_begin(text, filters, prompt=prompt)
        else:
            self.history[-1]['last'] = time.time()
            self.send_events(self.history)
            
    def on_event_begin(self, text, filters, prompt=None):
        event = {
            'id': len(self.history),
            'text': text.strip(),
            'filters': filters,
            'begin': time.time(),
            'last': time.time(),
        }
        
        if prompt:
            event['prompt'] = prompt
        
        if self.tags:
            event['tags'] = self.tags
            alert_text = f"EVENT OCCURRED  '{event['tags']}'"
        else:
            alert_text = f"EVENT OCCURRED  {event['filters']}"

        if self.server:
            event['alert'] = self.server.send_alert(alert_text, category='event_begin', level='warning')
        
        self.history.append(event)
        self.send_events(self.history)
        
        return event
        
    def on_event_end(self, event):
        event['end'] = time.time()
        if self.server:
            self.server.send_message({'end_alert': event['alert']['id']})
            alert_text = f"EVENT FINISHED  '{event.get('tags', event['filters'])}'  (duration {event['end']-event['begin']:.1f} seconds)"
            self.server.send_alert(alert_text, category='event_end', level='success')
        self.send_events(self.history)
        
    def parse_filters(self, filters):
        if not filters:
            self.filters = None
            return
            
        filters = filters.split('+')
        
        if len(filters) > 1:
            self.filter_type = 'and'
        else:
            filters = filters[0].split(',')
            self.filter_type = 'or'
            
        self.filters = [x.strip().lower() for x in filters]
        return self.filters
 
    def filter_text(self, text, filters, op='or'):
        if not text:
            return []
        if not filters:
            return []
        
        matches = [x for x in filters if x in text.lower()]
        
        if op == 'and' and len(matches) != len(filters):
            return []
            
        return matches

    def format_event(self, event):
        event = event.copy()
        time_format = '%-I:%M:%S'
        
        event['begin'] = datetime.fromtimestamp(event['begin']).strftime(time_format)
        event['last'] = datetime.fromtimestamp(event['last']).strftime(time_format)
        
        if 'end' in event:
            event['end'] = datetime.fromtimestamp(event['end']).strftime(time_format)
        else:
            event['end'] = event['last']
            
        event['filters'] = str(event['filters'])
        
        for stop in StopTokens:
            event['text'] = event['text'].replace(stop, '')
            
        del event['alert']
        return event
        
    def send_events(self, events, max_events=10):
        if max_events and len(events) > max_events:
            events = events[-max_events:]
        events = [self.format_event(event) for event in events]
        if self.server:
            self.server.send_message({'events': events})
      
    def on_websocket(self, msg, msg_type=0, metadata='', **kwargs):
        if not isinstance(msg, dict):  # msg_type != WebServer.MESSAGE_JSON:
            return 
        if 'event_filters' in msg:
            self.parse_filters(msg['event_filters'])
            logging.info(f'set event filters to "{msg["event_filters"]}" {self.filters}')
        elif 'event_tags' in msg:
            self.tags = msg['event_tags']
                