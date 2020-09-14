#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The Guardian ""
import requests
from requests_toolbelt import sessions
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import json
import getpass

class GuardianAPI:
    
    def __init__(self):
        self.api_key = getpass.getpass('api_key: ')
        self.endpoint = sessions.BaseUrlSession(base_url="https://content.guardianapis.com")
        adapter = HTTPAdapter(max_retries=Retry(total = 3,
                                                backoff_factor = 1,
                                                status_forcelist = [429, 500, 502, 503, 504]))
        self.endpoint.mount('https://', adapter)
    
    def get_content(self, parameters):
        response = self.endpoint.get("/search", params = parameters, headers = {'api-key':self.api_key,
                                                                                'format':'json'})
        if response.status_code != 200:
            raise Exception('API Error; status_code:{}, status:{}'.format(response.status_code, response.text))
        

        response_json = json.loads(response.text)                
        all_results = []
        all_results.extend(response_json["response"]["results"])
        npages = response_json["response"]["pages"]
        for page in range(2,npages+1):
            parameters['page'] = page
            page_response = self.endpoint.get("/search", params = parameters, headers = {'api-key':self.api_key,
                                                                                'format':'json'})
            if page_response.status_code != 200:
                raise Exception('API Error; status_code:{}, status:{}'.format(response.status_code, response.text))
    
            page_response_json = json.loads(page_response.text)
            all_results.extend(page_response_json["response"]["results"])
        return all_results
