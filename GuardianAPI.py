#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## The Guardian ""
import requests
import json
import getpass

class Guardian_API:
    
    def __init__(self):
        self.api_key = getpass.getpass('api_key: ')
    
    def get_content(self, parameters):
                
        parameter_string = ""
        for next in parameters.items():
            parameter_string +="{}={}&".format(next[0],next[1])
        
        #print(parameter_string)
        response = requests.get("https://content.guardianapis.com/search?{}page=1&api-key={}".format(parameter_string, self.api_key))
        response_json = json.loads(response.text)
                
        if response_json["response"]["status"] != "ok":
            print(response_json["response"])
            return([])
        
        all_results = []
        all_results.extend(response_json["response"]["results"])
        
        npages = response_json["response"]["pages"]
        for page in range(2,npages+1):
            page_response = requests.get("https://content.guardianapis.com/search?{}page={}&api-key={}".format(parameter_string, page, self.api_key))
            
            page_response_json = json.loads(page_response.text)

            all_results.extend(page_response_json["response"]["results"])
        
        return all_results
