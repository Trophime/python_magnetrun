#!/usr/bin/env python3
#-*- coding:utf-8 -*-

"""Magnet Record Object"""

import json
import datetime
from .requests.connect import download

class MRecord:
    """
    timestamp
    housing
    site
    link
    """

    def __init__(self, timestamp: datetime.datetime, housing: str, site: str, link: str):
        """default constructor"""
        self.timestamp = timestamp
        self.housing = housing
        self.site = site
        self.link = link

    def __repr__(self):
        """
        representation of object
        """
        return "%s(timestamp=%r, housing=%r, site=%r, link=%r)" % \
            (self.__class__.__name__,
             self.timestamp,
             self.housing,
             self.site,
             self.link
            )

    def getTimestamp(self):
        """get timestamp"""
        return self.timestamp

    def getHousing(self):
        """get experimental site"""
        return self.housing

    def getSite(self):
        """get experimental site magnet"""
        return self.site

    def getLink(self):
        """get link"""
        return self.link

    def setTimestamp(self, timestamp):
        """set timestamp"""
        self.timestamp = timestamp

    def setSite(self, site):
        """set Site"""
        self.site = site

    def setLink(self, link):
        """set Link"""
        self.link = link

    def getData(self, session, url, save=False, debug=False):
        """download record"""
        if not session:
            raise Exception("MRecord.download: no session defined")
    
        params = 'file=%s&download=1' % self.link
        data = download(session, url, params, self.link, debug)
        # print('MRecord/getData:', data)
        return data

    def saveData(self, data):
        filename = self.link.replace('../../../','')
        filename = filename.replace('/','_').replace('%20','-')
        # print(f"save to {filename}")
        fo = open(filename, "w", newline='\n')
        fo.write(data)
        fo.close()

    def to_json(self):
        """
        convert to json
        """
        from . import deserialize
        return json.dumps(self, default=deserialize.serialize_instance, sort_keys=True, indent=4)
    
    def __eq__(self, other):
        """compare MRecords"""
        if (isinstance(other, MRecord)):
            if self.timestamp != other.timestamp:
                return False
            if self.site != other.site:
                return False
            if self.link != other.link:
                return False
            return True
        return False
    
    # def __le__(self, other):
    #     """compare MRecords"""
    #     if (isinstance(other, MRecord)):
    #         return self.timestamp <= other.timestamp:
    #     return False
        
    # def __ge__(self, other):
    #     """compare MRecords"""
    #     if (isinstance(other, MRecord)):
    #         return self.timestamp >= other.timestamp:
    #     return False
        
