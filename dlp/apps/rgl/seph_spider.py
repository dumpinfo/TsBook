from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import urllib
import time
import json
import sys
sys.path.append('C:/Users/yt/AppData/Local/Google/Chrome/Application')

class SephSpider(object):
    @staticmethod
    def get_full_webpage(url):
        driver = webdriver.Chrome()
        driver.get(url)
        return driver.page_source
    
    @staticmethod
    def get_full_webpage0(url):
        driver = webdriver.PhantomJS(executable_path='C:/Users/yt/AppData/Roaming/npm/node_modules/phantomjs/lib/phantom/bin/phantomjs.exe')
        #driver.set_page_load_timeout(60000)  
        #driver.set_script_timeout(60000)#这两种设置都进行才有效  
        driver.get(url)
        time.sleep(5)
        return driver.page_source
        
    @staticmethod
    def test():
        print('hello world')