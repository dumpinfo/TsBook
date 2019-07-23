import time
from bs4 import BeautifulSoup
import requests
#from apps.rgl.spider_html_render import SpiderHtmlRender
import execjs
import json
import demjson
import csv
from apps.rgl.seph_spider import SephSpider as SephSpider

class WebsiteStats(object):
    pc_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
    pc_cookie = 'UM_distinctid=15dabfd5e91430-0c7e81214924c3-66547728-1fa400-15dabfd5e92894; qHistory=aHR0cDovL3Rvb2wuY2hpbmF6LmNvbS90b29scy9odHRwdGVzdC5hc3B4K+WcqOe6v0hUVFAgUE9TVC9HRVTmjqXlj6PmtYvor5V8aHR0cDovL3MudG9vbC5jaGluYXouY29tL3Rvb2xzL3JvYm90LmFzcHgr5pCc57Si6JyY6Jub44CB5py65Zmo5Lq65qih5ouf5oqT5Y+WfGh0dHA6Ly9zZW8uY2hpbmF6LmNvbStTRU/nu7zlkIjmn6Xor6J8aHR0cDovL3JhbmsuY2hpbmF6LmNvbSvnmb7luqbmnYPph43mn6Xor6J8aHR0cDovL3Rvb2wuY2hpbmF6LmNvbSvnq5nplb/lt6Xlhbc='
    
    post_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        #'Cookie': pc_cookie,
        'User-Agent': pc_user_agent
    }
    
    get_headers = {
        #'Cookie': pc_cookie,
        'User-Agent': pc_user_agent
    }
    
    @staticmethod
    def get_sub_domain_percent(domain, sub_domain):
        html = SephSpider.get_full_webpage('http://alexa.chinaz.com/{0}'.format(domain))
        soup = BeautifulSoup(html, 'lxml')
        tag_obj = soup.select('#form > div:nth-of-type(9) > div.mt1.subOlist.h70.sOlist > ul:nth-of-type(6) > li:nth-of-type(2)')
        if len(tag_obj) < 1:
            return 1.0
        try:
            return float(tag_obj[0].get_text()[0:-1])/100.0
        except Exception as ex:
            return 1.0
    
    @staticmethod
    def run_stats(params):
        crack_website_file = 'd:/awork/rungo/spider/crack_website_list.csv'
        recs = []
        to_be_processed = []
        with open(crack_website_file, 'r', newline='') as csv_file:
            rows = csv.reader(csv_file, delimiter=',', quotechar='|')
            for row in rows:
                to_be_processed.append(row)
        epoch = 0
        rows = to_be_processed
        to_be_processed = []
        
        while len(rows) > 1 and epoch<1:
            idx = 1
            for row in rows:
                try:
                    website = str(row[2])
                    alexa_top, cn_top, class_top, baidu_weight = WebsiteStats.get_all_tops(website)
                    ip_num = WebsiteStats.get_alexa_ip_num(website)
                    pv_num = WebsiteStats.get_alexa_pv_num(website)
                    sub_domain = str(row[3])
                    if '' != sub_domain:
                        percent = WebsiteStats.get_sub_domain_percent(website, sub_domain)
                        ip_num = int(float(ip_num)*percent)
                        pv_num = int(float(pv_num)*percent)
                    print('{8}:  {0}({7}): bw:{1} a:{2} cn:{3} c:{4} IP:{5}, PV:{6}'.format(website, baidu_weight, alexa_top, cn_top, class_top, ip_num, pv_num, sub_domain, idx))
                    item = [row[0], row[1], row[2], baidu_weight, alexa_top, cn_top, class_top, ip_num, pv_num]
                    recs.append(item)
                    if alexa_top<0 or cn_top<0 or class_top<0 or ip_num<0 or pv_num<0:
                        to_be_processed.append(row)
                    time.sleep(1)
                except Exception as ex:
                    print('############# Exception: {0}'.format(ex))
                    pass
                idx += 1
            rows = to_be_processed
            epoch += 1
        
        result_file = 'd:/awork/rungo/spider/crack_website_new.csv'
        with open(result_file, 'w', newline='') as csv_file:
            cw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            cw.writerow(['侵权网站', 'URL', '域名', '百度权重', '全球排名', '中国排名', '类目排名', '近一周IP', '近一周PV'])
            for rec in recs:
                print('write:{0}'.format(rec))
                cw.writerow(rec)
                
    @staticmethod
    def get_all_tops(website):
        html = SephSpider.get_full_webpage('http://seo.chinaz.com/{0}'.format(website))
        soup = BeautifulSoup(html, 'lxml')
        alexa_top = WebsiteStats.get_alexa_top(soup)
        cn_top = WebsiteStats.get_cn_top(soup)
        class_top = WebsiteStats.get_class_top(soup)
        baidu_weight = WebsiteStats.get_baidu_weight(soup)
        return alexa_top, cn_top, class_top, baidu_weight
        
    @staticmethod
    def get_alexa_top(soup):
        tag_obj = soup.select('#seoinfo > div > ul > li:nth-of-type(1) > div.SeoMaWr01Right > div:nth-of-type(1) > a:nth-of-type(2) > font')
        if len(tag_obj) < 1:
            return -1
        return WebsiteStats.convert_to_int(tag_obj[0].get_text().replace(',', ''))
        
    @staticmethod
    def get_cn_top(soup):
        tag_obj = soup.select('#topRankedSpan > em:nth-of-type(1)')
        if len(tag_obj) < 1:
            return -1
        return WebsiteStats.convert_to_int(tag_obj[0].get_text().replace(',', ''))
        
    @staticmethod
    def get_class_top(soup):
        tag_obj = soup.select('#topRankedSpan > em:nth-of-type(2)')
        if len(tag_obj) < 1:
            return -1
        return WebsiteStats.convert_to_int(tag_obj[0].get_text().replace(',', ''))
        
    @staticmethod
    def convert_to_int(text):
        try:
            return int(text.replace(',', ''))
        except Exception as ex:
            return -1
        
    @staticmethod
    def get_baidu_weight(soup):
        tag_obj = soup.select('#seoinfo > div > ul > li:nth-of-type(2) > div.SeoMaWr01Right > div:nth-of-type(1) > p > a > img')
        if len(tag_obj) < 1:
            return -1
        tag_str = tag_obj[0].get('src')
        start_pos = tag_str.rfind('/')
        end_pos = tag_str.rfind('.gif')
        return tag_str[start_pos + 1 : end_pos]
        
    
    @staticmethod
    def get_alexa_ip_num(website):        
        url = 'http://alexa.chinaz.com/Handlers/GetAlexaIpNumHandler.ashx'
        post_data = {
            'url': website
        }
        wb_data = requests.post(url, headers=WebsiteStats.post_headers, data=post_data)
        try:
            obj = demjson.decode(wb_data.text)
            if len(obj) < 1:
                return -1
            return int(obj[-1]['data']['IpNum'])
        except Exception as ex:
            pass
        
    @staticmethod
    def get_alexa_pv_num(website):        
        url = 'http://alexa.chinaz.com/Handlers/GetAlexaPvNumHandler.ashx'
        post_data = {
            'url': website
        }
        wb_data = requests.post(url, headers=WebsiteStats.post_headers, data=post_data)
        try:
            obj = demjson.decode(wb_data.text)
            if len(obj) < 1:
                return -1
            return int(obj[-1]['data']['PvNum'])
        except Exception as ex:
            pass
        return -1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
