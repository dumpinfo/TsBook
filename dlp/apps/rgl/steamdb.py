import sys
from bs4 import BeautifulSoup
import requests
#from apps.rgl.spider_html_render import SpiderHtmlRender
import execjs
import json
import demjson
import csv
import urllib
from apps.rgl.seph_spider import SephSpider as SephSpider
from apps.rgl.website_stats import WebsiteStats as WebsiteStats

class SteamDb(object):
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
    def get_icon_image(appid):
        url = 'https://steamdb.info/app/{0}/'.format(appid)
        wb_data = requests.get(url, headers=SteamDb.get_headers)
        soup = BeautifulSoup(wb_data.text, 'lxml')
        icon_obj = soup.select('body > div.footer-wrap > div.scope-app > div > div > div.pagehead.clearfix > img')
        img_obj = soup.select('body > div.footer-wrap > div.scope-app > div > div > div.row.app-row > div.span4 > img')
        icon_url = icon_obj[0].attrs['src']
        img_url = 'https://steamdb.info/{0}'.format(img_obj[0].attrs['src'])
        return icon_url, img_url
    
    @staticmethod
    def get_steam_apps():
        print('get steam apps...')
        page_sum = 980 + 1
        for page_num in range(57, page_sum):
            games = []
            print('process page:{0}! '.format(page_num))
            url = 'https://steamdb.info/apps/page{0}/'.format(page_num)
            wb_data = requests.get(url, headers=SteamDb.get_headers)
            soup = BeautifulSoup(wb_data.text, 'lxml')
            if page_sum < 1:
                page_sum_obj = soup.select('body > div.footer-wrap > div.header-wrapper > div > h1.header-title.pull-right')
                page_sum_str = page_sum_obj[0].text
                page_sum = int(page_sum_str[page_sum_str.rfind('/')+1:]) + 1
            for row in range(1, 10000000):
                game = {}
                app_img = soup.select('body > div.footer-wrap > div.container > table > tbody > tr:nth-of-type({0}) > td.applogo > img'.format(row))
                if len(app_img) <= 0:
                    break # 已经读完所有Table中的内容
                app_img_src = app_img[0].get('src')
                appid_obj = soup.select('body > div.footer-wrap > div.container > table > tbody > tr:nth-of-type({0}) > td:nth-of-type(2) > a'.format(row))
                appid = appid_obj[0].text
                app_name_obj = soup.select('body > div.footer-wrap > div.container > table > tbody > tr:nth-of-type({0}) > td:nth-of-type(3) > a.b'.format(row))
                if len(app_name_obj) > 0:
                    app_name = app_name_obj[0].text
                else:
                    app_name = 'noname'
                app_type_obj = soup.select('body > div.footer-wrap > div.container > table > tbody > tr:nth-of-type({0}) > td:nth-of-type(3) > i'.format(row))
                app_type = app_type_obj[0].text
                if 'Game' == app_type:
                    icon_url, img_url = SteamDb.get_icon_image(appid)
                    game['steamId'] = appid
                    game['articleName'] = app_name
                    game['type'] = 1
                    game['articleIcon'] = icon_url
                    game['articleImage'] = img_url
                    games.append(game)
            print('upload {0} page'.format(page_num))
            url = 'http://47.95.119.120/pada/index.php?f=c_ajax&c=CAjax&m=importSteamDbRecsAjax'        
            #post_data = urllib.parse.urlencode(game).encode('utf-8')
            post_data = bytes(json.dumps(games), 'utf8')        
            headers = {'Content-Type': 'application/json'}
            req = urllib.request.Request(url, post_data, headers)
            resp = urllib.request.urlopen(req).read().decode('utf-8')
            #resp = requests.post(url, data=json.dumps(games))
            print(resp)
        
        
        
        
    @staticmethod
    def startup(params):
        get_steam_apps()
        # WebsiteStats.run_stats({})
        #RglMain.run_normal_spider({})
        #SephSpider.test()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
