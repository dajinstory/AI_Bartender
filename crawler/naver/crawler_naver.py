import urllib.request
from bs4 import BeautifulSoup
import openpyxl
import os
from selenium import webdriver
'''
#filename="wine_list.xlsx"
#wb=openpyxl.load_workbook(filename)
#sheet=wb.worksheets[0]
#for# row in sheet.rows:
    wine_name=str(row[0].value)
    if not(os.path.isdir(wine_name)):
        os.makedirs(os.path.join(wine_name))
    driver = webdriver.Chrome('chromedriver')
    driver.implicitly_wait(5)
    url = str("https://search.naver.com/search.naver?where=image&sm=tab_jum&query=")+wine_name
    driver.get(url)
    for _ in range(1000):
        driver.execute_script("window.scrollBy(0,10000)")
    counter = 0
    for x in (driver.find_elements_by_xpath("//img")):
        #time.sleep(1)
        #counter = counter + 1
        #x.screenshot(wine_name + "/" + wine_name + "_naver_" + str(counter) + ".png")
        #print (wine_name+"URL:",json.loads(x.get_attribute('innerHTML'))["originalurl"])
        #print ((el.get_attribute('innerHTML'))["src"])
        try:
            soup= x.get_attribute('outerHTML')
            if soup.find("http")!=-1:
                soup=soup[soup.find("http"):soup.find("\"",soup.find("http"))]
                print ("success :" + soup)
                urllib.request.urlretrieve(soup, wine_name + "/" + wine_name + "_naver_" + str(counter) + ".jpg")
                counter = counter + 1
            #print (wine_name+"URL:",json.loads(x.get_attribute('outerHTML'))["src"])
            #img =json.loads(x.get_attribute('outerHTML'))["src"]
            #print (img)
        #imgtype =json.loads(x.get_attribute('innerHTML'))["ity"]   
        #img_path = wine_name + "/" + wine_name + "_" + str(counter) + ".jpg" 
            #try:

                #urllib.request.urlretrieve(img, wine_name + "/" + wine_name + "_naver_" + str(counter) + ".jpg")
                #counter = counter + 1
            #except:
                #print ("can't get img")
        except Exception as e:
            print(e)
    driver.close()
'''
wine_name="winebottle"
if not(os.path.isdir(wine_name)):
    os.makedirs(os.path.join(wine_name))
driver = webdriver.Chrome('chromedriver')
driver.implicitly_wait(5)
url = str("https://search.naver.com/search.naver?where=image&sm=tab_jum&query=")+wine_name
driver.get(url)
for _ in range(1000):
    driver.execute_script("window.scrollBy(0,10000)")
counter = 0
for x in (driver.find_elements_by_xpath("//img")):
        #time.sleep(1)
        #counter = counter + 1
        #x.screenshot(wine_name + "/" + wine_name + "_naver_" + str(counter) + ".png")
        #print (wine_name+"URL:",json.loads(x.get_attribute('innerHTML'))["originalurl"])
        #print ((el.get_attribute('innerHTML'))["src"])
    try:
        soup= x.get_attribute('outerHTML')
        if soup.find("http")!=-1:
            soup=soup[soup.find("http"):soup.find("\"",soup.find("http"))]
            print ("success :" + soup)
            urllib.request.urlretrieve(soup, wine_name + "/" + wine_name + "_naver_" + str(counter) + ".jpg")
            counter = counter + 1
            #print (wine_name+"URL:",json.loads(x.get_attribute('outerHTML'))["src"])
            #img =json.loads(x.get_attribute('outerHTML'))["src"]
            #print (img)
        #imgtype =json.loads(x.get_attribute('innerHTML'))["ity"]   
        #img_path = wine_name + "/" + wine_name + "_" + str(counter) + ".jpg" 
            #try:

                #urllib.request.urlretrieve(img, wine_name + "/" + wine_name + "_naver_" + str(counter) + ".jpg")
                #counter = counter + 1
            #except:
                #print ("can't get img")
    except Exception as e:
        print(e)
driver.close()

