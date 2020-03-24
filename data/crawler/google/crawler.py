import openpyxl
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib.request
#filename="wine_list.xlsx"
#wb=openpyxl.load_workbook(filename)
#sheet=wb.worksheets[0]
#header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
wine_name="winebottle"
if not(os.path.isdir(wine_name)):
    os.makedirs(os.path.join(wine_name))
driver = webdriver.Chrome('chromedriver')
driver.implicitly_wait(5)
url = "https://www.google.co.in/search?q="+wine_name+"&source=lnms&tbm=isch"
driver.get(url)
is_btn_clicked = 0
for _ in range(1000):
    driver.execute_script("window.scrollBy(0,10000)")
counter = 0
#succounter = 0
for x in driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
    #counter = counter + 1
    #print ("Total Count:", counter)
    #print ("Succsessful Count:", succounter)
    print (wine_name+"URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
    img =json.loads(x.get_attribute('innerHTML'))["ou"]
    imgtype =json.loads(x.get_attribute('innerHTML'))["ity"]
    img_path = wine_name + "/" + wine_name + "_" + str(counter) + "." + imgtype
    try:
        urllib.request.urlretrieve(img, wine_name + "/" + wine_name + "_google_" + str(counter) + "." + imgtype)
        counter = counter + 1
    except:
        print ("can't get img")       

'''
for row in sheet.rows:
    wine_name=str(row[0].value)
    if not(os.path.isdir(wine_name)):
        os.makedirs(os.path.join(wine_name))
    driver = webdriver.Chrome('chromedriver')
    driver.implicitly_wait(5)
    url = "https://www.google.co.in/search?q="+wine_name+"&source=lnms&tbm=isch"
    driver.get(url)
    is_btn_clicked = 0
    for _ in range(1000):
        driver.execute_script("window.scrollBy(0,10000)")
    counter = 0
    #succounter = 0
    for x in driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
        #counter = counter + 1
        #print ("Total Count:", counter)
        #print ("Succsessful Count:", succounter)
        print (wine_name+"URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
        img =json.loads(x.get_attribute('innerHTML'))["ou"]
        imgtype =json.loads(x.get_attribute('innerHTML'))["ity"]
        img_path = wine_name + "/" + wine_name + "_" + str(counter) + "." + imgtype
        try:
            urllib.request.urlretrieve(img, wine_name + "/" + wine_name + "_google_" + str(counter) + "." + imgtype)
            counter = counter + 1
        except:
            print ("can't get img")               
    driver.close()
'''