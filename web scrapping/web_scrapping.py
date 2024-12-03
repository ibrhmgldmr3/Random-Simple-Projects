from bs4 import BeautifulSoup
import requests
import time

def find_borsa(link:str):
    html_text = requests.get(link)
    soup=BeautifulSoup(html_text.text,'lxml')
    #print(soup.prettify())
    things = soup.find_all('ol')
    print(things)
    print("-------------------------------------------------------")
    for thing in things:
        title=thing.find('li').text
        with open('borsa.txt', 'a', encoding='utf-8') as f:
            f.write(f"soz: {title}\n")
        print(f''' 
            {title} 
            ''')
    print("Completed writing to file")

find_borsa("https://ikas.com/tr/blog/ilham-veren-motivasyon-sozleri")

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# WebDriver'ı başlatın
driver = webdriver.Chrome()  # veya webdriver.Firefox()
driver.get('https://web.whatsapp.com')

# QR kodunu taramak için kullanıcıya zaman tanıyın
input("QR kodunu taradıktan sonra Enter'a basın")

def send_whatsapp_message(contact, message):
    try:
        # Kişiyi arayın
        search_box = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, '//div[@contenteditable="true"][@data-tab="3"]'))
        )
        search_box.click()
        search_box.send_keys(contact)
        search_box.send_keys(Keys.ENTER)
        time.sleep(2)

        # Mesajı gönderin
        while True:
            message_box = driver.find_element(By.XPATH, '//div[@contenteditable="true" and @data-tab="10"]')
            message_box.click()
            message_box.send_keys(message)
            message_box.send_keys(Keys.ENTER)
            print(f"Message sent to {contact}")
            time.sleep(5)
        
    except Exception as e:
        print(f"Failed to send message to {contact}: {e}")

# Mesaj gönderme
send_whatsapp_message("Kişi Adı", "Merhaba!....")

# Tarayıcıyı kapat
driver.quit()

# if __name__=='__main__':
#     while True:
#         find_borsa('https://www.robotistan.com/')
#         time.wait=10
#         print(f'Waiting {time.wait} seconds...')
#         time.sleep(time.wait)

# with open('home.html', 'r') as html_file:
#     content = html_file.read()
#     soup = BeautifulSoup(content, 'lxml')
#     # print(soup.prettify())
#     tags = soup.find_all('h5')
#     courses = []
#     for tag in tags:
#         courses.append(tag.text)
#     print(courses)
