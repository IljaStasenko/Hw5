import requests
from bs4 import BeautifulSoup
import aiogram

def weatherA(city):
        #city = "Минск"
        link = f"https://www.google.com/search?q=погода+в+{city}"
        headers = {
                "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0"
                }

        response = requests.get(link, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        temperature = soup.select("#wob_tm")[0].getText()
        title = soup.select("#wob_dc")[0].getText()
        humidity = soup.select("#wob_hm")[0].getText()
        time = soup.select("#wob_dts")[0].getText()
        wind = soup.select("#wob_ws")[0].getText()

        return f"{time}\n{title}\nТемпература: {temperature}C\nВлажность: {humidity}\nВетер: {wind}"
"""        print(time)
        print(title)
        print(f"Температура: {temperature}C")
        print(f"Влажность: {humidity}")
        print(f"Ветер: {wind}")"""



