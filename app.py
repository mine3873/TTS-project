import sys
import atexit
from PyQt5.QtWidgets import QMainWindow, QApplication, QTextEdit, QLineEdit, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
from PyQt5.QtCore import QTimer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from selenium.webdriver.chrome.options import Options
import subprocess
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import os
import torch
import torchaudio
from TTS.TTS.tts.configs.xtts_config import XttsConfig
from TTS.TTS.tts.models.xtts import Xtts
import sounddevice as sd


CONFIG_PATH = "TTS/voice/tts/custum_ja/config.json"
TOKENIZER_PATH = "TTS/voice/tts/XTTS_v2.0_original_model_files/vocab.json"
XTTS_CHECKPOINT = "TTS/voice/tts/custum_ja/checkpoint_9300.pth"
SPEAKER_REFERENCE = "TTS/voice/tts/custum_ja/voiceFile0.wav"

def play_kayoko_TTS(text):
    out = model.inference(
        text,
        "ja",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7, # Add custom parameters here
    )
    
    audio_data = torch.tensor(out["wav"]).numpy()
    
    volume_factor = 0.3  # For example, reduce volume to 50%
    audio_data = audio_data * volume_factor
    sd.play(audio_data, samplerate=24000)

# TTS 
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])



subprocess.Popen(r'C:\Program Files\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\chromeCookie"')
option = Options()
option.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=option)
    # 웹사이트 접속
driver.get('https://chatgpt.com/g/g-1aLqNB1zW-kayoko')



chat_button_xpath = '//button[@class="mb-1 me-1 flex h-8 w-8 items-center justify-center rounded-full bg-black text-white transition-colors hover:opacity-70 focus-visible:outline-none focus-visible:outline-black disabled:bg-[#D7D7D7] disabled:text-[#f4f4f4] disabled:hover:opacity-100 dark:bg-white dark:text-black dark:focus-visible:outline-white disabled:dark:bg-token-text-quaternary dark:disabled:text-token-main-surface-secondary"]'

def addOneLenDiv():
    lendiv = lendiv + 1

def input_Text(text):
    inputarea = driver.find_element(By.ID, "prompt-textarea")
    inputarea.send_keys(text)
    driver.find_element("xpath",chat_button_xpath).click()

def get_latest_answer():
    selector = ".markdown.prose.w-full.break-words.dark\:prose-invert.dark"
    current_count = len(driver.find_elements(By.CSS_SELECTOR, selector))
    
    WebDriverWait(driver, 20).until(
        lambda x: len(x.find_elements(By.CSS_SELECTOR, selector)) > current_count
    )
    #time.sleep(4)
    
    button = driver.find_element("xpath",chat_button_xpath)
    while(button.is_enabled()):
        pass
    latest_answer_div = driver.find_elements(By.CSS_SELECTOR, selector)[-1]
    p_text = latest_answer_div.find_elements(By.TAG_NAME, "p")
    return p_text
        
    

def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class ChatUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set the background image
        self.initUI()

    def initUI(self):
        # Window settings
        self.setGeometry(100, 100, 1280, 960)
        self.setWindowTitle('Chat UI Example')
        QFontDatabase.addApplicationFont('./GodoB.ttf')
        
        # Set the background
        self.backgorund_label = QLabel(self)
        background_image_path = resource_path('img/kayoko.jpg')
        self.backgorund_label.setPixmap(QPixmap(background_image_path))
        self.backgorund_label.setScaledContents(True)
        self.backgorund_label.setFixedSize(1280,960)
        
        # Create the name area
        self.nameArea = QLabel("카요코",self)
        self.nameArea.setFixedSize(1000,70)
        self.nameArea.setStyleSheet(
            "background-color: rgba(255,255,255,0);"
            "border-width: 0px 0px 3px 0px;"
            "border-style: solid;"
            "border-color: rgba(138,138,138,200);"
            "color: white;"
        )
        self.nameArea.setFont(QFont('고도 B',40))
        self.nameArea.move(140,575)
        
        # Create the explaination area
        self.explain = QTextEdit(self)
        self.explain.setReadOnly(True)
        self.explain.append("흥신소 68")
        self.explain.setFixedSize(500,60)
        self.explain.move(320,590)
        self.explain.setStyleSheet(
            "background-color: rgba(255,255,255,0);"
            "border-style: solid;"
            "border-color: rgba(255,255,255,0);"
            "color: rgba(74,162,188,200);"
        )
        self.explain.setFont(QFont('고도 B',25))
        
        # Create the answer area
        self.answerArea = QTextEdit(self)
        self.answerArea.setReadOnly(True)
        
        self.answerArea.setLineWrapMode(1)
        self.answerArea.setLineWrapColumnOrWidth(1)
        
        self.answerArea.setStyleSheet(  
            "background-color: rgba(255,255,255,0);"
            "border-color: rgba(255,255,255,0);"
            "border-style: solid;"
            "font-weight: bold;"
            "color: white;"
                                      )
        self.answerArea.setFixedSize(1000,220)
        self.answerArea.move(140,660)
        self.answerArea.setFont(QFont('고도 B',25))
        
        # Create the chat display area
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "background-color: qLineargradient(y1: 0, y2: 1, stop: 0 rgba(20, 27, 59, 0), stop: 0.1 rgba(20, 27, 59, 160));"
            "border-color: rgba(255,255,255,0);"
            "border-style: solid;"
            "font-weight: bold;"
            "font-size: 20px;"
            "color: rgba(255,255,255,200);"
        )
        self.chat_display.setFixedSize(1280,450)
        self.chat_display.move(0,510)
        
        
        
        
        # Create the input text box
        self.text_input = QLineEdit(self)
        self.text_input.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200);"
            "border-color: rgba(255,255,255,0);"
            "border-radius: 20px;"
        ) 
        self.text_input.returnPressed.connect(self.on_enter)
        self.text_input.setFixedSize(1000,60)
        self.text_input.move(140,880)
        self.text_input.setFont(QFont('고도 B', 25))
        self.text_input.setTextMargins(20,0,0,0)
        
        layout = QVBoxLayout()
        layout.addWidget(self.nameArea)
        layout.addWidget(self.text_input)
        layout.addWidget(self.chat_display)
        layout.addWidget(self.answerArea)
        layout.addWidget(self.explain)
        
        self.chat_display.lower()
        self.backgorund_label.lower()
        
        self.setLayout(layout)
        self.show()

    
    def show_Answer(self):
        answer_arr = get_latest_answer()
        answer = answer_arr[0].text
        ja_answer = answer_arr[1].text
        play_kayoko_TTS(ja_answer)
        #answer = get_latest_answer()
        self.answerArea.setText('')
        self.text = answer
        self.idx = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.typeLetter)
        self.timer.start(80)
        
        self.text_input.clear()
        
    def typeLetter(self):
        if self.idx < len(self.text):
            self.answerArea.insertPlainText(self.text[self.idx])
            self.idx += 1
        else:
            self.timer.stop()
        
    def on_enter(self):
        # Get text from input box and display it in the chat display area
        input_textAPP = self.text_input.text()
        input_Text(input_textAPP)
        self.show_Answer()

  # 상태 확인 간격
    def closeEvent(self, event):
        # 윈도우가 닫힐 때 웹드라이버 종료
        driver.quit()
        event.accept()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChatUI()
    sys.exit(app.exec_())
    
# Register the driver.quit function to be called on program exit
atexit.register(driver.quit)