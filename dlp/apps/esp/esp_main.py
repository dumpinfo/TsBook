import time
import ctypes
import win32gui, win32api, win32con
import numpy as np
import cv2
from PIL import ImageGrab
from apps.esp.hdw.c_key_board import CKeyBoard as CKeyBoard
from apps.esp.hdw.c_mouse import CMouse as CMouse

from ann.cnn.slim_inresv2 import SlimInresV2 as SlimInresV2

class EspMain(object):
    @staticmethod
    def click_start_game():
        EspMain.mouse.move_mouse((50, 50))
        time.sleep(1);
        EspMain.mouse.click((50, 50), 'left')
        print('click start game')
        
    @staticmethod
    def click_return_lobby():
        EspMain.mouse.move_mouse((1200, 650))
        time.sleep(1);
        EspMain.mouse.click((1200, 650), 'left')
        print('click return lobby')
        
    @staticmethod
    def click_confirm_return():
        EspMain.mouse.move_mouse((600, 420))
        time.sleep(1);
        EspMain.mouse.click((600, 420), 'left')
        print('click confirm return')
        
    @staticmethod
    def t1(params):
        print('电子竞技平台 v0w.0')
        twin = win32gui.FindWindow(None, 'PLAYERUNKNOWN\'S BATTLEGROUNDS')
        print('twin={0}!'.format(twin))
        time.sleep(5);
        EspMain.mouse = CMouse()
        #EspMain.click_start_game()
        #EspMain.click_return_lobby()
        #EspMain.click_confirm_return()
        EspMain.emulate_kb()
        
    @staticmethod
    def onKeyboardEvent(event):
        if hasattr(event, 'MessageName'):
            print('MessageName:{0}'.format(event.MessageName))
        if hasattr(event, 'Message'):
            print('Message:{0}'.format(event.Message))
        if hasattr(event, 'Time'):
            print('Time:{0}'.format(event.Time))
        if hasattr(event, 'Window'):
            print('Window:{0}'.format(event.Window))
        if hasattr(event, 'WindowName'):
            print('WindowName:{0}'.format(event.WindowName))
        if hasattr(event, 'Ascii'):
            print('Ascii:{0}---{1}'.format(event.Ascii, chr(event.Ascii)))
        if hasattr(event, 'Key'):
            print('Key:{0}'.format(event.Key))
        if hasattr(event, 'KeyID'):
            print('KeyID:{0}'.format(event.KeyID))
        if hasattr(event, 'ScanCode'):
            print('ScanCode:{0}'.format(event.ScanCode))
        if hasattr(event, 'Extended'):
            print('Extended:{0}'.format(event.Extended))
        if hasattr(event, 'Injected'):
            print('Injected:{0}'.format(event.Injected))
        if hasattr(event, 'Alt'):
            print('Alt:{0}'.format(event.Alt))
        if hasattr(event, 'Transition'):
            print('Transition:{0}'.format(event.Transition))
        return True
        
    @staticmethod
    def onMouseEvent(event):
        if hasattr(event, 'MessageName'):
            print('MessageName:{0}'.format(event.MessageName))
        if hasattr(event, 'Message'):
            print('Message:{0}'.format(event.Message))
        if hasattr(event, 'Time'):
            print('Time:{0}'.format(event.Time))
        if hasattr(event, 'Window'):
            print('Window:{0}'.format(event.Window))
        if hasattr(event, 'WindowName'):
            print('WindowName:{0}'.format(event.WindowName))
        if hasattr(event, 'Position'):
            print('Position:{0}'.format(event.Position))
        if hasattr(event, 'Wheel'):
            print('Wheel:{0}'.format(event.Wheel))
        if hasattr(event, 'Injected'):
            print('Injected:{0}'.format(event.Injected))
        return True
    
    @staticmethod
    def esp_test_kb_m():
        print('s1')
        hm = pyHook.HookManager()
        print('s2')
        hm.KeyDown = EspMain.onKeyboardEvent
        print('s3')
        hm.HookKeyboard()
        print('s4')
        hm.MouseAll = EspMain.onMouseEvent
        hm.HookMouse()
        pythoncom.PumpMessages()
        
        
        
    @staticmethod
    def emulate_kb():
        print('press key!!!!!!!!!!')
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        CKeyBoard.press_key(CKeyBoard.DIK_W)
        
    @staticmethod
    def emulate_mouse():
        EspMain.mouse.click((500, 500), 'right')
        
    
    @staticmethod
    def screen_record(): 
        last_time = time.time()
        epoch = 0
        while(True):
            # 800x600 windowed mode
            epoch += 1
            if 50 == epoch:
                EspMain.emulate_kb()
            if 80 == epoch:
                EspMain.emulate_mouse()
            printscreen =  np.array(ImageGrab.grab(bbox=(450,250,1550,830)))
            last_time = time.time()
            new_screen = printscreen # EspMain.process_img(printscreen)
            cv2.imshow('window', cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB))
            mouse_pos = EspMain.mouse.get_position()
            mouse_lbv = EspMain.mouse._get_button_value('left')
            mouse_mbv = EspMain.mouse._get_button_value('middle')
            mouse_rbv = EspMain.mouse._get_button_value('right')
            mouse_bv = (mouse_lbv, mouse_mbv, mouse_rbv)
            kb_val = cv2.waitKey(25) & 0xFF
            #print('loop took {0} seconds; {1}; {2}; {3}'.format(time.time()-last_time, mouse_pos, mouse_bv, kb_val))
            if kb_val == ord('q'):
                cv2.destroyAllWindows()
                break
    
    @staticmethod
    def process_img(image):
        original_image = image
        # convert to gray
        processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # edge detection
        processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
        return processed_img
        
    @staticmethod
    def startup(params):
        inres = SlimInresV2()
        #inres.predict('d:/awork/d3.jpg')
        inres.predict('./work/t001.jpg')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
