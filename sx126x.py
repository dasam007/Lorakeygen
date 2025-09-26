# This file is used for LoRa and Raspberry pi4B related issues 
import RPi.GPIO as GPIO
import serial
import time

class sx126x:
    M0 = 22
    M1 = 27
    cfg_reg = [0xC2,0x00,0x09,0x00,0x00,0x00,0x62,0x00,0x12,0x43,0x00,0x00]
    get_reg = bytes(12)
    rssi = False
    addr = 65535
    serial_n = ""
    addr_temp = 0
    start_freq = 850
    offset_freq = 18
    last_payload = b""
    last_rssi = 0

    # --- UART strategy ---
    # Open at the module's known default first (9600), program new UART in cfg_reg,
    # then reopen at TARGET_UART_BAUD if different.
    BOOT_UART_BAUD   = 9600
    TARGET_UART_BAUD = 115200   # set to 9600 if you want to keep original behavior

    # Module UART baud bitfields for cfg_reg[6]
    SX126X_UART_BAUDRATE_1200 = 0x00
    SX126X_UART_BAUDRATE_2400 = 0x20
    SX126X_UART_BAUDRATE_4800 = 0x40
    SX126X_UART_BAUDRATE_9600 = 0x60
    SX126X_UART_BAUDRATE_19200 = 0x80
    SX126X_UART_BAUDRATE_38400 = 0xA0
    SX126X_UART_BAUDRATE_57600 = 0xC0
    SX126X_UART_BAUDRATE_115200 = 0xE0

    SX126X_PACKAGE_SIZE_240_BYTE = 0x00
    SX126X_PACKAGE_SIZE_128_BYTE = 0x40
    SX126X_PACKAGE_SIZE_64_BYTE  = 0x80
    SX126X_PACKAGE_SIZE_32_BYTE  = 0xC0

    lora_air_speed_dic = {1200:0x01, 2400:0x02, 4800:0x03, 9600:0x04, 19200:0x05, 38400:0x06, 62500:0x07}
    lora_power_dic = {22:0x00, 17:0x01, 13:0x02, 10:0x03}
    lora_buffer_size_dic = {240:SX126X_PACKAGE_SIZE_240_BYTE, 128:SX126X_PACKAGE_SIZE_128_BYTE, 
                            64:SX126X_PACKAGE_SIZE_64_BYTE, 32:SX126X_PACKAGE_SIZE_32_BYTE}

    def __init__(self, serial_num, freq, addr, power, rssi, air_speed=2400, net_id=0, 
                 buffer_size=64, crypt=0, relay=False, lbt=False, wor=False):
        self.rssi = rssi
        self.addr = addr
        self.freq = freq
        self.serial_n = serial_num
        self.power = power
        self.last_payload = b""
        self.last_rssi = 0
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.M0, GPIO.OUT)
        GPIO.setup(self.M1, GPIO.OUT)
        GPIO.output(self.M0, GPIO.LOW)
        GPIO.output(self.M1, GPIO.HIGH)

        # Open at the module's current/known baud FIRST (9600)
        self.ser = serial.Serial(serial_num, self.BOOT_UART_BAUD, timeout=0.1)
        self.ser.flushInput()

        # Program config (this writes TARGET UART into the module)
        ok = self.set(freq, addr, power, rssi, air_speed, net_id, buffer_size, crypt, relay, lbt, wor)

        # If we changed the module to a different UART, reopen our end to match
        if ok and self.TARGET_UART_BAUD != self.BOOT_UART_BAUD:
            try:
                time.sleep(0.1)
                self.ser.close()
                self.ser = serial.Serial(self.serial_n, self.TARGET_UART_BAUD, timeout=0.1)
                self.ser.flushInput()
            except Exception:
                # Fallback: stay at boot baud if reopen fails
                self.ser = serial.Serial(self.serial_n, self.BOOT_UART_BAUD, timeout=0.1)
                self.ser.flushInput()

    def clear_buffer(self):
        while self.ser.inWaiting():
            self.ser.read(self.ser.inWaiting())
            time.sleep(0.01)

    def _uart_sel_from_baud(self, baud):
        return {
            1200:  self.SX126X_UART_BAUDRATE_1200,
            2400:  self.SX126X_UART_BAUDRATE_2400,
            4800:  self.SX126X_UART_BAUDRATE_4800,
            9600:  self.SX126X_UART_BAUDRATE_9600,
            19200: self.SX126X_UART_BAUDRATE_19200,
            38400: self.SX126X_UART_BAUDRATE_38400,
            57600: self.SX126X_UART_BAUDRATE_57600,
            115200:self.SX126X_UART_BAUDRATE_115200,
        }.get(int(baud), self.SX126X_UART_BAUDRATE_9600)

    def set(self, freq, addr, power, rssi, air_speed=2400, net_id=0, buffer_size=64, 
            crypt=0, relay=False, lbt=False, wor=False):
        self.clear_buffer()
        self.send_to = addr
        self.addr = addr
        GPIO.output(self.M0, GPIO.LOW)
        GPIO.output(self.M1, GPIO.HIGH)
        time.sleep(0.2)

        low_addr = addr & 0xff
        high_addr = (addr >> 8) & 0xff
        net_id_temp = net_id & 0xff
        
        if freq > 850:
            freq_temp = freq - 850
            self.start_freq = 850
            self.offset_freq = freq_temp
        elif freq > 410:
            freq_temp = freq - 410
            self.start_freq  = 410
            self.offset_freq = freq_temp
        else:
            freq_temp = self.offset_freq
        
        air_speed_temp   = self.lora_air_speed_dic.get(air_speed, 0x02)  # Default 2400bps
        buffer_size_temp = self.lora_buffer_size_dic.get(buffer_size, self.SX126X_PACKAGE_SIZE_64_BYTE)
        power_temp       = self.lora_power_dic.get(power, 0x00)          # Default 22dBm
        rssi_temp        = 0x80 if rssi else 0x00
        l_crypt          = crypt & 0xff
        h_crypt          = (crypt >> 8) & 0xff

        # IMPORTANT: While sending cfg_reg we are still at BOOT baud.
        # We set the module's future UART to TARGET here:
        uart_sel = self._uart_sel_from_baud(self.TARGET_UART_BAUD)

        if not relay:
            self.cfg_reg[3] = high_addr
            self.cfg_reg[4] = low_addr
            self.cfg_reg[5] = net_id_temp
            self.cfg_reg[6] = uart_sel | air_speed_temp
            self.cfg_reg[7] = buffer_size_temp | power_temp | 0x20
            self.cfg_reg[8] = freq_temp
            self.cfg_reg[9] = 0x43 | rssi_temp
            self.cfg_reg[10] = h_crypt
            self.cfg_reg[11] = l_crypt
        
        self.ser.flushInput()
        success = False
        
        for _ in range(3):
            self.ser.write(bytes(self.cfg_reg))
            time.sleep(0.3)
            if self.ser.inWaiting() >= 12:
                r_buff = self.ser.read(12)
                if len(r_buff) >= 1 and r_buff[0] == 0xC1:
                    success = True
                    break
            time.sleep(0.2)

        GPIO.output(self.M0, GPIO.LOW)
        GPIO.output(self.M1, GPIO.LOW)
        time.sleep(0.2)
        return success

    def send(self, data):
        self.clear_buffer()
        GPIO.output(self.M1, GPIO.LOW)
        GPIO.output(self.M0, GPIO.LOW)
        time.sleep(0.1)
        
        # Split large packets
        max_chunk = 32
        for i in range(0, len(data), max_chunk):
            chunk = data[i:i+max_chunk]
            self.ser.write(chunk)
            time.sleep(0.05)
        time.sleep(0.2)

    # Low-latency receive using idle-gap to detect end of frame (keeps original 3-byte strip)
    def receive(self, timeout=1.0):
        self.last_payload = b""
        self.last_rssi = 0
        start_time = time.time()
        idle_threshold = 0.003  # 3 ms inter-byte idle ⇒ end of frame
        
        while time.time() - start_time < timeout:
            if self.ser.inWaiting() > 0:
                # wait for idle gap rather than fixed 100 ms
                last_n = self.ser.inWaiting()
                last_change = time.time()
                while True:
                    time.sleep(0.001)
                    n2 = self.ser.inWaiting()
                    if n2 != last_n:
                        last_n = n2
                        last_change = time.time()
                    elif (time.time() - last_change) >= idle_threshold:
                        break

                r_buff = self.ser.read(self.ser.inWaiting() or last_n)
                
                if len(r_buff) >= 4:  # Minimum viable packet
                    if self.rssi and len(r_buff) > 3:
                        self.last_rssi = 256 - r_buff[-1]
                        self.last_payload = r_buff[3:-1]  # original 3-byte front strip
                    else:
                        self.last_payload = r_buff[3:]    # original 3-byte front strip
                    return True
            time.sleep(0.005)
        return False

    def get_channel_rssi(self):
        self.clear_buffer()
        GPIO.output(self.M1, GPIO.LOW)
        GPIO.output(self.M0, GPIO.LOW)
        time.sleep(0.1)
        
        self.ser.write(bytes([0xC0, 0xC1, 0xC2, 0xC3, 0x00, 0x02]))
        time.sleep(0.3)
        
        if self.ser.inWaiting() >= 5:
            re_temp = self.ser.read(5)
            if re_temp[0] == 0xC1 and re_temp[1] == 0x00 and re_temp[2] == 0x02:
                return 256 - re_temp[3]
        return -1
