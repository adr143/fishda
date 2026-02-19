from RPLCD.i2c import CharLCD
from time import sleep

# Change address if needed (0x27 or 0x3F usually)
lcd = CharLCD(
    i2c_expander='PCF8574',
    address=0x27,
    port=1,
    cols=16,   # change to 20 if using 20x4 LCD
    rows=2,    # change to 4 if using 20x4 LCD
    dotsize=8,
    charmap='A02'
)

try:
    lcd.clear()
    lcd.write_string("I2C LCD Test")
    sleep(2)

    lcd.clear()
    lcd.write_string("Line 1 OK")
    lcd.cursor_pos = (1, 0)
    lcd.write_string("Line 2 OK")
    sleep(3)

    lcd.clear()

    # simple counter test
    for i in range(10):
        lcd.cursor_pos = (0, 0)
        lcd.write_string(f"Count: {i:02d}")
        sleep(1)

    lcd.clear()
    lcd.write_string("Test Done")

except KeyboardInterrupt:
    pass

finally:
    lcd.clear()
