import redis
import keyboard
import time
from time import sleep

if __name__ == '__main__':
    r = redis.Redis(host='localhost', port=6379, db=0)
    id = 0
    while not keyboard.is_pressed('q'):
        r.publish('say', str(id))
        id += 1
        sleep(1.0)

    print('[Main Thread]: Finish')