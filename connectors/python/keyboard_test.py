from pynput.keyboard import Key, KeyCode, Listener
import keyboard

# def on_press(key):
#     pass
#
# def on_release(key):
#     print('{0} release'.format(
#         key))
#
#     if key == KeyCode.from_char("q"):
#         print("released q")
#
#
#     if key == Key.esc:
#         # Stop listener
#         return False
#
# # Collect events until released
# with Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()

while True:
    input = keyboard.read_hotkey(suppress=False)
    print("you jus pressed {}".format(input))

    # keyboard.unhook_all_hotkeys()

    if input == "q":
        print("released q")
        break
    elif input == "s":
        print("released s")
    elif input == "p":
        print("released p")