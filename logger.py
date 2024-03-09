def log(text, color):
    text = str(text)
    if color == "red":
        red_text = "\033[91m" + text + "\033[0m"
        print(red_text)
    if color == "green":
        green_text = "\033[92m" + text + "\033[0m"
        print(green_text)
    if color == "yellow":
        yellow_text = "\033[93m" + text + "\033[0m"
        print(yellow_text)
    if color == "blue":
        blue_text = "\033[94m" + text + "\033[0m"
        print(blue_text)