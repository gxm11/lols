from time import strftime

def info(message):    
    print("[%s] %s" % (strftime("%x %X"), message), flush=True)

def title(message):
    print("[%s]=== %s ===" % (strftime("%x %X"), message.center(30)), flush=True)

def error(message):
    print("[%s] [ERROR!] %s" % (strftime("%x %X"), message), flush=True)