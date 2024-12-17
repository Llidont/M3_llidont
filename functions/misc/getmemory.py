import os
import psutil

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def getmemory(point):
    ''' Devolver por pantalla la memoria ocupada por el proceso
        en un momento concreto '''
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info()
    print(f"{point} están ocupados {sizeof_fmt(memory_info.rss)}")
