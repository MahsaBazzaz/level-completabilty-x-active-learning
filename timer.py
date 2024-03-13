import atexit, subprocess, sys, time, os

def mute_time():
    return os.environ.get('STG_MUTE_TIME')

def mute_port():
    return os.environ.get('STG_MUTE_PORT')

def write_time(ss):
    if not mute_time():
        sys.stdout.write(ss)
        sys.stdout.flush()
        
class SectionTimer:
    def __init__(self):
        self._start_time = time.time()
        self._last_section = None
        self._last_time = None

    def print_done(self):
        if not mute_time():
            print('--TOTALTIME %.2f' % (time.time() - self._start_time))

    def print_section(self, section):
        curr_time = time.time()

        if self._last_section is not None:
            if mute_time():
                print('...%s done' % self._last_section, flush=True)
            else:
                last = '%.2f' % (curr_time - self._last_time)
                total = '%.2f' % (curr_time - self._start_time)
                print('...%s took %s, %s' % (self._last_section, last, total), flush=True)

        self._last_section = section
        self._last_time = curr_time

        if section is not None:
            print('starting %s...' % (section), flush=True)

def _timer_stop():
    global _section_timer
    _section_timer.print_section(None)
    _section_timer.print_done()
    _section_timer = None

def timer_start(print_cmdline=True):
    global _section_timer
    _section_timer = SectionTimer()
    atexit.register(_timer_stop)
    if print_cmdline:
        print('running ' + subprocess.list2cmdline(sys.argv))