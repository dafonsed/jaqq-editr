"""Real loopback HTTP lifecycle checks without AI models or Resolve."""
import json
import subprocess
import sys
import threading
import time
import unittest
import urllib.request

from automatic_cut import Cancelled
from local_editor import Editor


SERVER = '''
import http.server, json, sys, time
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(float(sys.argv[1]))
        payload=json.dumps({'complete': True}).encode()
        self.send_response(200)
        self.send_header('Content-Length',str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    def log_message(self,*args):pass
server=http.server.HTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True)
server.serve_forever()
'''


class RequestLifecycleTests(unittest.TestCase):
    def start_editor(self, delay, cancel=lambda: False):
        editor = Editor(cancel)
        editor.process = subprocess.Popen([sys.executable, '-u', '-c', SERVER, str(delay)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.addCleanup(editor.__exit__, None, None, None)
        self.addCleanup(editor.process.stdout.close)
        port = int(editor.process.stdout.readline().strip())
        return editor, urllib.request.Request(f'http://127.0.0.1:{port}/')

    def assert_reader_stopped(self):
        self.assertFalse([thread for thread in threading.enumerate()
                          if thread.name == 'LocalEditorResponse'])

    def test_slow_response_completes_and_reader_is_joined(self):
        editor, request = self.start_editor(.35)
        editor.request_timeout = 2
        self.assertEqual(editor._request_json(request), {'complete': True})
        self.assert_reader_stopped()
        editor.__exit__(None, None, None)
        self.assertIsNotNone(editor.process.poll())

    def test_cancel_interrupts_pending_headers_and_stops_owned_engine(self):
        cancelled = threading.Event()
        editor, request = self.start_editor(3600, cancelled.is_set)
        timer = threading.Timer(.2, cancelled.set)
        timer.start()
        self.addCleanup(timer.join)
        started = time.monotonic()
        with self.assertRaises(Cancelled):
            editor._request_json(request)
        self.assertLess(time.monotonic()-started, 2)
        self.assertIsNotNone(editor.process.poll())
        self.assert_reader_stopped()

    def test_timeout_is_bounded_and_engine_can_be_cleaned_up(self):
        editor, request = self.start_editor(3600)
        editor.request_timeout = .3
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, 'took too long'):
            editor._request_json(request)
        editor.__exit__(None, None, None)
        self.assertLess(time.monotonic()-started, 2)
        self.assertIsNotNone(editor.process.poll())
        self.assert_reader_stopped()


if __name__ == '__main__':
    unittest.main()
