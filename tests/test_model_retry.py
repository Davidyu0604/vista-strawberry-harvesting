
# Allow this development script to run from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'src'))
from project_paths import PROJECT_ROOT, project_source

import json
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from centered_grasp import TrackingError
from vista import Client, ModelServiceError, MotionSafetyStop, Session


class ImmediatePool:
    def __init__(self, **kwargs): pass

    def submit(self, fn, *args):
        future = Future()
        try: future.set_result(fn(*args))
        except Exception as exc: future.set_exception(exc)
        return future

    def shutdown(self, **kwargs): pass


class ModelRetryTests(unittest.TestCase):
    def session(self, outcomes, budget=32):
        s = Session.__new__(Session)
        s.e = SimpleNamespace(last_action=[0] * 6, drive=Mock(), frame=0)
        s.max_calls = budget; s.max_wait = 90; s.wait_ticks = 0
        s.messages = [{'role': 'system', 'content': 'test'}]
        s.log = Mock(); s.tick = Mock(); s.finished = False
        frames = iter(range(100))
        def observe():
            frame = f'f{next(frames)}'
            return frame, {}, {'role': 'user', 'content': [{'type': 'text', 'text': frame}]}
        s.observe = Mock(side_effect=observe)
        s.client = SimpleNamespace(call=Mock(side_effect=outcomes))
        def dispatch(name, *args):
            s.finished = name == 'finish'
            return {'status': 'executed'}, []
        s.dispatch = Mock(side_effect=dispatch)
        return s

    def response(self, name='finish'):
        return {'role': 'assistant', 'tool_calls': [
            {'id': 'one', 'function': {'name': name, 'arguments': '{}'}}]}, {}

    def run_session(self, s):
        clock = iter(range(10000))
        with patch('vista.ThreadPoolExecutor', ImmediatePool), patch('vista.time.monotonic', side_effect=lambda: next(clock)):
            s.run()

    def failure(self):
        return ModelServiceError('Model HTTP 500; response body omitted', retryable=True)

    def test_http_classification_and_sanitization(self):
        for status in (400, 401, 403, 408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                c = Client.__new__(Client)
                c.endpoint = 'https://example.test/chat/completions'
                c.headers = {}; c.model = 'test'; c.options = {}
                with httpx.Client(transport=httpx.MockTransport(
                        lambda request: httpx.Response(status, text='SECRET response'))) as c.http:
                    with self.assertRaises(ModelServiceError) as error: c.call([])
                self.assertEqual(error.exception.retryable, status in (408, 429, 500, 502, 503, 504))
                self.assertNotIn('SECRET', str(error.exception))

    def test_timeout_is_sanitized_and_retryable(self):
        c = Client.__new__(Client)
        c.endpoint = 'https://example.test'; c.headers = {}; c.model = 'test'; c.options = {}
        c.http = Mock(); c.http.post.side_effect = httpx.ReadTimeout('SECRET')
        with self.assertRaises(ModelServiceError) as error: c.call([])
        self.assertTrue(error.exception.retryable)
        self.assertNotIn('SECRET', str(error.exception))

    def test_retry_refreshes_observation_and_counts_budget(self):
        s = self.session([self.failure(), self.response()])
        self.run_session(s)
        self.assertEqual(s.calls, 2)
        s.dispatch.assert_called_once()
        self.assertEqual(s.dispatch.call_args.args[2], 'f1')
        messages = s.client.call.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]['content'][0]['text'], 'f1')
        budget = json.loads(messages[-1]['content'][-1]['text'])
        self.assertEqual(budget['remaining_calls_after_this_response'], 30)
        self.assertTrue(any(call.args[1] == 'model_wait' for call in s.tick.call_args_list))

    def test_three_failures_stop_without_dispatch(self):
        s = self.session([self.failure() for _ in range(4)])
        with self.assertRaises(ModelServiceError): self.run_session(s)
        self.assertEqual(s.client.call.call_count, 3)
        self.assertEqual([r['retry_delay_s'] for r in s.model_service_failures], [2, 4, 0])
        s.dispatch.assert_not_called()

    def test_budget_prevents_extra_request(self):
        s = self.session([self.failure(), self.failure(), self.response()], budget=2)
        with self.assertRaises(ModelServiceError): self.run_session(s)
        self.assertEqual(s.client.call.call_count, 2)
        self.assertFalse(s.model_service_failures[-1]['retry_scheduled'])
        s.dispatch.assert_not_called()

    def test_nonretryable_failure_stops_immediately(self):
        s = self.session([ModelServiceError('Model HTTP 401'), self.response()])
        with self.assertRaises(ModelServiceError): self.run_session(s)
        s.client.call.assert_called_once(); s.dispatch.assert_not_called()

    def test_success_resets_consecutive_failure_count(self):
        s = self.session([self.failure(), self.failure(), self.response('inspect'),
                          self.failure(), self.failure(), self.response()])
        self.run_session(s)
        self.assertEqual(s.calls, 6)
        self.assertEqual(s.dispatch.call_count, 2)

    def test_tracking_protection_remains_active_during_backoff(self):
        s = self.session([self.failure(), self.response()])
        def tick(action, phase, **kwargs):
            if phase == 'model_wait':
                exc = TrackingError('tracking during retry hold')
                exc.phase = phase; exc.tracking = {'error_m': .009}
                raise exc
        s.tick.side_effect = tick
        with self.assertRaises(MotionSafetyStop): self.run_session(s)
        s.client.call.assert_called_once(); s.dispatch.assert_not_called()
        self.assertEqual(s.motion_stop_trace['phase'], 'model_wait')


if __name__ == '__main__': unittest.main()
