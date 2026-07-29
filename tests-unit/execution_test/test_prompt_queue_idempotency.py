from types import SimpleNamespace
import unittest

from execution import PromptQueue


def _item(number: int, prompt_id: str):
    return (number, prompt_id, {}, {}, [], {})


class PromptQueueIdempotencyTests(unittest.TestCase):
    def test_prompt_ids_are_unique_across_queue_running_and_history(self):
        server = SimpleNamespace(queue_updated=lambda: None)
        queue = PromptQueue(server)

        self.assertTrue(queue.put_if_prompt_absent(_item(1, "prompt-a")))
        self.assertFalse(queue.put_if_prompt_absent(_item(2, "prompt-a")))

        running, task_id = queue.get()
        self.assertEqual(running[1], "prompt-a")
        self.assertFalse(queue.put_if_prompt_absent(_item(3, "prompt-a")))

        queue.task_done(task_id, {}, None)
        self.assertFalse(queue.put_if_prompt_absent(_item(4, "prompt-a")))
        self.assertTrue(queue.put_if_prompt_absent(_item(5, "prompt-b")))


if __name__ == "__main__":
    unittest.main()
