from __future__ import annotations

import time
import unittest
from concurrent import futures

import grpc
import numpy as np

from server import policy_pb2, policy_pb2_grpc
from tools.local import piper_remote_runtime as runtime


class FakePolicyService(policy_pb2_grpc.PolicyServiceServicer):
    def GetServerInfo(self, request, context):
        return policy_pb2.ServerInfoResponse(
            protocol_version=runtime.PROTOCOL_VERSION,
            model_version="fake-dp3",
            output_dir="/workspace/fake",
            policy_subdir="bc",
            n_obs_steps=3,
            n_action_steps=4,
            image_shape=runtime.IMAGE_SHAPE,
            point_cloud_shape=runtime.POINT_CLOUD_SHAPE,
            agent_pos_shape=runtime.AGENT_POS_SHAPE,
            action_shape=runtime.ACTION_SHAPE,
            state_min=[-1.0] * 6 + [0.0],
            state_max=[1.0] * 6 + [0.07],
            action_min=[-1.0] * 6 + [0.0],
            action_max=[1.0] * 6 + [1.0],
            point_cloud_low=[-2.0, -2.0, 0.1],
            point_cloud_high=[2.0, 2.0, 2.0],
            gripper_action_mode="command",
            action_key="policy_action",
        )

    def Infer(self, request, context):
        chunk = np.zeros((4, 7), dtype="<f4")
        chunk[:, 6] = 1.0
        return policy_pb2.InferenceResponse(
            protocol_version=runtime.PROTOCOL_VERSION,
            episode_id=request.episode_id,
            sequence_id=request.sequence_id,
            capture_timestamp_ns=request.capture_timestamp_ns,
            action_chunk_f32=chunk.tobytes(),
            action_steps=4,
            action_dims=7,
            inference_time_ms=1.0,
            model_version="fake-dp3",
            ready=True,
        )

    def ResetEpisode(self, request, context):
        return policy_pb2.ResetEpisodeResponse(
            protocol_version=runtime.PROTOCOL_VERSION,
            episode_id=request.episode_id,
            reset=True,
        )


class RemoteRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        policy_pb2_grpc.add_PolicyServiceServicer_to_server(
            FakePolicyService(), cls.server
        )
        port = cls.server.add_insecure_port("127.0.0.1:0")
        cls.server.start()
        cls.client = runtime.PolicyClient(f"127.0.0.1:{port}", 2.0)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.server.stop(0).wait()

    def test_contract_and_inference(self):
        contract = self.client.get_contract(1.0)
        self.assertEqual(contract.n_obs_steps, 3)
        self.assertEqual(contract.n_action_steps, 4)
        contract.validate_model_identity("fake", "bc")
        self.client.reset_episode("test", 1.0)
        request = runtime.make_request(
            "test",
            7,
            123,
            np.zeros(7, dtype=np.float32),
            np.zeros((3, 84, 84), dtype=np.float32),
            np.zeros((512, 3), dtype=np.float32),
        )
        response = self.client.infer(request, 1.0)
        chunk = runtime.decode_f32(
            response.action_chunk_f32, (4, 7), "action_chunk"
        )
        self.assertEqual(chunk.shape, (4, 7))
        self.assertTrue(np.all(chunk[:, 6] == 1.0))

    def test_async_client(self):
        worker = runtime.AsyncPolicyClient(self.client, 1.0, 4)
        worker.start()
        try:
            worker.submit(
                runtime.make_request(
                    "async",
                    1,
                    456,
                    np.zeros(7, dtype=np.float32),
                    np.zeros((3, 84, 84), dtype=np.float32),
                    np.zeros((512, 3), dtype=np.float32),
                )
            )
            deadline = time.monotonic() + 2.0
            result = None
            while result is None and time.monotonic() < deadline:
                result, error = worker.snapshot()
                self.assertIsNone(error)
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertEqual(result.sequence_id, 1)
            self.assertEqual(result.action_chunk.shape, (4, 7))
        finally:
            worker.stop()


if __name__ == "__main__":
    unittest.main()
