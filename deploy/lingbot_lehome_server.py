import argparse
import logging
from typing import Dict

from .lingbot_lehome_policy import LingbotLehomePolicy
from .websocket_policy_server import WebsocketPolicyServer


logger = logging.getLogger(__name__)


class LingbotLehomeServer:
    def __init__(
        self,
        model_path: str,
        task_description: str,
        use_length: int,
        chunk_ret: bool,
        use_bf16: bool,
        use_fp32: bool,
        training_config_path: str | None = None,
    ) -> None:
        self._policy = LingbotLehomePolicy(
            path_to_pi_model=model_path,
            task_description=task_description,
            use_length=use_length,
            chunk_ret=chunk_ret,
            use_bf16=use_bf16,
            use_fp32=use_fp32,
            training_config_path=training_config_path,
        )

    def reset(self, robo_name: str | None = None) -> None:
        self._policy.reset()

    def infer(self, obs: Dict) -> Dict:
        if obs.get("reset"):
            self.reset(obs.get("robo_name"))
            return {"action": None}
        logger.info(f"[DEBUG] obs keys: {list(obs.keys())}")
        for k, v in obs.items():
            if hasattr(v, 'shape'):
                logger.info(f"[DEBUG]   {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                logger.info(f"[DEBUG]   {k}: type={type(v).__name__}, val={repr(v)[:120]}")
        action = self._policy.infer(obs)
        return {"action": action}


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 LingBot-LeHome WebSocket 策略服务")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--training_config_path", type=str, default=None)
    parser.add_argument("--task_description", type=str, default="fold the garment on the table")
    parser.add_argument("--use_length", type=int, default=1)
    parser.add_argument("--chunk_ret", action="store_true")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    model = LingbotLehomeServer(
        model_path=args.model_path,
        training_config_path=args.training_config_path,
        task_description=args.task_description,
        use_length=args.use_length,
        chunk_ret=args.chunk_ret,
        use_bf16=args.use_bf16,
        use_fp32=args.use_fp32,
    )
    metadata = {
        "policy": "lingbot_lehome",
        "task_description": args.task_description,
        "use_length": args.use_length,
        "chunk_ret": args.chunk_ret,
    }
    model_server = WebsocketPolicyServer(model, host=args.host, port=args.port, metadata=metadata)
    model_server.serve_forever()


if __name__ == "__main__":
    main()
