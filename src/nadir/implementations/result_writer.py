# src/nadir/implementations/result_writer.py
import json
from pathlib import Path
from typing import Dict
from ..interfaces import IResultWriter


class JSONResultWriter(IResultWriter):
    """JSON 형식으로 결과를 저장하는 writer"""

    def save_results(self, results: Dict, output_path: str) -> None:
        """처리 결과를 JSON 파일로 저장"""
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(
                    results,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=lambda o: o.model_dump()
                    if hasattr(o, "model_dump")
                    else str(o),
                )

        except Exception as e:
            self.logger.error(f"결과 저장 실패: {str(e)}")
            raise
