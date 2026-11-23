from pathlib import Path
from io import BytesIO
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

SOURCE = Path("data/parquet/ar25_experiment_qa_corrected.parquet")
OUTPUT = Path("data/parquet/ar25_experiment_qa_submission_embedded.parquet")
REPORT = Path("reports/image_embedding_report.json")

# Deterministic categorical visualization for the grid's observed values.
PALETTE = [
    (0, 0, 0), (0, 92, 230), (220, 35, 35), (30, 160, 65),
    (245, 210, 30), (130, 130, 130), (190, 50, 190),
    (245, 130, 25), (40, 190, 210), (120, 70, 35),
    (245, 245, 245), (70, 70, 150), (40, 120, 80),
    (180, 100, 50), (100, 180, 180), (220, 100, 120),
]

def encode_frame(path_string):
    path = Path(path_string)
    if not path.is_file():
        raise FileNotFoundError(f"Missing source frame: {path}")

    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D frame: {path}, got {array.shape}")

    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"Non-integer grid values in {path}")

    if array.min() < 0 or array.max() >= len(PALETTE):
        raise ValueError(f"Grid values outside palette range in {path}")

    # Preserve the original numeric color IDs consistently across all frames.
    rgb = np.asarray(PALETTE, dtype=np.uint8)[array.astype(np.int64)]

    image = Image.fromarray(rgb, mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    png_bytes = buffer.getvalue()
    npy_bytes = path.read_bytes()

    # Verify the embedded PNG is readable.
    with Image.open(BytesIO(png_bytes)) as check:
        check.verify()

    return png_bytes, npy_bytes, array.shape

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    table = pq.read_table(SOURCE)
    rows = table.to_pylist()

    initial_png, final_png = [], []
    initial_npy, final_npy = [], []
    image_shapes = set()
    image_hashes = []

    for index, row in enumerate(rows):
        ipng, inpy, ishape = encode_frame(row["initial_frame_path"])
        fpng, fnpy, fshape = encode_frame(row["final_frame_path"])

        expected_shape = row["frame_shape"]
        expected = tuple(map(int, expected_shape.split("x")))
        if ishape != expected or fshape != expected:
            raise ValueError(
                f"Frame shape mismatch at row {index}: "
                f"expected {expected}, got {ishape}/{fshape}"
            )

        initial_png.append(ipng)
        final_png.append(fpng)
        initial_npy.append(inpy)
        final_npy.append(fnpy)
        image_shapes.add(ishape)

        image_hashes.append({
            "record_id": row["record_id"],
            "initial_png_sha256": hashlib.sha256(ipng).hexdigest(),
            "final_png_sha256": hashlib.sha256(fpng).hexdigest(),
        })

    additions = [
        pa.array(initial_png, type=pa.binary()),
        pa.array(final_png, type=pa.binary()),
        pa.array(initial_npy, type=pa.binary()),
        pa.array(final_npy, type=pa.binary()),
    ]

    result = table.append_column("initial_image_png", additions[0])
    result = result.append_column("final_image_png", additions[1])
    result = result.append_column("initial_frame_npy", additions[2])
    result = result.append_column("final_frame_npy", additions[3])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(result, OUTPUT, compression="zstd")

    # Reopen the written artifact and verify every embedded image.
    check_table = pq.read_table(OUTPUT)
    if check_table.num_rows != table.num_rows:
        raise AssertionError("Output row count changed")

    for name in (
        "initial_image_png", "final_image_png",
        "initial_frame_npy", "final_frame_npy",
    ):
        if check_table[name].null_count:
            raise AssertionError(f"Null values found in {name}")

    for index in range(check_table.num_rows):
        for column in ("initial_image_png", "final_image_png"):
            with Image.open(BytesIO(check_table[column][index].as_py())) as image:
                image.verify()

    report = {
        "status": "passed",
        "source_parquet": str(SOURCE),
        "output_parquet": str(OUTPUT),
        "rows": check_table.num_rows,
        "question_answer_pairs": check_table.num_rows,
        "embedded_initial_png_images": len(initial_png),
        "embedded_final_png_images": len(final_png),
        "embedded_original_initial_npy_frames": len(initial_npy),
        "embedded_original_final_npy_frames": len(final_npy),
        "frame_shapes": [f"{h}x{w}" for h, w in sorted(image_shapes)],
        "binary_columns": [
            "initial_image_png", "final_image_png",
            "initial_frame_npy", "final_frame_npy",
        ],
        "image_hashes": image_hashes,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("IMAGE EMBEDDING: PASSED")
    print(f"Rows: {check_table.num_rows}")
    print(f"Initial PNG images: {len(initial_png)}")
    print(f"Final PNG images: {len(final_png)}")
    print(f"Original NPY frames embedded: {len(initial_npy) + len(final_npy)}")
    print(f"Output: {OUTPUT}")
    print(f"Report: {REPORT}")

if __name__ == "__main__":
    main()
