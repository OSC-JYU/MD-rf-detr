import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union
from PIL import Image, ImageDraw

from .utils import load_image_paths
from .seg_inference import load_rfdetr_model, predict_polygons
from .reading_order import GraphBasedOrdering

MODEL_PATH = "./models/rfdetr_text_seg_model.pth"   


def _polygon_to_list(polygon: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Convert a polygon represented as a numpy array into a JSON-serializable list.
    """
    return [[int(x), int(y)] for x, y in polygon]


def _serialize_polygons(polygons: Iterable[Sequence[Sequence[int]]]) -> List[List[List[int]]]:
    """
    Convert an iterable of polygons into JSON-serializable nested lists.
    """
    return [_polygon_to_list(polygon) for polygon in polygons]


def _order_detections(
    polygons: Sequence[Sequence[Sequence[int]]],
    confidences: Sequence[float],
    boxes: Sequence[Sequence[float]],
) -> Tuple[
    Sequence[Sequence[Sequence[int]]],
    Sequence[float],
    Sequence[Sequence[float]],
]:
    if not boxes:
        return polygons, confidences, boxes

    orderer = GraphBasedOrdering()
    indices = orderer.order(boxes)
    if not indices:
        return polygons, confidences, boxes

    return (
        [polygons[i] for i in indices],
        [confidences[i] for i in indices],
        [boxes[i] for i in indices],
    )


def _build_output(
    image_path: str,
    line_polygons,
    line_confs,
    line_boxes,
    region_polygons,
    region_confs,
    region_boxes,
    image_shape,
):
    """
    Construct a serializable dictionary containing detection results.
    """
    return {
        "image_path": image_path,
        "image_shape": {"height": int(image_shape[0]), "width": int(image_shape[1])},
        "lines": [
            {
                "polygon": polygon,
                "confidence": float(conf),
                "bbox": {
                    "xmin": int(box[0]),
                    "ymin": int(box[1]),
                    "xmax": int(box[2]),
                    "ymax": int(box[3]),
                },
            }
            for polygon, conf, box in zip(
                _serialize_polygons(line_polygons), line_confs, line_boxes
            )
        ],
        "regions": [
            {
                "polygon": polygon,
                "confidence": float(conf),
                "bbox": {
                    "xmin": int(box[0]),
                    "ymin": int(box[1]),
                    "xmax": int(box[2]),
                    "ymax": int(box[3]),
                },
            }
            for polygon, conf, box in zip(
                _serialize_polygons(region_polygons), region_confs, region_boxes
            )
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect text line and paragraph polygons with RF-DETR."
    )
    parser.add_argument(
        "--detection_model_path",
        type=str,
        default="./models/rfdetr_text_seg_model.pth",
        help="Path to the RF-DETR detection model weights.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to an image file or a folder containing images.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for filtering detections.",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=768,
        help="Maximum image dimension for preprocessing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./detections.json",
        help="Path to save results as JSON. Use '-' to print to stdout.",
    )
    parser.add_argument(
        "--show_polygons",
        action="store_true",
        help="Show each image with detected polygons overlaid.",
    )
    return parser.parse_args()


def show_polygons(
    image_path: str,
    line_polygons: Sequence[Sequence[Sequence[int]]],
    region_polygons: Sequence[Sequence[Sequence[int]]],
):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for polygon in line_polygons:
        draw.polygon(
            [tuple(point) for point in polygon],
            outline="red",
            width=2,
        )
    for polygon in region_polygons:
        draw.polygon(
            [tuple(point) for point in polygon],
            outline="blue",
            width=3,
        )

    image.show()

def detect_polygons_from_image(
    image_path: Union[str, Path],
    confidence_threshold: float = 0.15,
    max_size: int = 768,
):
    """
    Run polygon detection on the image located at the provided filesystem path.
    """
    model = load_rfdetr_model(MODEL_PATH)
    (
        line_polygons,
        line_confs,
        line_boxes,
        region_polygons,
        region_confs,
        region_boxes,
        image_shape,
    ) = predict_polygons(
        model,
        str(image_path),
        max_size=max_size,
        confidence_threshold=confidence_threshold,
    )

    line_polygons, line_confs, line_boxes = _order_detections(
        line_polygons, line_confs, line_boxes
    )
    region_polygons, region_confs, region_boxes = _order_detections(
        region_polygons, region_confs, region_boxes
    )

    return {
        "line_polygons": line_polygons,
        "line_confs": line_confs,
        #"line_boxes": line_boxes,
        "region_polygons": region_polygons,
        #"region_confs": region_confs,
        #"region_boxes": region_boxes,
        "image_shape": image_shape
    }

def main():
    args = parse_args()
    model = load_rfdetr_model(args.detection_model_path)

    input_path = Path(args.input)
    if input_path.is_dir():
        images = load_image_paths(str(input_path))
    else:
        images = [str(input_path)]

    outputs = []
    for image_path in images:
        (
            line_polygons,
            line_confs,
            line_boxes,
            region_polygons,
            region_confs,
            region_boxes,
            image_shape,
        ) = predict_polygons(
            model,
            image_path,
            max_size=args.max_size,
            confidence_threshold=args.confidence_threshold,
        )

        line_polygons, line_confs, line_boxes = _order_detections(
            line_polygons, line_confs, line_boxes
        )
        region_polygons, region_confs, region_boxes = _order_detections(
            region_polygons, region_confs, region_boxes
        )
        outputs.append(
            _build_output(
                image_path,
                line_polygons,
                line_confs,
                line_boxes,
                region_polygons,
                region_confs,
                region_boxes,
                image_shape,
            )
        )
        if args.show_polygons:
            show_polygons(
                image_path,
                _serialize_polygons(line_polygons),
                _serialize_polygons(region_polygons),
            )

    if args.output == "-":
        print(json.dumps(outputs, indent=2))
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2)
        print(f"Saved detection results to {output_path}")


if __name__ == "__main__":
    main()

