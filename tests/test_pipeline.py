from aa.annotate import classify_status
from aa.export import split_images
from aa.fetch import odgt_to_coco_annotations


def test_classify_status():
    assert classify_status(0.9, 0.5, 0.25) == "auto"
    assert classify_status(0.5, 0.5, 0.25) == "auto"
    assert classify_status(0.3, 0.5, 0.25) == "needs_review"
    assert classify_status(0.1, 0.5, 0.25) == "discard"


def test_odgt_conversion():
    rec = {
        "ID": "img1",
        "gtboxes": [
            {"tag": "person", "fbox": [10, 20, 30, 60], "hbox": [15, 20, 10, 10]},
            {"tag": "mask", "fbox": [0, 0, 5, 5]},  # ignore 領域
        ],
    }
    anns = odgt_to_coco_annotations(rec, image_id=1, start_ann_id=1)
    persons = [a for a in anns if a["category_id"] == 1]
    heads = [a for a in anns if a["category_id"] == 2]
    assert len(persons) == 2
    assert len(heads) == 1  # mask の頭部は出力しない
    assert persons[0]["iscrowd"] == 0
    assert persons[1]["iscrowd"] == 1  # ignore 領域は iscrowd=1
    assert [a["id"] for a in anns] == [1, 2, 3]


def test_split_images_deterministic():
    images = [{"id": i} for i in range(10)]
    s1 = split_images(images, 0.9)
    s2 = split_images(images, 0.9)
    assert s1 == s2
    assert len(s1["train"]) == 9
    assert len(s1["val"]) == 1
