import json
from drr_pipeline import negatives as N


def _data():
    e = lambda cats: {"categories": cats, "findings": {k: "x" for k in cats}}
    return {"train": {"train_1_a_1.nii.gz": e({"0": "2e"}), "train_2_a_1.nii.gz": e({"0": "1a"}),
                      "train_3_a_1.nii.gz": e({"0": "1a"}), "train_4_a_1.nii.gz": e({"0": "1a"}),
                      "train_5_a_1.nii.gz": e({}), "train_6_a_1.nii.gz": e({"0": "2g"})}}


def _rep(*sent):
    return {"Findings": {"Lungs/Airways/Pleura": list(sent)}}


def test_tiers():
    reports = {"train": {"train_2_a.nii.gz": _rep("There is a small right pleural effusion."),
                         "train_3_a.nii.gz": _rep("No pleural effusion is seen."),
                         "train_4_a.nii.gz": _rep("The lungs are clear."),
                         "train_5_a.nii.gz": _rep("No effusion, but consolidation in the left lower lobe.")}}
    rows = {r["volume_name"].split("_a_")[0]: r for r in N.build(_data(), reports, "pleural_effusion")}
    assert rows["train_1"]["tier"] == N.POSITIVE
    assert rows["train_2"]["tier"] == N.EXCL_MENTION
    assert rows["train_3"]["tier"] == N.NEG_CONFIRMED
    assert rows["train_4"]["tier"] == N.NEG_UNMENTIONED
    assert rows["train_5"]["tier"] == N.EXCL_MENTION             # phủ định lẫn mệnh đề khác ("but"): loại cho chắc
    assert rows["train_5"]["n_findings"] == 0
    assert rows["train_6"]["tier"] == N.NEG_UNMENTIONED          # có finding 2g nhưng không phải dấu hiệu tràn dịch
    reports["train"]["train_3_a.nii.gz"] = _rep("No effusion, but consolidation in the left lower lobe.")
    assert N.build(_data(), reports, "consolidation")[2]["tier"] == N.EXCL_MENTION  # phủ định lẫn mệnh đề khác: loại
    s = N.summarize(N.build(_data(), reports, "pleural_effusion"))["train"]
    assert s["ct"] == 6 and s["ct_zero_findings"] == 1 and s["tiers"][N.EXCL_MENTION] == 3
