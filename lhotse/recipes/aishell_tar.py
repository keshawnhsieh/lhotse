"""
About the Aishell corpus
Aishell is an open-source Chinese Mandarin speech corpus published by Beijing Shell Shell Technology Co.,Ltd.
publicly available on https://www.openslr.org/33
"""

import logging
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm
from multiprocessing import Pool

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, resumable_download, safe_extract

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

def text_normalize(line: str):
    """
    Modified from https://github.com/wenet-e2e/wenet/blob/main/examples/multi_cn/s0/local/aishell_data_prep.sh#L54
    sed 's/ａ/a/g' | sed 's/ｂ/b/g' |\
    sed 's/ｃ/c/g' | sed 's/ｋ/k/g' |\
    sed 's/ｔ/t/g' > $dir/transcripts.t

    """
    line = line.replace("ａ", "a")
    line = line.replace("ｂ", "b")
    line = line.replace("ｃ", "c")
    line = line.replace("ｋ", "k")
    line = line.replace("ｔ", "t")
    line = line.upper()
    return line


def download_aishell(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "http://www.openslr.org/resources",
) -> Path:
    """
    Downdload and untar the dataset
    :param target_dir: Pathlike, the path of the dir to storage the dataset.
    :param force_download: Bool, if True, download the tars no matter if the tars exist.
    :param base_url: str, the url of the OpenSLR resources.
    :return: the path to downloaded and extracted directory with data.
    """
    url = f"{base_url}/33"
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "aishell"
    dataset_tar_name = "data_aishell.tgz"
    resources_tar_name = "resource_aishell.tgz"
    for tar_name in [dataset_tar_name, resources_tar_name]:
        tar_path = target_dir / tar_name
        extracted_dir = corpus_dir / tar_name[:-4]
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            logging.info(
                f"Skipping download of {tar_name} because {completed_detector} exists."
            )
            continue
        resumable_download(
            f"{url}/{tar_name}", filename=tar_path, force_download=force_download
        )
        shutil.rmtree(extracted_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            safe_extract(tar, path=corpus_dir)
        if tar_name == dataset_tar_name:
            wav_dir = extracted_dir / "wav"
            for sub_tar_name in os.listdir(wav_dir):
                with tarfile.open(wav_dir / sub_tar_name) as tar:
                    safe_extract(tar, path=wav_dir)
        completed_detector.touch()

    return corpus_dir


def prepare_aishell_tar(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    # transcript_path = corpus_dir / "data_aishell/transcript/aishell_transcript_v0.8.txt"
    # transcript_dict = {}
    # with open(transcript_path, "r", encoding="utf-8") as f:
    #     for line in f.readlines():
    #         idx_transcript = line.split()
    #         content = " ".join(idx_transcript[1:])
    #         content = text_normalize(content)
    #         transcript_dict[idx_transcript[0]] = content
    manifests = defaultdict(dict)
    dataset_parts = ["train_l"]#, "dev", "test"]
    for part in tqdm(
        dataset_parts,
        desc="Process aishell audio, it takes about 102 seconds.",
    ):
        logging.info(f"Processing aishell subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []

        tar_path = corpus_dir / f"{part}"
        for tar in tqdm(sorted(tar_path.rglob("*.tar"))):
            ftar = open(tar, 'rb')
            stream = tarfile.open(fileobj=ftar, mode="r|*")
            prev_prefix = None
            example = {}
            valid = True
            for i, tarinfo in enumerate(stream):
                name = tarinfo.name
                pos = name.rfind('.')
                assert pos > 0
                prefix, postfix = name[:pos], name[pos + 1:]
                if prev_prefix is not None and prefix != prev_prefix:
                    example['key'] = prev_prefix
                    if valid:
                        recording = example["recording"]
                        segment = example["segment"]
                        recordings.append(recording)
                        supervisions.append(segment)
                    example = {}
                    valid = True

                with stream.extractfile(tarinfo) as file_obj:

                    if postfix == 'txt':
                        text = file_obj.read().decode('utf8').strip()
                        recording = example["recording"]
                        example["segment"] = SupervisionSegment(
                            id=prefix,
                            recording_id=prefix,
                            start=0.0,
                            duration=recording.duration,
                            channel=0,
                            language="Chinese",
                            speaker="_".join(prefix.split("_")[:-1]),
                            text=text.strip(),
                        )
                    elif postfix in AUDIO_FORMAT_SETS:
                        example["recording"] = Recording.from_tar(file_obj.read(), recording_id=prefix, tar_path=str(tar), tar_idx=i)
                    else:
                        example[postfix] = file_obj.read()

                prev_prefix = prefix
            if prev_prefix is not None:
                recording = example["recording"]
                segment = example["segment"]
                recordings.append(recording)
                supervisions.append(segment)
            stream.close()
            ftar.close()

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"aishell_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"aishell_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests

def helper(tar):
    recordings = []
    supervisions = []
    ftar = open(tar, 'rb')
    stream = tarfile.open(fileobj=ftar, mode="r|*")
    prev_prefix = None
    example = {}
    valid = True
    for i, tarinfo in enumerate(stream):
        name = tarinfo.name
        pos = name.rfind('.')
        assert pos > 0
        prefix, postfix = name[:pos], name[pos + 1:]
        if prev_prefix is not None and prefix != prev_prefix:
            example['key'] = prev_prefix
            if valid:
                recording = example["recording"]
                segment = example["segment"]
                recordings.append(recording)
                supervisions.append(segment)
            example = {}
            valid = True

        with stream.extractfile(tarinfo) as file_obj:

            if postfix == 'txt':
                text = file_obj.read().decode('utf8').strip()
                recording = example["recording"]
                example["segment"] = SupervisionSegment(
                    id=prefix,
                    recording_id=prefix,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="Chinese",
                    speaker="_".join(prefix.split("_")[:-1]),
                    text=text.strip(),
                )
            elif postfix in AUDIO_FORMAT_SETS:
                example["recording"] = Recording.from_tar(file_obj.read(), recording_id=prefix, tar_path=str(tar),
                                                          tar_idx=i)
            else:
                example[postfix] = file_obj.read()

        prev_prefix = prefix
    if prev_prefix is not None:
        recording = example["recording"]
        segment = example["segment"]
        recordings.append(recording)
        supervisions.append(segment)
    stream.close()
    ftar.close()
    return recordings, supervisions

def prepare_aishell_tar_mp(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    # transcript_path = corpus_dir / "data_aishell/transcript/aishell_transcript_v0.8.txt"
    # transcript_dict = {}
    # with open(transcript_path, "r", encoding="utf-8") as f:
    #     for line in f.readlines():
    #         idx_transcript = line.split()
    #         content = " ".join(idx_transcript[1:])
    #         content = text_normalize(content)
    #         transcript_dict[idx_transcript[0]] = content
    manifests = defaultdict(dict)
    dataset_parts = ["train_l"]#, "dev", "test"]
    for part in tqdm(
        dataset_parts,
        desc="Process aishell audio, it takes about 102 seconds.",
    ):
        logging.info(f"Processing aishell subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)


        tar_path = corpus_dir / f"{part}"
        all_tars = sorted(tar_path.rglob("*.tar"))[:100]
        # with Pool() as pool:
        #     result = list(tqdm(pool.imap(helper, all_tars), total=len(all_tars)))
        result = []
        for tar in tqdm(all_tars):
            result.append(helper(tar))
        recordings = [r[0] for r in result]
        supervisions = [r[1] for r in result]

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"aishell_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"aishell_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests