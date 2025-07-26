# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3

import csv
import os

import datasets

_CITATION = """\
@article{jeon2022user,
  title={User Guide for KOTE: Korean Online Comments Emotions Dataset},
  author={Jeon, Duyoung and Lee, Junho and Kim, Cheongtag},
  journal={arXiv preprint arXiv:2205.05300},
  year={2022}
}
"""

_DESCRIPTION = """\
50k Korean online comments labeled for 44 emotion categories.
"""

_HOMEPAGE = "https://github.com/searle-j/KOTE"

_LICENSE = "MIT License"

_BASE_URL = "https://raw.githubusercontent.com/searle-j/KOTE/main/"

_LABELS = [
'불평/불만',
 '환영/호의',
 '감동/감탄',
 '지긋지긋',
 '고마움',
 '슬픔',
 '화남/분노',
 '존경',
 '기대감',
 '우쭐댐/무시함',
 '안타까움/실망',
 '비장함',
 '의심/불신',
 '뿌듯함',
 '편안/쾌적',
 '신기함/관심',
 '아껴주는',
 '부끄러움',
 '공포/무서움',
 '절망',
 '한심함',
 '역겨움/징그러움',
 '짜증',
 '어이없음',
 '없음',
 '패배/자기혐오',
 '귀찮음',
 '힘듦/지침',
 '즐거움/신남',
 '깨달음',
 '죄책감',
 '증오/혐오',
 '흐뭇함(귀여움/예쁨)',
 '당황/난처',
 '경악',
 '부담/안_내킴',
 '서러움',
 '재미없음',
 '불쌍함/연민',
 '놀람',
 '행복',
 '불안/걱정',
 '기쁨',
 '안심/신뢰'
]

class KOTEConfig(datasets.BuilderConfig):
    @property
    def features(self):
        if self.name == "dichotomized":
            return {
                "ID": datasets.Value("string"),
                "text": datasets.Value("string"),
                "labels": datasets.Sequence(datasets.ClassLabel(names=_LABELS)),
            }

class KOTE(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [KOTEConfig(name="dichotomized")]
    BUILDER_CONFIG_CLASS = KOTEConfig
    DEFAULT_CONFIG_NAME = "dichotomized"
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(self.config.features),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
    
    
    def _split_generators(self, dl_manager):
        if self.config.name=="dichotomized":
            train_path = dl_manager.download_and_extract(os.path.join(_BASE_URL, "train.tsv"))
            test_path = dl_manager.download_and_extract(os.path.join(_BASE_URL, "test.tsv"))
            val_path = dl_manager.download_and_extract(os.path.join(_BASE_URL, "val.tsv"))
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": [train_path],}),
                datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": [test_path],}),
                datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": [val_path],}),
            ]
            
    def _generate_examples(self, filepaths):
        if self.config.name=="dichotomized":
            for filepath in filepaths:
                with open(filepath, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="\t", fieldnames=list(self.config.features.keys()))
                    for idx, row in enumerate(reader):
                        row["labels"] = [int(lab) for lab in row["labels"].split(",")]
                        yield idx, row