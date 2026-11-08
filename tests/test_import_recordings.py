"""Filename handling matters for the uploaded Korean M4A archives."""
import unittest
import unicodedata
import zipfile

from src.import_recordings import decoded_name, infer_label


class RecordingImportTests(unittest.TestCase):
    def test_cp949_zip_names_are_decoded(self):
        info = zipfile.ZipInfo('방15o대기.m4a'.encode('cp949').decode('cp437'))
        self.assertEqual(decoded_name(info), '방15o대기.m4a')

    def test_decomposed_unicode_and_overlapping_keywords(self):
        self.assertEqual(infer_label(unicodedata.normalize('NFD', '화자_재시작_자택.m4a')), '재시작')
        self.assertEqual(infer_label('화자_시작_자택.m4a'), '시작')

    def test_all_uploaded_silence_naming_styles(self):
        for name in ['방소xSi.m4a', '복도소oSi.m4a', '거리 - silence.m4a', '화자_무음_집.m4a']:
            self.assertEqual(infer_label(name), 'silence')

    def test_unknown_and_ambiguous_labels_are_rejected(self):
        for name in ['unknown.m4a', '시작_출발.m4a']:
            with self.assertRaises(ValueError):
                infer_label(name)


if __name__ == '__main__':
    unittest.main()
