# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014-15, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import datetime
import os
import pandas
import tempfile
import shutil
import unittest

import nab.corpus
import nab.labeler
from nab.util import strp
from nab.test_helpers import writeCorpus, writeCorpusLabel, generateTimestamps



class CorpusLabelTest(unittest.TestCase):


  def setUp(self):
    self.tempDir = os.path.join(tempfile.mkdtemp(), "test")
    self.tempCorpusPath = os.path.join(self.tempDir, "data")
    self.tempCorpusLabelPath = os.path.join(
      self.tempDir, "labels", "label.json")


  def tearDown(self):
    shutil.rmtree(self.tempDir)


  def testWindowTimestampsNotInDataFileThrowsError(self):
    """
    A ValueError should be thrown when label windows contain timestamps
    that do no exist in the data file.
    """
    data = pandas.DataFrame({"timestamp" :
      generateTimestamps(strp("2014-01-01"), None, 1)})

    windows = [["2015-01-01", "2015-01-01"]]

    writeCorpus(self.tempCorpusPath, {"test_data_file.csv" : data})
    writeCorpusLabel(self.tempCorpusLabelPath, {"test_data_file.csv": windows})

    corpus = nab.corpus.Corpus(self.tempCorpusPath)

    self.assertRaises(ValueError,
      nab.labeler.CorpusLabel, self.tempCorpusLabelPath, corpus)


  def testWindowTimestampsNonChronologicalThrowsError(self):
    """
    A ValueError should be thrown when a label window's start and end
    times are not in chronological order.
    """
    data = pandas.DataFrame({"timestamp" :
      generateTimestamps(strp("2014-01-01"),
      datetime.timedelta(minutes=5), 10)})

    # Windows both in and out of order
    windows = [["2014-01-01 00:45", "2014-01-01 00:00"],
               ["2014-01-01 10:15", "2014-01-01 11:15"]]

    writeCorpus(self.tempCorpusPath, {"test_data_file.csv" : data})
    writeCorpusLabel(self.tempCorpusLabelPath, {"test_data_file.csv": windows})

    corpus = nab.corpus.Corpus(self.tempCorpusPath)

    self.assertRaises(
      ValueError, nab.labeler.CorpusLabel, self.tempCorpusLabelPath, corpus)


  def testRowsLabeledAnomalousWithinAWindow(self):
    """
    All timestamps labeled as anomalous should be within a label window.
    """
    data = pandas.DataFrame({"timestamp" :
      generateTimestamps(strp("2014-01-01"),
      datetime.timedelta(minutes=5), 10)})

    windows = [["2014-01-01 00:15", "2014-01-01 00:30"]]

    writeCorpus(self.tempCorpusPath, {"test_data_file.csv": data})
    writeCorpusLabel(self.tempCorpusLabelPath, {"test_data_file.csv": windows})

    corpus = nab.corpus.Corpus(self.tempCorpusPath)

    corpusLabel = nab.labeler.CorpusLabel(self.tempCorpusLabelPath, corpus)

    for relativePath, lab in corpusLabel.labels.items():
      windows = corpusLabel.windows[relativePath]

      for row in lab[lab["label"] == 1].iterrows():
        self.assertTrue(
          all([w[0] <= row[1]["timestamp"] <= w[1] for w in windows]),
            "The label at %s of file %s is not within a label window"
            % (row[1]["timestamp"], relativePath))


  def testNonexistentDatafileForLabelsThrowsError(self):
    data = pandas.DataFrame({"timestamp" :
      generateTimestamps(strp("2014-01-01"),
      datetime.timedelta(minutes=5), 10)})

    windows = [["2014-01-01 00:15", "2014-01-01 00:30"]]

    writeCorpus(self.tempCorpusPath, {"test_data_file.csv": data})
    writeCorpusLabel(self.tempCorpusLabelPath,
      {"test_data_file.csv": windows, "non_existent_data_file.csv": windows})

    corpus = nab.corpus.Corpus(self.tempCorpusPath)

    self.assertRaises(
      KeyError, nab.labeler.CorpusLabel, self.tempCorpusLabelPath, corpus)


  def testGetLabels(self):
    """
    Labels dictionary generated by CorpusLabel.getLabels() should match the
    label windows.
    """
    data = pandas.DataFrame({"timestamp" :
      generateTimestamps(strp("2014-01-01"),
      datetime.timedelta(minutes=5), 10)})

    windows = [["2014-01-01 00:00", "2014-01-01 00:10"],
               ["2014-01-01 00:10", "2014-01-01 00:15"]]

    writeCorpus(self.tempCorpusPath, {"test_data_file.csv" : data})
    writeCorpusLabel(self.tempCorpusLabelPath, {"test_data_file.csv": windows})

    corpus = nab.corpus.Corpus(self.tempCorpusPath)

    corpusLabel = nab.labeler.CorpusLabel(self.tempCorpusLabelPath, corpus)

    for relativePath, l in corpusLabel.labels.items():
      windows = corpusLabel.windows[relativePath]

      for t, lab in corpusLabel.labels["test_data_file.csv"].values:
        for w in windows:
          if (w[0] <= t and t <= w[1]):
            self.assertEqual(lab, 1,
              "Incorrect label value for timestamp %r" % t)


  def testRedundantTimestampsRaiseException(self):
    data = pandas.DataFrame({"timestamp" :
      generateTimestamps(strp("2015-01-01"),
      datetime.timedelta(days=1), 365)})
    dataFileName = "test_data_file.csv"
    writeCorpus(self.tempCorpusPath, {dataFileName : data})

    labels = ["2015-12-25 00:00:00",
              "2015-12-26 00:00:00",
              "2015-12-31 00:00:00"]
    labelsDir = self.tempCorpusLabelPath.replace(
      "/label.json", "/raw/label.json")
    writeCorpusLabel(labelsDir, {dataFileName: labels})

    corpus = nab.corpus.Corpus(self.tempCorpusPath)
    labDir = labelsDir.replace("/label.json", "")
    labelCombiner = nab.labeler.LabelCombiner(
      labDir, corpus, 0.5, 0.10, 0.15, 0)

    self.assertRaises(ValueError, labelCombiner.combine)


  def testBucketMerge(self):
    data = pandas.DataFrame({"timestamp" :
      generateTimestamps(strp("2015-12-01"),
      datetime.timedelta(days=1), 31)})
    dataFileName = "test_data_file.csv"
    writeCorpus(self.tempCorpusPath, {dataFileName : data})

    rawLabels = (["2015-12-24 00:00:00",
                  "2015-12-31 00:00:00"],
                 ["2015-12-01 00:00:00",
                  "2015-12-25 00:00:00",
                  "2015-12-31 00:00:00"],
                 ["2015-12-25 00:00:00"])

    for i, labels in enumerate(rawLabels):
      labelsPath = self.tempCorpusLabelPath.replace(
        os.path.sep+"label.json", os.path.sep+"raw"+os.path.sep+"label{}.json".format(i))
      writeCorpusLabel(labelsPath, {"test_data_file.csv": labels})
    labelsDir = labelsPath.replace(os.path.sep+"label{}.json".format(i), "")

    corpus = nab.corpus.Corpus(self.tempCorpusPath)
    labelCombiner = nab.labeler.LabelCombiner(
      labelsDir, corpus, 0.5, 0.10, 0.15, 0)
    labelCombiner.getRawLabels()
    labelTimestamps, _ = labelCombiner.combineLabels()

    expectedLabels = ['2015-12-25 00:00:00', '2015-12-31 00:00:00']
    self.assertEqual(expectedLabels, labelTimestamps[dataFileName],
      "The combined labels did not bucket and merge as expected.")



if __name__ == '__main__':
  unittest.main()